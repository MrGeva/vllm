# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL MiniMax M3 index decode score with the exact top-16 block select
split into a per-CTA epilogue and a one-warp-per-row merge.

The score pass is ``IndexDecodeScoreKernel``'s (same TMA + ``mma.sync`` math and
k-order, so every block score is bit-identical and is still written to the
unified score buffer). It replaces the ``sparse_topk_select`` pass that re-read
and histogram-sorted every score row:

- Epilogue: while scoring, each of a request's ``SPLIT_K`` CTAs keeps, per
  (token, head) row, its top-3 eligible (non-forced, causally valid) block
  scores as order-preserving keys plus their count, and publishes that
  16-byte record.
- Merge (``IndexTopKFromRecordsKernel``, PDL-launched, one warp per row): the
  row's ``SPLIT_K`` records are read in one round. The (16 - F)'th largest of 32
  group maxima (CTA ``j`` in group ``j % 32``) of the CTA maxima is a lower
  bound on the (16 - F)'th largest eligible score (F forced init/local blocks),
  so only record keys reaching it are candidates. A CTA whose 3rd key still
  reaches the bound is rescanned from the score row; more than 64 candidates
  fall back to an exact scan of the row. One shuffle pass ranks the
  candidates exactly (ties: the higher block first) and places the kept ones
  in ascending block order.
- The output matches ``sparse_topk_select``: the forced blocks plus the top
  16 - F blocks in ascending order; rows with at most 16 valid pages get
  ``0..nvp-1`` then -1.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import (
    _TORCH_TO_CUTE_DTYPE,
    EVICT_FIRST,
    mma_sync,
    simple_tma_copy,
)
from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
    IndexDecodeScoreKernel,
    _fp8_to_f16_mma_fragments,
)

TOPK = 16
# CTAs per request (the stock decode score kernel's split).
SPLIT_K = 256
# Score tile dim of the unified buffer (indexer_msa.MAX_K_TILES).
MAX_K_TILES = 8192
# Max rows (index heads x decode query length) of one request.
MAX_ROWS = 32
# Per-CTA record of one row: top-3 keys, then slots (8 bits each) | count << 24.
REC_WORDS = 4
REC_TOP = 3
# Candidate blocks a row's fast path ranks.
CAND_CAP = 64
# Record groups per row: CTA j belongs to lane j % 32.
GROUPS = 32
CTAS_PER_LANE = SPLIT_K // GROUPS
# Warps of the score kernel: TMA warp 4, MMA warps 0-3.
SCORE_WARPS = 5


@dsl_user_op
def _f32_key(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Order-preserving uint32 key of a finite float (-0 == +0)."""
    out = llvm.inline_asm(
        T.i32(),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b, m;\n\t"
        ".reg .pred z, n;\n\t"
        "mov.b32 b, $1;\n\t"
        "and.b32 m, b, 0x7fffffff;\n\t"
        "setp.eq.u32 z, m, 0;\n\t"
        "selp.b32 b, 0, b, z;\n\t"
        "setp.lt.s32 n, b, 0;\n\t"
        "not.b32 m, b;\n\t"
        "or.b32 b, b, 0x80000000;\n\t"
        "selp.b32 $0, m, b, n;\n\t"
        "}\n",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def _ld_f32_key(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """``_f32_key`` of the float at ``addr`` (coherent at GPU scope)."""
    out = llvm.inline_asm(
        T.i32(),
        [addr.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 b, m;\n\t"
        ".reg .pred z, n;\n\t"
        "ld.relaxed.gpu.global.b32 b, [$1];\n\t"
        "and.b32 m, b, 0x7fffffff;\n\t"
        "setp.eq.u32 z, m, 0;\n\t"
        "selp.b32 b, 0, b, z;\n\t"
        "setp.lt.s32 n, b, 0;\n\t"
        "not.b32 m, b;\n\t"
        "or.b32 b, b, 0x80000000;\n\t"
        "selp.b32 $0, m, b, n;\n\t"
        "}\n",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def _ld_v4_u32(
    addr: Int64, *, loc=None, ip=None
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _st_v4_u32(
    addr: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        T.i32(),
        [addr.ir_value(loc=loc, ip=ip)]
        + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "mov.u32 $0, 0;\n\tst.global.v4.u32 [$1], {$2, $3, $4, $5};",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _warp_sum(x: Int32) -> Int32:
    for i in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=1 << i)
    return x


@cute.jit
def _warp_max_u32(x: Uint32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(x, offset=1 << i)
        x = other if other > x else x
    return x


@cute.jit
def _warp_exclusive_sum(x: Int32, lane: Int32) -> Int32:
    inc = x
    for i in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(inc, offset=1 << i)
        inc = inc + (other if lane >= (1 << i) else Int32(0))
    return inc - x


class _RowSelect:
    """Exact top-16 of one (token, head) row from its CTA records (one warp)."""

    force_begin: int
    force_end: int

    @cute.jit
    def load_records(
        self,
        rec_base: Int64,
        rec_row: Int64,
        n_active: Int32,
        lane: Int32,
        recs: cute.Tensor,
    ):
        for i in cutlass.range_constexpr(CTAS_PER_LANE):
            j = lane + i * GROUPS
            recs[i, 0] = Uint32(0)
            recs[i, 1] = Uint32(0)
            recs[i, 2] = Uint32(0)
            recs[i, 3] = Uint32(0)
            if j < n_active:
                a, b, c, d = _ld_v4_u32(rec_base + (rec_row + Int64(j)) * 16)
                recs[i, 0] = a
                recs[i, 1] = b
                recs[i, 2] = c
                recs[i, 3] = d

    @cute.jit
    def select_row(
        self,
        score: cute.Tensor,
        topk_out: cute.Tensor,
        recs: cute.Tensor,
        s_cand: cute.Tensor,
        widx: Int32,
        t: Int32,
        head_id: Int32,
        nvp: Int32,
        lane: Int32,
    ):
        FB = self.force_begin
        FE = self.force_end
        SEL = TOPK - FB - FE
        hi = nvp - FE
        lane_lt = (Uint32(1) << Uint32(lane)) - Uint32(1)
        row_off = cute.crd2idx((head_id, t, Int32(0)), score.layout)
        row_addr = score.iterator.toint() + Int64(row_off) * 4

        # lo = SEL'th largest of the 32 group maxima (ties ranked by lane).
        lane_max = recs[0, 0]
        for i in cutlass.range_constexpr(1, CTAS_PER_LANE):
            lane_max = recs[i, 0] if recs[i, 0] > lane_max else lane_max
        rank = Int32(0)
        for m in cutlass.range_constexpr(GROUPS):
            other = cute.arch.shuffle_sync(lane_max, offset=m)
            ahead = other > lane_max or (other == lane_max and m < lane)
            rank = rank + (Int32(1) if ahead else Int32(0))
        lo = cute.arch.warp_redux_sync(
            lane_max if rank == SEL - 1 else Uint32(0), kind="umax"
        )
        lo = lo if lo > Uint32(0) else Uint32(1)

        # Record keys >= lo are the candidates. A CTA whose 3rd key reaches lo
        # with more eligible blocks overflows (its blocks are rescanned).
        mask = Uint32(0)
        ovf = Uint32(0)
        for i in cutlass.range_constexpr(CTAS_PER_LANE):
            for q in cutlass.range_constexpr(REC_TOP):
                if recs[i, q] >= lo:
                    mask = mask | Uint32(1 << (REC_TOP * i + q))
            if recs[i, 2] >= lo and (recs[i, 3] >> Uint32(24)) > Uint32(REC_TOP):
                ovf = ovf | Uint32(1 << i)
        n_mine = Int32(cute.arch.popc(mask))
        n_cand = cute.arch.warp_redux_sync(n_mine, kind="add")
        any_ovf = cute.arch.warp_redux_sync(ovf, kind="or")
        if any_ovf == Uint32(0) and n_cand <= GROUPS:
            off = _warp_exclusive_sum(n_mine, lane)
            for i in cutlass.range_constexpr(CTAS_PER_LANE):
                for q in cutlass.range_constexpr(REC_TOP):
                    bit = REC_TOP * i + q
                    if (mask >> Uint32(bit)) & Uint32(1) != Uint32(0):
                        pos = off + Int32(cute.arch.popc(mask & Uint32((1 << bit) - 1)))
                        sl = (recs[i, 3] >> Uint32(8 * q)) & Uint32(0xFF)
                        blk = Uint32(lane + i * GROUPS) + sl * Uint32(SPLIT_K)
                        s_cand[widx, pos] = (Uint64(recs[i, q]) << Uint64(32)) | Uint64(
                            blk
                        )
        else:
            n_cand = self.gather_overflow(
                row_addr, recs, mask, ovf, s_cand, widx, lo, hi, lane, lane_lt
            )
        cute.arch.sync_warp()
        if n_cand <= CAND_CAP and n_cand >= SEL:
            self.rank_and_write(topk_out, s_cand, widx, n_cand, t, head_id, nvp, lane)
        else:
            self.select_row_exact(topk_out, row_addr, t, head_id, nvp, lane, lane_lt)

    @cute.jit
    def gather_overflow(
        self,
        row_addr: Int64,
        recs: cute.Tensor,
        mask: Uint32,
        ovf: Uint32,
        s_cand: cute.Tensor,
        widx: Int32,
        lo: Uint32,
        hi: Int32,
        lane: Int32,
        lane_lt: Uint32,
    ) -> Int32:
        """Candidates (up to CAND_CAP) when some CTA overflowed or more than 32
        record keys reach lo: record keys of the other CTAs, every eligible
        block >= lo of an overflowing CTA (its blocks loaded by the warp)."""
        FB = self.force_begin
        n = Int32(0)
        for i in cutlass.range_constexpr(CTAS_PER_LANE):
            keep = mask
            if (ovf >> Uint32(i)) & Uint32(1) != Uint32(0):
                keep = mask & ~Uint32(((1 << REC_TOP) - 1) << (REC_TOP * i))
            for q in cutlass.range_constexpr(REC_TOP):
                bit = REC_TOP * i + q
                take = (keep >> Uint32(bit)) & Uint32(1) != Uint32(0)
                bal = Uint32(cute.arch.vote_ballot_sync(take))
                if take:
                    p = n + Int32(cute.arch.popc(bal & lane_lt))
                    if p < CAND_CAP:
                        sl = (recs[i, 3] >> Uint32(8 * q)) & Uint32(0xFF)
                        blk = Uint32(lane + i * GROUPS) + sl * Uint32(SPLIT_K)
                        s_cand[widx, p] = (Uint64(recs[i, q]) << Uint64(32)) | Uint64(
                            blk
                        )
                n = n + Int32(cute.arch.popc(bal))
            # Overflowing CTAs of this record slot, one per warp-wide rescan.
            pending = Uint32(
                cute.arch.vote_ballot_sync((ovf >> Uint32(i)) & Uint32(1) != Uint32(0))
            )
            for _ in range(Int32(cute.arch.popc(pending))):
                src = Int32(cute.arch.bfind(pending & (~pending + Uint32(1))))
                pending = pending & (pending - Uint32(1))
                blk = src + i * GROUPS + lane * SPLIT_K
                kv = Uint32(0)
                if blk >= FB and blk < hi:
                    kv = _ld_f32_key(row_addr + Int64(blk) * 4)
                take = kv >= lo
                bal = Uint32(cute.arch.vote_ballot_sync(take))
                if take:
                    p = n + Int32(cute.arch.popc(bal & lane_lt))
                    if p < CAND_CAP:
                        s_cand[widx, p] = (Uint64(kv) << Uint64(32)) | Uint64(blk)
                n = n + Int32(cute.arch.popc(bal))
        return n

    @cute.jit
    def rank_and_write(
        self,
        topk_out: cute.Tensor,
        s_cand: cute.Tensor,
        widx: Int32,
        n_cand: Int32,
        t: Int32,
        head_id: Int32,
        nvp: Int32,
        lane: Int32,
    ):
        """Keep the SEL largest candidate keys (key desc, block desc) and write
        them with the forced blocks in ascending block order: one shuffle pass
        ranks each key and records which candidates have a smaller block."""
        FB = self.force_begin
        FE = self.force_end
        SEL = TOPK - FB - FE
        ka = Uint64(0)
        if lane < n_cand:
            ka = s_cand[widx, lane]
        ba = Uint32((ka & Uint64(0xFFFFFFFF)).to(Uint32))
        ra = Int32(0)
        la = Uint32(0)
        if n_cand <= 32:
            for m in cutlass.range_constexpr(32):
                km = cute.arch.shuffle_sync(ka, offset=m)
                bm = Uint32((km & Uint64(0xFFFFFFFF)).to(Uint32))
                ra = ra + (Int32(1) if km > ka else Int32(0))
                la = la | (Uint32(1 << m) if bm < ba and km != Uint64(0) else Uint32(0))
            sa = lane < n_cand and ra < SEL
            sel_a = Uint32(cute.arch.vote_ballot_sync(sa))
            if sa:
                topk_out[t, head_id, FB + Int32(cute.arch.popc(la & sel_a))] = Int32(ba)
        else:
            kb = Uint64(0)
            if lane + 32 < n_cand:
                kb = s_cand[widx, lane + 32]
            bb = Uint32((kb & Uint64(0xFFFFFFFF)).to(Uint32))
            rb = Int32(0)
            lb = Uint32(0)
            la2 = Uint32(0)
            lb2 = Uint32(0)
            for m in cutlass.range_constexpr(32):
                km = cute.arch.shuffle_sync(ka, offset=m)
                bm = Uint32((km & Uint64(0xFFFFFFFF)).to(Uint32))
                ra = ra + (Int32(1) if km > ka else Int32(0))
                rb = rb + (Int32(1) if km > kb else Int32(0))
                la = la | (Uint32(1 << m) if bm < ba and km != Uint64(0) else Uint32(0))
                lb = lb | (Uint32(1 << m) if bm < bb and km != Uint64(0) else Uint32(0))
            for m in cutlass.range_constexpr(32):
                km = cute.arch.shuffle_sync(kb, offset=m)
                bm = Uint32((km & Uint64(0xFFFFFFFF)).to(Uint32))
                ra = ra + (Int32(1) if km > ka else Int32(0))
                rb = rb + (Int32(1) if km > kb else Int32(0))
                la2 = la2 | (
                    Uint32(1 << m) if bm < ba and km != Uint64(0) else Uint32(0)
                )
                lb2 = lb2 | (
                    Uint32(1 << m) if bm < bb and km != Uint64(0) else Uint32(0)
                )
            sa = lane < n_cand and ra < SEL
            sb = lane + 32 < n_cand and rb < SEL
            sel_a = Uint32(cute.arch.vote_ballot_sync(sa))
            sel_b = Uint32(cute.arch.vote_ballot_sync(sb))
            if sa:
                pos = FB + Int32(
                    cute.arch.popc(la & sel_a) + cute.arch.popc(la2 & sel_b)
                )
                topk_out[t, head_id, pos] = Int32(ba)
            if sb:
                pos = FB + Int32(
                    cute.arch.popc(lb & sel_a) + cute.arch.popc(lb2 & sel_b)
                )
                topk_out[t, head_id, pos] = Int32(bb)
        # Forced init blocks first, forced local blocks last.
        if lane < FB:
            topk_out[t, head_id, lane] = lane
        if lane >= 32 - FE:
            e = lane - (32 - FE)
            topk_out[t, head_id, FB + SEL + e] = nvp - FE + e

    @cute.jit
    def select_row_exact(
        self,
        topk_out: cute.Tensor,
        row_addr: Int64,
        t: Int32,
        head_id: Int32,
        nvp: Int32,
        lane: Int32,
        lane_lt: Uint32,
    ):
        """Exact select by scanning the full score row (rare fallback)."""
        FB = self.force_begin
        FE = self.force_end
        SEL = TOPK - FB - FE
        hi = nvp - FE
        # v = largest key with count(key >= v) >= SEL over the eligible range.
        v = Uint32(0)
        for it in range(32):
            probe = v | (Uint32(1) << Uint32(31 - it))
            c = Int32(0)
            for k in range(FB + lane, hi, 32):
                if _ld_f32_key(row_addr + Int64(k) * 4) >= probe:
                    c = c + 1
            if cute.arch.warp_redux_sync(c, kind="add") >= SEL:
                v = probe
        c = Int32(0)
        for k in range(FB + lane, hi, 32):
            if _ld_f32_key(row_addr + Int64(k) * 4) > v:
                c = c + 1
        need = SEL - cute.arch.warp_redux_sync(c, kind="add")
        # Among keys == v, the `need` highest blocks: largest i with
        # count(key == v, k >= i) >= need.
        i_thr = Int32(0)
        for it in range(14):
            probe_i = i_thr | (Int32(1) << (13 - it))
            c = Int32(0)
            for k in range(FB + lane, hi, 32):
                kv = _ld_f32_key(row_addr + Int64(k) * 4)
                if kv == v and k >= probe_i:
                    c = c + 1
            if cute.arch.warp_redux_sync(c, kind="add") >= need:
                i_thr = probe_i
        # Emit forced + selected blocks in ascending order.
        written = Int32(0)
        for b0 in range(0, nvp, 32):
            k = b0 + lane
            sel = False
            if k < nvp:
                if k < FB or k >= hi:
                    sel = True
                else:
                    kv = _ld_f32_key(row_addr + Int64(k) * 4)
                    sel = kv > v or (kv == v and k >= i_thr)
            bal = Uint32(cute.arch.vote_ballot_sync(sel))
            if sel:
                topk_out[t, head_id, written + Int32(cute.arch.popc(bal & lane_lt))] = k
            written = written + Int32(cute.arch.popc(bal))

    @cute.jit
    def select_request_rows(
        self,
        score: cute.Tensor,
        num_valid_pages: cute.Tensor,
        topk_out: cute.Tensor,
        rec_base: Int64,
        s_cand: cute.Tensor,
        batch_id: Int32,
        decode_query_len: Int32,
        n_active: Int32,
        widx: Int32,
        lane: Int32,
        row_first: Int32,
        row_step: cutlass.Constexpr[int],
        rows_per_warp: cutlass.Constexpr[int],
    ):
        """Select rows row_first + k * row_step (k < rows_per_warp) of a
        request. The page counts (not written by the score kernel) are read
        before the PDL wait, every row's records right after it."""
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = self.num_heads * MAX_DQL
        recs = cute.make_rmem_tensor((rows_per_warp, CTAS_PER_LANE, 4), Uint32)
        nvps = cute.make_rmem_tensor((rows_per_warp,), Int32)
        for rk in cutlass.range_constexpr(rows_per_warp):
            r = row_first + rk * row_step
            head_id = r // MAX_DQL
            q_local = r - head_id * MAX_DQL
            nvps[rk] = Int32(0)
            if r < BLOCK_Q and q_local < decode_query_len:
                nvps[rk] = num_valid_pages[batch_id * decode_query_len + q_local]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for rk in cutlass.range_constexpr(rows_per_warp):
            r = row_first + rk * row_step
            head_id = r // MAX_DQL
            q_local = r - head_id * MAX_DQL
            if r < BLOCK_Q and q_local < decode_query_len:
                rec_row = (Int64(batch_id) * MAX_ROWS + Int64(r)) * SPLIT_K
                self.load_records(
                    rec_base, rec_row, n_active, lane, recs[rk, None, None]
                )
        for rk in cutlass.range_constexpr(rows_per_warp):
            r = row_first + rk * row_step
            head_id = r // MAX_DQL
            q_local = r - head_id * MAX_DQL
            if r < BLOCK_Q and q_local < decode_query_len:
                t = batch_id * decode_query_len + q_local
                nvp = nvps[rk]
                nvp = nvp if nvp < MAX_K_TILES else Int32(MAX_K_TILES)
                nvp = nvp if nvp > 0 else Int32(0)
                if nvp <= TOPK:
                    if lane < TOPK:
                        topk_out[t, head_id, lane] = lane if lane < nvp else Int32(-1)
                else:
                    self.select_row(
                        score,
                        topk_out,
                        recs[rk, None, None],
                        s_cand,
                        widx,
                        t,
                        head_id,
                        nvp,
                        lane,
                    )


class IndexDecodeScoreRecordKernel:
    BLOCK_K = IndexDecodeScoreKernel.BLOCK_K
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int,
        force_begin: int,
        force_end: int,
    ):
        self.dtype = dtype
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim
        self.force_begin = force_begin
        self.force_end = force_end
        assert split_k == SPLIT_K
        assert num_heads * max_decode_query_len <= MAX_ROWS
        assert force_begin + force_end < TOPK

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        score: cute.Tensor,  # [num_heads, bs * runtime_decode_query_len, max_pages]
        seq_lens: cute.Tensor,  # [bs]
        num_valid_pages: cute.Tensor,  # [bs * runtime_decode_query_len]
        records: cute.Tensor,  # [>= bs * MAX_ROWS * SPLIT_K * REC_WORDS]
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        assert BLOCK_Q <= 32

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (batch, self.split_k, 1)
        block = (32 * SCORE_WARPS, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // dtype.width

        sQ_layout = cute.make_layout(
            (MAX_DQL, num_heads, (elems, head_dim // elems)),
            stride=(elems, MAX_DQL * elems, (1, BLOCK_Q * elems)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            sQ_layout,
            cta_tiler=(MAX_DQL, num_heads, head_dim),
        )

        sK_layout = cute.make_layout(
            (1, BLOCK_K, (elems, head_dim // elems), num_stages),
            stride=(0, elems, (1, BLOCK_K * elems), BLOCK_K * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            sK_layout,
            cta_tiler=(1, BLOCK_K, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            score,
            seq_lens,
            num_valid_pages,
            records,
            decode_query_len,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        num_valid_pages: cute.Tensor,
        records: cute.Tensor,
        decode_query_len,
    ):
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        EPI_Q = Q_TILES * MMA_N
        FB = self.force_begin
        FE = self.force_end

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            K_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=K_tma.smem_layout.inner,
        )[0, None, None, None]
        # alias sQ with the 1st stage of sK
        sQ_tma = cute.make_tensor(
            sK[None, None, 0].iterator, layout=Q_tma.smem_layout.outer
        )
        q_tma_elems = 128 * 8 // dtype.width
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
        )
        epi_buffer = smem.allocate_tensor(Float32, cute.make_layout((EPI_Q, 4)))

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, BLOCK_K)
        rec_base = records.iterator.toint()

        if split_id < num_blocks:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                        cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(Q_tma.atom)
                cpasync.prefetch_descriptor(K_tma.atom)
            cute.arch.sync_threads()

            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

            if warp_id == 4:
                # TMA warp
                tma_stage = 0
                tma_parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset(
                        (batch_id * decode_query_len, 0, 0),
                        Q_tma.tma_tensor,
                    ),
                    tiler=(MAX_DQL, NUM_HEADS, head_dim),
                    coord=(0, 0, 0),
                )
                cute.arch.mbarrier_wait(tma_empty_mbar, tma_parity)
                with cute.arch.elect_one():
                    Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_full_mbar, Q_size)
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for block_id in range(split_id, num_blocks, split_k):
                    page_id = block_table[batch_id, block_id]
                    gK_tile = K_tma.tma_tensor[page_id, None, None]
                    k_mbar = tma_full_mbar + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * (dtype.width // 8)
                        cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, K_size)
                    simple_tma_copy(
                        K_tma.atom,
                        gK_tile,
                        sK[None, None, tma_stage],
                        k_mbar,
                        cache_policy=EVICT_FIRST,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            elif warp_id < 4:
                # MMA warps
                # each warp handles K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
                sK_warp = cute.local_tile(
                    sK, (32, head_dim, num_stages), (warp_id, 0, 0)
                )
                q_start = seqlen - decode_query_len

                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)), 1)
                )
                sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))

                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor((elems, 2, head_dim // MMA_K), dtype)
                rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)

                # Warp 0 lane r tracks row r: its causal page count, eligible
                # (non-forced) block range and this CTA's top-3 there.
                head_id = lane_id // MAX_DQL
                q_local_pos = lane_id - head_id * MAX_DQL
                valid_q = head_id < NUM_HEADS and q_local_pos < decode_query_len
                track = warp_id == 0 and lane_id < BLOCK_Q and valid_q
                t_row = batch_id * decode_query_len + q_local_pos
                row_nvp = Int32(0)
                if track:
                    row_nvp = num_valid_pages[t_row]
                    row_nvp = row_nvp if row_nvp < MAX_K_TILES else Int32(MAX_K_TILES)
                elig_hi = row_nvp - FE
                k0 = Uint32(0)
                k1 = Uint32(0)
                k2 = Uint32(0)
                s0 = Uint32(0)
                s1 = Uint32(0)
                s2 = Uint32(0)
                cnt = Uint32(0)
                slot = Uint32(0)

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar, 0)
                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                for q in cutlass.range_constexpr(Q_TILES):
                    cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                cute.arch.mbarrier_arrive(tma_empty_mbar)

                tma_stage = 1 % self.num_stages
                tma_parity = 0
                if tma_stage == 0:
                    tma_parity ^= 1

                if cutlass.const_expr(dtype is Float8E4M3FN):
                    rQ_f16 = cute.make_rmem_tensor(
                        (4, head_dim // MMA_K, Q_TILES, 2), Float16
                    )
                    q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                    rQ_f16[None, None, None, 0].store(q_lower.load())
                    rQ_f16[None, None, None, 1].store(q_upper.load())

                for block_id in range(split_id, num_blocks, split_k):
                    rC.fill(0.0)

                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    for k in cutlass.range_constexpr(head_dim // MMA_K):
                        cute.copy(
                            ldsm_atom,
                            sK_ldsm[None, (None, k, tma_stage)],
                            rK[None, None, k],
                        )
                        for m in cutlass.range_constexpr(2):
                            if cutlass.const_expr(dtype is Float8E4M3FN):
                                rK_lower, rK_upper = _fp8_to_f16_mma_fragments(
                                    rK[None, m, k]
                                )
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK_lower,
                                        rQ_f16[None, k, n, 0],
                                        rC[None, m, n],
                                    )
                                    rC[None, m, n] = mma_sync(
                                        rK_upper,
                                        rQ_f16[None, k, n, 1],
                                        rC[None, m, n],
                                    )
                            else:
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK[None, m, k],
                                        rQ[(None, k % 2), k // 2, n],
                                        rC[None, m, n],
                                    )

                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    k_start = block_id * BLOCK_K + warp_id * 32

                    # causal mask
                    for q in cutlass.range_constexpr(Q_TILES):
                        for i in cutlass.range_constexpr(4):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local = col % MAX_DQL
                                q_pos = q_start + q_local
                                k_pos = k_start + i * 8 + lane_id // 4
                                rC[q * 8 + i * 2 + j] = (
                                    rC[q * 8 + i * 2 + j]
                                    if q_pos >= k_pos
                                    else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        # thread-reduction along BLOCK_K dim
                        rScore = cute.make_rmem_tensor(2, Float32)
                        rScore.fill(float("-inf"))
                        for i in cutlass.range_constexpr(4):
                            rScore[0] = cute.arch.fmax(rScore[0], rC[i * 2 + 0 + q * 8])
                            rScore[1] = cute.arch.fmax(rScore[1], rC[i * 2 + 1 + q * 8])

                        # warp-reduction among lanes 0,4,8,12,...
                        for i in cutlass.range_constexpr(3):
                            offset = 4 << i
                            other0 = cute.arch.shuffle_sync_bfly(
                                rScore[0], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            other1 = cute.arch.shuffle_sync_bfly(
                                rScore[1], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            rScore[0] = cute.arch.fmax(rScore[0], other0)
                            rScore[1] = cute.arch.fmax(rScore[1], other1)

                        # store to smem for 4-warp reduction
                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    if track:
                        final_score = epi_buffer[lane_id, 0]
                        for i in cutlass.range_constexpr(1, 4):
                            final_score = cute.arch.fmax(
                                final_score, epi_buffer[lane_id, i]
                            )
                        score[head_id, t_row, block_id] = final_score

                        # Keep this CTA's top-3 eligible scores of the row; on
                        # a tie the later (higher) block ranks first.
                        if block_id >= FB and block_id < elig_hi:
                            key = _f32_key(final_score)
                            c0 = key >= k0
                            c1 = key >= k1
                            c2 = key >= k2
                            k2 = k1 if c1 else (key if c2 else k2)
                            s2 = s1 if c1 else (slot if c2 else s2)
                            k1 = k0 if c0 else (key if c1 else k1)
                            s1 = s0 if c0 else (slot if c1 else s1)
                            k0 = key if c0 else k0
                            s0 = slot if c0 else s0
                            cnt = cnt + Uint32(1)
                    slot = slot + Uint32(1)

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

                # Publish the row's top-3 record.
                if track:
                    idx = (
                        Int64(batch_id) * MAX_ROWS + Int64(lane_id)
                    ) * SPLIT_K + Int64(split_id)
                    _st_v4_u32(
                        rec_base + idx * (REC_WORDS * 4),
                        k0,
                        k1,
                        k2,
                        s0
                        | (s1 << Uint32(8))
                        | (s2 << Uint32(16))
                        | (cnt << Uint32(24)),
                    )

    @cache
    @staticmethod
    def compile(
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        head_dim: int,
        force_begin: int,
        force_end: int,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        BLOCK_K = IndexDecodeScoreRecordKernel.BLOCK_K

        q = make_fake_tensor(
            dtype, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            dtype, (cute.sym_int(), BLOCK_K, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        score = make_fake_tensor(
            Float32, (num_heads, total_tokens, cute.sym_int()), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        nvp = make_fake_tensor(Int32, (total_tokens,), divisibility=1)
        records = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=4)
        kernel = IndexDecodeScoreRecordKernel(
            dtype,
            num_heads,
            max_decode_query_len,
            SPLIT_K,
            head_dim,
            force_begin,
            force_end,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            score,
            seq_lens,
            nvp,
            records,
            stream,
            options="--enable-tvm-ffi",
        )


class IndexTopKFromRecordsKernel(_RowSelect):
    """One warp per (request, row): PDL-launched after the score kernel."""

    def __init__(
        self,
        num_heads: int,
        max_decode_query_len: int,
        force_begin: int,
        force_end: int,
    ):
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.force_begin = force_begin
        self.force_end = force_end

    @cute.jit
    def __call__(
        self,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        num_valid_pages: cute.Tensor,
        records: cute.Tensor,
        topk_out: cute.Tensor,
        stream: CUstream,
    ):
        batch = seq_lens.shape[0]
        decode_query_len = num_valid_pages.shape[0] // batch
        BLOCK_Q = self.num_heads * self.max_decode_query_len
        self.kernel(
            score, seq_lens, num_valid_pages, records, topk_out, decode_query_len
        ).launch(
            grid=(batch, BLOCK_Q, 1), block=(32, 1, 1), stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        num_valid_pages: cute.Tensor,
        records: cute.Tensor,
        topk_out: cute.Tensor,
        decode_query_len,
    ):
        batch_id, r, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        s_cand = smem.allocate_tensor(
            Uint64, cute.make_layout((1, CAND_CAP)), byte_alignment=8
        )
        rec_base = records.iterator.toint()
        num_blocks = cute.ceil_div(seq_lens[batch_id], 128)
        n_active = num_blocks if num_blocks < SPLIT_K else Int32(SPLIT_K)
        self.select_request_rows(
            score,
            num_valid_pages,
            topk_out,
            rec_base,
            s_cand,
            batch_id,
            decode_query_len,
            n_active,
            Int32(0),
            lane,
            r,
            1,
            1,
        )

    @cache
    @staticmethod
    def compile(
        num_heads: int, max_decode_query_len: int, force_begin: int, force_end: int
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        score = make_fake_tensor(
            Float32, (num_heads, total_tokens, cute.sym_int()), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        nvp = make_fake_tensor(Int32, (total_tokens,), divisibility=1)
        records = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=4)
        topk_out = make_fake_tensor(
            Int32, (total_tokens, num_heads, TOPK), divisibility=1
        )
        kernel = IndexTopKFromRecordsKernel(
            num_heads, max_decode_query_len, force_begin, force_end
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            score,
            seq_lens,
            nvp,
            records,
            topk_out,
            stream,
            options="--enable-tvm-ffi",
        )


class DecodeTopKWorkspace:
    """Persistent per-request CTA records (shared by every indexer layer; the
    layers run one after another on one stream)."""

    def __init__(self, max_batch: int, device: torch.device) -> None:
        self.max_batch = max_batch
        self.records = torch.zeros(
            max_batch * MAX_ROWS * SPLIT_K * REC_WORDS, dtype=torch.int32, device=device
        )


def minimax_m3_index_decode_score_topk_cutedsl(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor,
    num_valid_pages: torch.Tensor,
    topk_out: torch.Tensor,
    workspace: DecodeTopKWorkspace,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Decode index scores into ``score_out`` (``[H, T, MAX_K_TILES]`` view of
    the unified buffer, as the stock kernel) and the exact top-16 blocks of
    every (token, head) row into ``topk_out`` (``[T, H, 16]``), bounded by
    ``num_valid_pages`` with ``init_blocks`` / ``local_blocks`` forced, as
    ``sparse_topk_select`` would select them."""
    if idx_q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("CuteDSL index decode score supports BF16 and FP8 E4M3 only")
    total_tokens, num_heads, head_dim = idx_q.shape
    batch = block_table.shape[0]
    assert index_kv_cache.shape[1] == IndexDecodeScoreRecordKernel.BLOCK_K
    assert total_tokens == batch * decode_query_len
    assert 1 <= decode_query_len <= max_decode_query_len
    assert num_heads * max_decode_query_len <= MAX_ROWS
    assert batch <= workspace.max_batch
    assert score_out.shape[2] == MAX_K_TILES and score_out.stride(2) == 1
    score = IndexDecodeScoreRecordKernel.compile(
        _TORCH_TO_CUTE_DTYPE[idx_q.dtype],
        num_heads,
        max_decode_query_len,
        head_dim,
        init_blocks,
        local_blocks,
    )
    score(
        idx_q,
        index_kv_cache,
        block_table,
        score_out,
        seq_lens,
        num_valid_pages,
        workspace.records,
    )
    select = IndexTopKFromRecordsKernel.compile(
        num_heads, max_decode_query_len, init_blocks, local_blocks
    )
    select(score_out, seq_lens, num_valid_pages, workspace.records, topk_out)
