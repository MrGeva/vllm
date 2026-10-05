# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch k2q CSR + sparse-attention schedule for MiniMax-M3 MSA prefill.

Drop-in for ``fmha_sm100.sparse.build_k2q_csr(..., return_schedule=True)`` with
one KV head per rank, top-16 and 128-token blocks. The stock builder runs four
memsets, a fill and five CUDA kernels (row map, histogram, row prefix, tile
prefix, scatter) per sparse layer; this runs one Triton kernel whose programs
meet at two grid barriers (every program is resident: at most one per SM):

A. Program ``g`` owns queries ``[g * QB, (g + 1) * QB)``. It maps each valid
   ``(q, t)`` edge to its packed KV row (the stock round-robin row map,
   computed in place), counts each query's edges (``split_counts``) and its
   ``split_slot`` (valid edges at lower ``t``), sorts its edges by
   ``(row, q, t)`` and ranks them within each row. It writes its per-row counts
   densely to ``cnt[g, :]`` and adds them to ``tot[row]``; it also pre-fills
   its share of ``q_idx`` with -1 and of the schedule with 0.
B. Program ``g`` owns a slice of rows: from ``tot`` it gets the row offsets of
   earlier rows, then ``row_ptr`` for its rows, each program's base offset in
   each row (``cnt`` is overwritten with it) and the schedule work items of
   its rows (rows ascending, ``target_q_per_cta`` queries per item).
C. Program ``g`` writes its edges at ``base[g, row] + rank``.

``row_ptr``, ``q_idx``, ``qsplit_indices``' valid prefix, ``split_counts`` and
``work_count`` equal the stock builder's. The stock row-prefix kernel claims
work items with an atomic counter, so its item order varies run to run; this
emits the same items in row order. The attention reads its item list
order-independently.

``--additional-config '{"aria_msa_k2q_fused": false}'`` keeps the stock
builder; unsupported shapes always use it.
"""

from __future__ import annotations

import sys

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

CONFIG_KEY = "aria_msa_k2q_fused"
TOPK = 16
BLK_KV = 128
MAX_PROGRAMS = 128
# Packed KV rows the scratch holds (8M tokens of prefill context per step).
ROW_CAP = 1 << 16
# Above ~8k prefill tokens the per-program sort outweighs the saved launches
# (16k tokens: 52 vs 49 us), so those steps keep the stock builder.
MAX_PREFILL_TOKENS = 8192
# Rows per phase-B tile and tile width of the row scans.
SUB_ROWS = 64
ROW_BLOCK = 1024
_KEY_INVALID = tl.constexpr((1 << 63) - 1)

_BUILDER: K2qCsrBuilder | None = None


def enabled(vllm_config) -> bool:
    additional = vllm_config.additional_config
    override = additional.get(CONFIG_KEY) if isinstance(additional, dict) else None
    return not (override is False or override == 0)


def get_builder(device: torch.device) -> K2qCsrBuilder:
    global _BUILDER
    if _BUILDER is None:
        _BUILDER = K2qCsrBuilder(device)
    return _BUILDER


def _query_block(num_q: int, max_programs: int) -> int:
    return max(16, triton.next_power_of_2(triton.cdiv(num_q, max_programs)))


class K2qCsrBuilder:
    """Persistent scratch + launcher (one per device)."""

    def __init__(self, device: torch.device) -> None:
        self.schedule_model = None
        self.schedule_cls = None
        self.device = device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.max_programs = min(MAX_PROGRAMS, sms)
        self.cnt = torch.empty(
            self.max_programs * ROW_CAP, dtype=torch.int32, device=device
        )
        self.tot = torch.zeros((2, ROW_CAP), dtype=torch.int32, device=device)
        self.coords = torch.empty((ROW_CAP, 2), dtype=torch.int32, device=device)
        # Barrier targets start above int32 so their argument type never changes.
        self.bar_base = 1 << 32
        self.bar = torch.full((1,), self.bar_base, dtype=torch.int64, device=device)
        self.parity = 0
        self.dirty = [0, 0]
        self.calls = 0
        self._warmup()
        logger.info_once(
            "MiniMax M3 MSA k2q CSR + schedule built in one Triton launch "
            "(up to %d programs, %d prefill tokens)",
            self.max_programs,
            MAX_PREFILL_TOKENS,
        )

    def _schedule_api(self) -> None:
        """The stock schedule sizing model and schedule class, taken from the
        module fmha_sm100's sparse API loaded (its ``cute/`` modules import as
        top-level ``src.*`` only once that API has put them on sys.path)."""
        if self.schedule_cls is not None:
            return
        from vllm.third_party.fmha_sm100 import sparse as fmha_sparse

        prepare = sys.modules[fmha_sparse.SparseK2qCsrBuilderSm100.__module__]
        self.schedule_model = prepare.SPARSE_SCHEDULE_MODEL
        self.schedule_cls = prepare.SparseAttentionSchedule

    def supports(self, q2k: torch.Tensor, total_rows: int, batch: int) -> bool:
        h, num_q, topk = q2k.shape
        return (
            h == 1
            and topk == TOPK
            and q2k.stride(2) == 1
            and 0 < num_q <= MAX_PREFILL_TOKENS
            and 0 < total_rows <= ROW_CAP
            and batch > 0
        )

    def _warmup(self) -> None:
        """Compile every query-block variant once (at model construction)."""
        for num_q in (1, 1000, 3000, 5000, MAX_PREFILL_TOKENS):
            seq_k = 4096
            q2k = torch.full(
                (1, num_q, TOPK), -1, dtype=torch.int32, device=self.device
            )
            q2k[0, :, 0] = 0
            cu_q = torch.tensor([0, num_q], dtype=torch.int32, device=self.device)
            cu_k = torch.tensor([0, seq_k], dtype=torch.int32, device=self.device)
            rows = seq_k // BLK_KV
            self._launch(q2k, cu_q, cu_k, seq_k, rows, target=8, capacity=rows + num_q)
        torch.cuda.synchronize(self.device)

    def build(
        self,
        q2k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int,
        total_rows: int,
        qhead_per_kv: int,
    ):
        """``(k2q_row_ptr, k2q_q_indices, schedule)`` as the stock builder."""
        self._schedule_api()
        head_kv, num_q, topk = q2k.shape
        model = self.schedule_model
        target = model.balanced_target_q_per_cta(
            total_q=num_q,
            topk=topk,
            blk_kv=BLK_KV,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=q2k.device,
        )
        capacity = model.flat_schedule_capacity(
            total_rows=total_rows,
            total_q=num_q,
            topk=topk,
            head_kv=head_kv,
            target_q_per_cta=target,
        )
        row_ptr, q_idx, qsplit, split_counts, sched, work_count = self._launch(
            q2k, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, total_rows, target, capacity
        )
        if self.calls == 1:
            logger.info(
                "MiniMax M3 first prefill k2q CSR built by the one-launch kernel"
            )
        schedule = self.schedule_cls(
            enabled=True,
            scheduler_metadata=sched,
            work_count=work_count,
            qsplit_indices=qsplit,
            split_counts=split_counts,
            target_q_per_cta=target,
        )
        return row_ptr, q_idx, schedule

    def _launch(
        self,
        q2k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        total_rows: int,
        target: int,
        capacity: int,
    ):
        head_kv, num_q, topk = q2k.shape
        batch = cu_seqlens_q.shape[0] - 1
        device = q2k.device
        max_kv_blocks = triton.cdiv(max(int(max_seqlen_k), BLK_KV), BLK_KV)
        row_ptr = torch.empty(
            (head_kv, total_rows + 1), dtype=torch.int32, device=device
        )
        q_idx = torch.empty((head_kv, num_q * topk), dtype=torch.int32, device=device)
        qsplit = torch.empty_like(q_idx)
        split_counts = torch.empty((num_q, head_kv), dtype=torch.int32, device=device)
        sched = torch.empty((capacity, 6), dtype=torch.int32, device=device)
        work_count = torch.empty((1,), dtype=torch.int32, device=device)

        qb = _query_block(num_q, self.max_programs)
        programs = triton.cdiv(num_q, qb)
        par = self.parity
        clear = self.dirty[1 - par]
        self.dirty[1 - par] = 0
        self.dirty[par] = total_rows
        self.parity ^= 1
        bar1 = self.bar_base + programs
        bar2 = self.bar_base + 2 * programs
        self.bar_base = bar2
        self.calls += 1
        ne = qb * TOPK
        _k2q_csr_kernel[(programs,)](
            q2k,
            q2k.stride(1),
            cu_seqlens_q,
            cu_seqlens_k,
            batch,
            num_q,
            total_rows,
            max_kv_blocks,
            target,
            capacity,
            row_ptr,
            q_idx,
            qsplit,
            split_counts,
            sched,
            work_count,
            self.cnt,
            self.tot[par],
            self.tot[1 - par],
            clear,
            self.coords,
            self.bar,
            bar1,
            bar2,
            QB=qb,
            TOPK=TOPK,
            BLK=BLK_KV,
            SUB=SUB_ROWS,
            GP=triton.next_power_of_2(self.max_programs),
            RBLK=ROW_BLOCK,
            num_warps=4 if ne <= 512 else (8 if ne <= 2048 else 16),
        )
        return row_ptr, q_idx, qsplit, split_counts, sched, work_count


@triton.jit
def _prev_combine(a_cur, a_prev, b_cur, b_prev):
    # Scans (value, previous value): the previous value of a one-element range
    # is unknown (-1); of a longer one, its second-to-last element.
    return b_cur, tl.where(b_prev == -1, a_cur, b_prev)


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _grid_barrier(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    seen = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while seen < target:
        seen = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


# Runtime integers are not value-specialized: a new divisibility class would
# JIT-compile mid-step (seconds) instead of once at warm-up.
@triton.jit(
    do_not_specialize=[
        "q_stride",
        "B",
        "S_Q",
        "R",
        "max_kv_blocks",
        "target_q",
        "work_capacity",
        "clear_rows",
        "bar_target1",
        "bar_target2",
    ]
)
def _k2q_csr_kernel(
    q2k_ptr,
    q_stride,
    cu_q_ptr,
    cu_k_ptr,
    B,
    S_Q,
    R,
    max_kv_blocks,
    target_q,
    work_capacity,
    row_ptr_ptr,
    q_idx_ptr,
    qsplit_ptr,
    split_counts_ptr,
    sched_ptr,
    work_count_ptr,
    cnt_ptr,
    tot_ptr,
    tot_clear_ptr,
    clear_rows,
    coords_ptr,
    bar_ptr,
    bar_target1,
    bar_target2,
    QB: tl.constexpr,
    TOPK: tl.constexpr,
    BLK: tl.constexpr,
    SUB: tl.constexpr,
    GP: tl.constexpr,
    RBLK: tl.constexpr,
):
    g = tl.program_id(0)
    G = tl.num_programs(0)
    NE: tl.constexpr = QB * TOPK
    span = tl.arange(0, RBLK)

    # ---- A: shares of the fills, then this program's edges -------------
    # Totals the previous call left in the other parity's buffer.
    clear_share = tl.cdiv(clear_rows, G)
    for r0 in range(
        g * clear_share, tl.minimum((g + 1) * clear_share, clear_rows), RBLK
    ):
        rr = r0 + span
        tl.store(tot_clear_ptr + rr, tl.zeros([RBLK], tl.int32), mask=rr < clear_rows)
    sched_words = work_capacity * 6
    sched_share = tl.cdiv(sched_words, G)
    sched_end = tl.minimum((g + 1) * sched_share, sched_words)
    for w0 in range(g * sched_share, sched_end, RBLK):
        ww = w0 + span
        tl.store(sched_ptr + ww, tl.zeros([RBLK], tl.int32), mask=ww < sched_end)
    my_cnt = cnt_ptr + g.to(tl.int64) * R
    for r0 in range(0, R, RBLK):
        rr = r0 + span
        tl.store(my_cnt + rr, tl.zeros([RBLK], tl.int32), mask=rr < R)
    fill = g * NE + tl.arange(0, NE)
    tl.store(q_idx_ptr + fill, tl.full([NE], -1, tl.int32), mask=fill < S_Q * TOPK)

    q = g * QB + tl.arange(0, QB)
    qmask = q < S_Q
    bq = tl.zeros([QB], tl.int32)
    for b in range(0, B):
        bq += (tl.load(cu_q_ptr + b + 1) <= q).to(tl.int32)
    bq = tl.minimum(bq, B - 1)
    q_local = q - tl.load(cu_q_ptr + bq)
    k_len = tl.load(cu_k_ptr + bq + 1) - tl.load(cu_k_ptr + bq)
    rows_q = (k_len + BLK - 1) // BLK
    t = tl.arange(0, TOPK)
    kvb = tl.load(
        q2k_ptr + q[:, None].to(tl.int64) * q_stride + t[None, :],
        mask=qmask[:, None],
        other=-1,
    )
    valid = (kvb >= 0) & (kvb < max_kv_blocks) & (kvb < rows_q[:, None])
    level = tl.where(valid, kvb, 0)
    # Stock row map: rows are packed level by level, batches in order.
    row = tl.zeros([QB, TOPK], tl.int32)
    for b in range(0, B):
        kl = tl.load(cu_k_ptr + b + 1) - tl.load(cu_k_ptr + b)
        rb = (kl + BLK - 1) // BLK
        row += tl.minimum(rb, level) + ((b < bq[:, None]) & (rb > level)).to(tl.int32)
    valid = valid & (row < R)
    vi = valid.to(tl.int32)
    split = tl.cumsum(vi, axis=1) - vi
    tl.store(split_counts_ptr + q, tl.sum(vi, axis=1), mask=qmask)
    tl.store(
        coords_ptr + row * 2, bq[:, None] + tl.zeros([QB, TOPK], tl.int32), mask=valid
    )
    tl.store(coords_ptr + row * 2 + 1, level, mask=valid)

    key = (
        (row.to(tl.int64) << 25)
        | (q_local[:, None].to(tl.int64) << 10)
        | (t[None, :].to(tl.int64) << 5)
        | split.to(tl.int64)
    )
    key = tl.where(valid, key, _KEY_INVALID)
    sk = tl.sort(tl.reshape(key, [NE]))
    idx = tl.arange(0, NE)
    vs = sk != _KEY_INVALID
    srow = tl.where(vs, sk >> 25, -2).to(tl.int32)
    unknown = tl.full([NE], -1, tl.int32)
    _, prev = tl.associative_scan((srow, unknown), 0, _prev_combine)
    _, nxt = tl.associative_scan((srow, unknown), 0, _prev_combine, reverse=True)
    first = (idx == 0) | (srow != prev)
    start = tl.associative_scan(tl.where(first, idx, 0), 0, _max_combine)
    rank = idx - start
    last = vs & ((idx == NE - 1) | (srow != nxt))
    tl.debug_barrier()
    tl.store(my_cnt + srow, rank + 1, mask=last)
    tl.atomic_add(tot_ptr + srow, rank + 1, mask=last, sem="relaxed", scope="gpu")

    _grid_barrier(bar_ptr, bar_target1)

    # ---- B: row offsets, per-program bases and schedule of a row slice --
    rs = tl.cdiv(R, G)
    r_lo = tl.minimum(g * rs, R)
    r_hi = tl.minimum(r_lo + rs, R)
    carry = tl.zeros([], tl.int32)
    carry_w = tl.zeros([], tl.int32)
    for r0 in range(0, r_lo, RBLK):
        rr = r0 + span
        tv = tl.load(tot_ptr + rr, mask=rr < r_lo, other=0, cache_modifier=".cg")
        carry += tl.sum(tv)
        carry_w += tl.sum((tv + target_q - 1) // target_q)
    gg = tl.arange(0, GP)
    gm = gg < G
    sub = tl.arange(0, SUB)
    for r0 in range(r_lo, r_hi, SUB):
        rr = r0 + sub
        rm = rr < r_hi
        cptr = cnt_ptr + gg[:, None].to(tl.int64) * R + rr[None, :]
        cm = gm[:, None] & rm[None, :]
        c = tl.load(cptr, mask=cm, other=0, cache_modifier=".cg")
        tot = tl.sum(c, axis=0)
        incl = tl.cumsum(tot, axis=0)
        excl = carry + incl - tot
        tl.store(row_ptr_ptr + rr + 1, carry + incl, mask=rm)
        tl.store(cptr, excl[None, :] + tl.cumsum(c, axis=0) - c, mask=cm)
        carry += tl.sum(tot)
        chunks = (tot + target_q - 1) // target_q
        wbase = carry_w + tl.cumsum(chunks, axis=0) - chunks
        has = rm & (tot > 0)
        bidx = tl.load(coords_ptr + rr * 2, mask=has, other=0, cache_modifier=".cg")
        lvl = tl.load(coords_ptr + rr * 2 + 1, mask=has, other=0, cache_modifier=".cg")
        for ci in range(0, tl.max(chunks)):
            w = wbase + ci
            mm = has & (ci < chunks) & (w < work_capacity)
            item = sched_ptr + w.to(tl.int64) * 6
            tl.store(item + 1, rr, mask=mm)
            tl.store(item + 2, ci * target_q + tl.zeros([SUB], tl.int32), mask=mm)
            tl.store(item + 3, tl.minimum(target_q, tot - ci * target_q), mask=mm)
            tl.store(item + 4, bidx, mask=mm)
            tl.store(item + 5, lvl, mask=mm)
        carry_w += tl.sum(chunks)
    if g == 0:
        tl.store(row_ptr_ptr, 0)
    if (r_lo < R) & (r_hi == R):
        tl.store(work_count_ptr, carry_w)

    _grid_barrier(bar_ptr, bar_target2)

    # ---- C: scatter -------------------------------------------------------
    base = tl.load(my_cnt + srow, mask=vs, other=0, cache_modifier=".cg")
    pos = base + rank
    ql = ((sk >> 10) & 0x7FFF).to(tl.int32)
    sp = (sk & 0x1F).to(tl.int32)
    tl.store(q_idx_ptr + pos, ql, mask=vs)
    tl.store(qsplit_ptr + pos, ql | ((sp & 0xFF) << 24), mask=vs)
