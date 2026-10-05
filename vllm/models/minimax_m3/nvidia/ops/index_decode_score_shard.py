# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL MiniMax M3 index decode score over a TP-sharded index-K cache.

With the index-K cache sharded across TP ranks (rank ``r`` keeps tokens
``r, r + tp, ...`` of every 128-token block, as a ``[128 // tp, D]`` page), each
rank scores its token slice for *all* index heads and pushes the per-block
partial maxima of head ``h`` straight into the receive slot of the rank owning
``h`` (a relaxed system-scope store over NVLink); the owner's cross-rank max
(``idxk_shard``) polls those slots, so no barrier is needed.

The kernel is ``IndexDecodeScoreKernel`` with one shard page per MMA warp:
every (token, query) dot product uses the same fp8->fp16 conversion and
``mma.sync`` k-order, so the partial maxima (and their max) are bit-identical
to the replicated scores. fp8 only, ``tp == 4`` (32-row pages).
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import EVICT_FIRST, mma_sync, simple_tma_copy
from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
    _fp8_to_f16_mma_fragments,
)

# Tokens per KV block (== M3 sparse block == score tile).
BLOCK = 128
# Shard page rows: one MMA warp (32 rows) per page.
ROWS = 32
SUPPORTED_TP = BLOCK // ROWS
# Requests' super-tiles of 4 pages are split over this many CTAs per request.
DEFAULT_SPLIT_K = 128
# Tile dim of a receive slot (== the unified score buffer's, 1M tokens).
MAX_K_TILES = 8192


@dsl_user_op
def _store_relaxed_sys_v4_f32(
    addr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        T.i32(),
        [addr.ir_value(loc=loc, ip=ip)]
        + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "mov.u32 $0, 0;\n\tst.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};",
        "=r,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


class IndexDecodeScoreShardKernel:
    PAGES = 4  # shard pages (score tiles) per pipeline stage, one per MMA warp
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
    ):
        self.dtype = Float8E4M3FN
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, ROWS, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        recv_bases: cute.Tensor,  # [tp] int64: every rank's receive buffer
        seq_lens: cute.Tensor,  # [bs]
        rank: Int32,  # this rank's token offset inside every block
        buf: Int32,  # receive slot set of this layer
        t_max: Int32,  # token capacity of a receive slot
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        assert BLOCK_Q <= 32
        assert head_dim * dtype.width == 128 * 8  # one 128 B row per token

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (batch, self.split_k, 1)
        block = (32 * 5, 1, 1)

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

        # One TMA box per shard page; a stage holds PAGES pages back to back,
        # i.e. the same [128, 128 B] swizzled tile as the replicated kernel.
        # Page p of stage s is buffer s * PAGES + p.
        sK_layout = cute.make_layout(
            (1, ROWS, (elems, head_dim // elems), self.PAGES * num_stages),
            stride=(0, elems, (1, ROWS * elems), ROWS * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            sK_layout,
            cta_tiler=(1, ROWS, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            recv_bases,
            seq_lens,
            decode_query_len,
            rank,
            buf,
            t_max,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        recv_bases: cute.Tensor,
        seq_lens: cute.Tensor,
        decode_query_len,
        rank,
        buf,
        t_max,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        PAGES = self.PAGES
        TP = SUPPORTED_TP
        HPR = NUM_HEADS // TP
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        row_elems = 128 * 8 // dtype.width
        page_bytes = ROWS * head_dim * (dtype.width // 8)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            K_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=K_tma.smem_layout.inner,
        )
        # alias sQ with the 1st page of the 1st stage of sK
        sQ_tma = cute.make_tensor(
            sK[0, None, None, 0].iterator, layout=Q_tma.smem_layout.outer
        )
        q_tma_elems = row_elems
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
        )

        # Per-super-tile maxima [column, page]: a column's 4 tiles are adjacent
        # in the receive slot, so one 16 B store pushes them.
        EPI_Q = Q_TILES * MMA_N
        epi_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((EPI_Q, PAGES), stride=(PAGES, 1)), 16
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, BLOCK)
        num_super = cute.ceil_div(num_blocks, PAGES)

        if split_id < num_super:
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
                # TMA bounds-checks rows when runtime decode_query_len is smaller
                # than MAX_DQL; padded Q columns are masked before global stores.
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for super_id in range(split_id, num_super, split_k):
                    k_mbar = tma_full_mbar + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            k_mbar, PAGES * page_bytes
                        )
                    for p in cutlass.range_constexpr(PAGES):
                        # Past the request's last block, reload its last page;
                        # the causal mask turns such tiles into -inf.
                        tile = super_id * PAGES + p
                        if tile >= num_blocks:
                            tile = num_blocks - 1
                        page_id = block_table[batch_id, tile]
                        simple_tma_copy(
                            K_tma.atom,
                            K_tma.tma_tensor[page_id, None, None],
                            sK[None, None, None, tma_stage * PAGES + p],
                            k_mbar,
                            cache_policy=EVICT_FIRST,
                        )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warps: warp w handles shard page w of every stage,
                # K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
                sK_pages = cute.make_tensor(
                    sK.iterator,
                    cute.make_layout(
                        (ROWS, (row_elems, head_dim // row_elems), PAGES, num_stages),
                        stride=(
                            row_elems,
                            (1, ROWS * row_elems),
                            ROWS * head_dim,
                            PAGES * ROWS * head_dim,
                        ),
                    ),
                )
                sK_warp = sK_pages[None, None, warp_id, None]
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

                rQ_f16 = cute.make_rmem_tensor(
                    (4, head_dim // MMA_K, Q_TILES, 2), Float16
                )
                q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                rQ_f16[None, None, None, 0].store(q_lower.load())
                rQ_f16[None, None, None, 1].store(q_upper.load())

                for super_id in range(split_id, num_super, split_k):
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

                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    tile = super_id * PAGES + warp_id
                    # Page row i holds the block's token i * TP + rank.
                    k_base = tile * BLOCK + rank

                    # causal mask
                    for q in cutlass.range_constexpr(Q_TILES):
                        for i in cutlass.range_constexpr(4):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local_pos = col % MAX_DQL
                                q_pos = q_start + q_local_pos
                                k_pos = k_base + (i * 8 + lane_id // 4) * TP
                                rC[q * 8 + i * 2 + j] = (
                                    rC[q * 8 + i * 2 + j]
                                    if q_pos >= k_pos
                                    else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        # thread-reduction along the page's 32 rows
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

                        # lanes 0..3 hold columns q * 8 + 2 * lane + {0, 1}
                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    # Push column c's 4 tile maxima (head c // MAX_DQL) as one
                    # 16 B store into slot [buf, rank, h % HPR, t, 4 * super_id]
                    # of rank h // HPR. Tiles past the last block of the final
                    # super-tile are fully masked, so they push -inf.
                    if warp_id == 0 and lane_id < BLOCK_Q:
                        head_id = lane_id // MAX_DQL
                        q_local_pos = lane_id - head_id * MAX_DQL
                        if q_local_pos < decode_query_len:
                            t = batch_id * decode_query_len + q_local_pos
                            dst = head_id // HPR
                            slot = (buf * TP + rank) * HPR + head_id % HPR
                            elem = (slot * t_max + t) * MAX_K_TILES + super_id * PAGES
                            _store_relaxed_sys_v4_f32(
                                recv_bases[dst] + Int64(elem) * 4,
                                epi_buffer[lane_id, 0],
                                epi_buffer[lane_id, 1],
                                epi_buffer[lane_id, 2],
                                epi_buffer[lane_id, 3],
                            )

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cache
    @staticmethod
    def compile(
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        dtype = Float8E4M3FN

        q = make_fake_tensor(
            dtype, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            dtype, (cute.sym_int(), ROWS, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        recv_bases = make_fake_tensor(Int64, (SUPPORTED_TP,), divisibility=1)
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        kernel = IndexDecodeScoreShardKernel(
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            recv_bases,
            seq_lens,
            Int32(0),
            Int32(0),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )


def minimax_m3_index_decode_score_shard_cutedsl(
    idx_q: torch.Tensor,
    index_kv_shard: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rank: int,
    decode_query_len: int,
    max_decode_query_len: int,
    recv_bases: torch.Tensor,
    buf: int,
    t_max: int,
    split_k: int = DEFAULT_SPLIT_K,
) -> None:
    """Push the partial block maxima of this rank's index-K shard, for every
    index head, into the owning ranks' receive slots.

    ``idx_q`` is ``[bs * dql, num_heads, D]`` e4m3 holding every index head;
    ``index_kv_shard`` is ``[num_pages, 32, D]`` e4m3, page row ``i`` holding
    token ``i * 4 + rank`` of the block. ``recv_bases`` holds each rank's
    receive buffer address, laid out ``[bufs, tp, heads_per_rank, t_max,
    MAX_K_TILES]`` fp32. Tiles below ``cdiv(seq_len, 128)`` rounded up to a
    multiple of 4 are written (those past the last block are -inf).
    """
    if idx_q.dtype != torch.float8_e4m3fn or index_kv_shard.dtype != idx_q.dtype:
        raise TypeError("sharded index decode score supports FP8 E4M3 only")
    total_tokens, num_heads, head_dim = idx_q.shape
    batch = block_table.shape[0]
    assert index_kv_shard.shape[1] == ROWS
    assert total_tokens == batch * decode_query_len
    assert 1 <= decode_query_len <= max_decode_query_len
    assert num_heads * max_decode_query_len <= 32
    assert 0 <= rank < SUPPORTED_TP and num_heads % SUPPORTED_TP == 0
    assert total_tokens <= t_max
    kernel = IndexDecodeScoreShardKernel.compile(
        num_heads,
        max_decode_query_len,
        split_k,
        head_dim,
    )
    kernel(idx_q, index_kv_shard, block_table, recv_bases, seq_lens, rank, buf, t_max)
