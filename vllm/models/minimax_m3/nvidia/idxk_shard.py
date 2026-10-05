# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP-sharded index-K side cache for the MiniMax-M3 MSA indexer (fp8, TP4).

Every TP rank otherwise stores and streams the whole replicated index-K cache
(one 128-dim key per token per sparse layer) while the index heads are already
TP-split. Sharded, rank ``r`` keeps tokens ``r, r + tp, ...`` of every 128-token
block as a ``[128 // tp, D]`` page, so the decode index score reads ``1 / tp`` of
the bytes and the KV pool budgets ``1 / tp`` of the index-K per token.

- Write: the fused qknorm/rope/insert kernel addresses the index cache by flat
  row (``slot * D``) and skips negative slots, so a remapped slot mapping makes
  it write only this rank's rows (``shard_slot_mapping``).
- Decode: every rank computes all index-query heads (the fused qkv GEMM carries
  the whole index_q projection; its columns are bit-identical to the per-head
  GEMM), scores its token slice for every head and pushes each head's per-block
  partial maxima into the owning rank's receive slot
  (``index_decode_score_shard``). The owner's ``reduce_max`` polls its slots
  until no sentinel is left (no barrier), takes the max -- bit-identical to the
  replicated scores, so the top-k is unchanged -- and resets the slots.
- Prefill: the prefill requests' pages are gathered and re-interleaved into
  full pages for the stock ``fmha_sm100`` OnlyScore pass. Each rank copies
  its rows of those pages into a symmetric staging buffer and signals every
  peer; each rank then reads all ranks' staged rows over NVLink straight into
  the interleaved order (``PageGather``, two launches per layer). Steps with
  more pages than the staging buffer holds, or
  ``--additional-config '{"aria_idxk_symm_gather": false}'``, use the NCCL
  all_gather + re-interleave.

Slot reuse needs no barrier: the layers rotate over receive slot sets so that a
set is reused two invocations later at the earliest, and a rank writes
invocation ``k + 2`` only after it has read every rank's invocation ``k + 1``
pushes, which those ranks issued after finishing (and resetting) ``k``.

On by default on a TP4 fp8 MSA indexer when ``max_num_active_seqs`` (else
``max_num_seqs``) >= 6, or when ``max_model_len`` >= 256k tokens: at decode
batch 1-5 the exchange's fixed cost is only repaid by long contexts, where the
replicated score streams far more bytes. ``--additional-config
'{"aria_msa_index_k_shard": false}'`` restores the replicated cache, ``true``
forces it on below both thresholds.
"""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

CONFIG_KEY = "aria_msa_index_k_shard"
# Smallest decode batch cap where the cut idec bytes beat the exchange cost.
MIN_ACTIVE_SEQS = 6
# Below that cap, the smallest max_model_len that turns the shard on: at decode
# batch 1 the sharded score + exchange beats the replicated score from ~200k
# tokens of context on, and the gap grows with context.
LONG_CONTEXT_MIN_LEN = 256 * 1024
# Score-kernel split (CTAs per request) of a small-cap deployment's decode
# batches of 1 and 2 requests (B300 sweep over 64k-560k contexts); larger
# batches keep the default.
SMALL_BATCH_SPLITS = {1: 512, 2: 256}
BLOCK = 128
SUPPORTED_TP = 4
ROWS = BLOCK // SUPPORTED_TP
# Score tile dim of the unified buffer (indexer_msa.MAX_K_TILES).
MAX_K_TILES = 8192
# Receive slot sets the layers rotate over.
NUM_SLOT_SETS = 3
SENTINEL = -1  # 0xFFFFFFFF: a NaN, which no score is
REDUCE_CHUNK = 512
SPIN_LIMIT = 1 << 26
# Speculative methods whose drafter does not reuse the target's sparse layers.
_SPEC_METHODS = (None, "eagle", "eagle3", "ngram", "suffix", "draft_model")
GATHER_CONFIG_KEY = "aria_idxk_symm_gather"
# Prefill pages one staging buffer holds (2M tokens of prefill context).
GATHER_MAX_PAGES = 16384
# Staging buffers the layers alternate over (see PageGather).
GATHER_STAGES = 2
# Epochs start above int32 so the kernels' epoch argument never changes type.
GATHER_EPOCH0 = 1 << 32

_COMM: IdxKShardComm | None = None
_COMM_FAILED = False
_GATHER: PageGather | None = None


def _max_decode_query_len(vllm_config: VllmConfig) -> int:
    spec = vllm_config.speculative_config
    n = getattr(spec, "num_speculative_tokens", None) if spec is not None else None
    return 1 + (n or 0)


def resolve_shard(
    vllm_config: VllmConfig,
    total_index_heads: int,
    indexer_kv_dtype: str,
    topk_blocks: int,
) -> int:
    """TP shard factor of the index-K cache (1 keeps it replicated).

    Depends only on the config (identical on every rank) plus a symmetric
    memory set-up that all ranks agree on.
    """
    additional = vllm_config.additional_config
    override = additional.get(CONFIG_KEY) if isinstance(additional, dict) else None
    if override is False or override == 0:
        return 1
    tp = get_tensor_model_parallel_world_size()
    if (
        tp != SUPPORTED_TP
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        return 1
    # The MSA indexer (fmha_sm100 top-16) with an fp8 index cache only.
    if topk_blocks != 16 or indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
        return 1
    if total_index_heads % tp:
        return 1
    parallel = vllm_config.parallel_config
    if (
        getattr(parallel, "decode_context_parallel_size", 1) > 1
        or getattr(parallel, "prefill_context_parallel_size", 1) > 1
    ):
        return 1
    spec = vllm_config.speculative_config
    if getattr(spec, "method", None) not in _SPEC_METHODS:
        return 1
    if total_index_heads * _max_decode_query_len(vllm_config) > 32:
        return 1
    cap = _decode_batch_cap(vllm_config)
    max_model_len = vllm_config.model_config.max_model_len
    if (
        override is not True
        and cap < MIN_ACTIVE_SEQS
        and max_model_len < LONG_CONTEXT_MIN_LEN
    ):
        return 1
    if get_comm(vllm_config, total_index_heads) is None:
        return 1
    logger.info_once(
        "MiniMax M3 index-K cache sharded over TP%d (%d of %d tokens per block "
        "per rank; decode-batch cap %d, max_model_len %d)",
        tp,
        ROWS,
        BLOCK,
        cap,
        max_model_len,
    )
    return tp


def _decode_batch_cap(vllm_config: VllmConfig) -> int:
    sched = vllm_config.scheduler_config
    return getattr(sched, "max_num_active_seqs", None) or sched.max_num_seqs


def get_comm(
    vllm_config: VllmConfig | None = None, total_index_heads: int = 0
) -> IdxKShardComm | None:
    """Process-wide receive buffers; created (collectively) on first use."""
    global _COMM, _COMM_FAILED
    if _COMM is not None or _COMM_FAILED or vllm_config is None:
        return _COMM
    from vllm.distributed.parallel_state import get_tp_group

    comm, error = None, None
    try:
        comm = IdxKShardComm(vllm_config, total_index_heads)
    except Exception as e:  # noqa: BLE001 - every rank falls back together
        error = e
    ok = torch.tensor([int(comm is not None)], dtype=torch.int32)
    torch.distributed.all_reduce(
        ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
    )
    if not int(ok.item()):
        logger.warning(
            "MiniMax M3 index-K sharding disabled: symmetric memory set-up failed (%r)",
            error,
        )
        _COMM_FAILED = True
        return None
    _COMM = comm
    _init_page_gather(vllm_config)
    return _COMM


def _init_page_gather(vllm_config: VllmConfig) -> None:
    """Collectively set up the one-sided prefill page gather (``PageGather``);
    every rank keeps the NCCL gather if any rank fails."""
    global _GATHER
    from vllm.distributed.parallel_state import get_tp_group

    additional = vllm_config.additional_config
    override = (
        additional.get(GATHER_CONFIG_KEY) if isinstance(additional, dict) else None
    )
    if override is False or override == 0:
        return
    sparse_cfg = vllm_config.model_config.hf_text_config.sparse_attention_config
    gather, error = None, None
    try:
        gather = PageGather(int(sparse_cfg["sparse_index_dim"]))
    except Exception as e:  # noqa: BLE001 - every rank falls back together
        error = e
    ok = torch.tensor([int(gather is not None)], dtype=torch.int32)
    torch.distributed.all_reduce(
        ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
    )
    if not int(ok.item()):
        logger.warning(
            "MiniMax M3 index-K symmetric page gather disabled (%r); "
            "prefill pages use the NCCL all_gather",
            error,
        )
        return
    _GATHER = gather


class IdxKShardComm:
    """Symmetric receive slots shared by every sharded indexer layer.

    ``recv[set, src, h, t, tile]`` (fp32 bits, sentinel -1) holds rank ``src``'s
    partial maximum of this rank's head ``h`` for decode token ``t``.
    """

    def __init__(self, vllm_config: VllmConfig, total_index_heads: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.num_heads = total_index_heads
        self.hpr = total_index_heads // self.world
        self.max_decode_tokens = (
            vllm_config.scheduler_config.max_num_seqs
            * _max_decode_query_len(vllm_config)
        )
        # Small-cap deployments (sharded for long contexts) spread a 1-2
        # request decode batch over more CTAs per request.
        self.small_batch_splits = (
            SMALL_BATCH_SPLITS
            if _decode_batch_cap(vllm_config) < MIN_ACTIVE_SEQS
            else {}
        )
        self.device = torch.device("cuda", torch.cuda.current_device())
        shape = (
            NUM_SLOT_SETS,
            self.world,
            self.hpr,
            self.max_decode_tokens,
            MAX_K_TILES,
        )
        recv = symm_mem.empty(shape, dtype=torch.int32, device=self.device)
        recv.fill_(SENTINEL)
        handle = symm_mem.rendezvous(recv, group.device_group.group_name)
        assert handle is not None
        self.recv, self.handle = recv, handle
        bases = [int(p) for p in handle.buffer_ptrs]
        assert bases[self.rank] == recv.data_ptr()
        self.recv_bases = torch.tensor(bases, dtype=torch.int64, device=self.device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.num_layers = 0
        torch.cuda.synchronize(self.device)
        handle.barrier()
        logger.info_once(
            "MiniMax M3 index-K shard exchange: %.1f MiB symmetric receive slots "
            "(max %d decode tokens)",
            recv.numel() * 4 / (1 << 20),
            self.max_decode_tokens,
        )

    def decode_split(self, batch: int) -> int | None:
        """Score-kernel split for a decode batch of ``batch`` requests (None
        keeps the kernel's default)."""
        return self.small_batch_splits.get(batch)

    def register_layer(self) -> int:
        """Ordinal of a sharded indexer layer (layers run in ordinal order)."""
        self.num_layers += 1
        return self.num_layers - 1

    def slot_set(self, ordinal: int) -> int:
        """Receive slot set of a layer: consecutive invocations, including the
        last layer followed by the first, never share a set."""
        n = self.num_layers
        if n % 2 == 0:
            return ordinal % 2
        if n % 3 == 0:
            return ordinal % 3
        return 2 if ordinal == n - 1 else ordinal % 2

    def reduce_max(
        self,
        nd: int,
        seq_lens: torch.Tensor,
        decode_query_len: int,
        slot_set: int,
        out: torch.Tensor,
    ) -> None:
        """``out[t, h, :n(t)]`` = max over ranks of the pushed partial maxima of
        this rank's heads, ``n(t)`` = the request's block count rounded up to a
        multiple of 4; resets the consumed slots. ``out`` is ``[nd, hpr, tiles]``.
        """
        assert nd <= self.max_decode_tokens
        nchunk = triton.cdiv(MAX_K_TILES, REDUCE_CHUNK)
        _idxk_max_reduce_kernel[(nd * self.hpr, nchunk)](
            self.recv[slot_set],
            out,
            out.stride(0),
            out.stride(1),
            seq_lens,
            decode_query_len,
            self.errors,
            self.recv.stride(1),
            self.recv.stride(2),
            self.recv.stride(3),
            W=self.world,
            HPR=self.hpr,
            CHUNK=REDUCE_CHUNK,
            SPIN=SPIN_LIMIT,
            num_warps=4,
            # Polls the pushed slots, so it may start under the score kernel's
            # tail (programmatic dependent launch hides the launch latency).
            launch_pdl=True,
        )


@triton.jit
def _idxk_max_reduce_kernel(
    recv_ptr,
    out_ptr,
    out_stride_t,
    out_stride_h,
    seq_lens_ptr,
    decode_query_len,
    err_ptr,
    recv_stride_src,
    recv_stride_h,
    recv_stride_t,
    W: tl.constexpr,
    HPR: tl.constexpr,
    CHUNK: tl.constexpr,
    SPIN: tl.constexpr,
):
    th = tl.program_id(0)
    t = th // HPR
    hh = th % HPR
    # The score kernel pushes whole 4-tile super-tiles (-inf past the last
    # block, which the unified buffer already holds there).
    seq_len = tl.load(seq_lens_ptr + t // decode_query_len)
    n_tiles = (seq_len + 511) // 512 * 4
    start = tl.program_id(1) * CHUNK
    if start < n_tiles:
        offs = start + tl.arange(0, CHUNK)
        mask = offs < n_tiles
        row = recv_ptr + hh * recv_stride_h + t * recv_stride_t + offs
        # Poll every source's slot until none holds the sentinel.
        acc = tl.full([CHUNK], float("-inf"), tl.float32)
        pending = tl.full([], 1, tl.int32)
        spins = tl.full([], 0, tl.int32)
        while (pending > 0) & (spins < SPIN):
            acc = tl.full([CHUNK], float("-inf"), tl.float32)
            missing = tl.zeros([CHUNK], dtype=tl.int32)
            for p in tl.static_range(W):
                bits = tl.load(
                    row + p * recv_stride_src, mask=mask, other=0, volatile=True
                )
                missing += (bits == -1).to(tl.int32)
                acc = tl.maximum(acc, bits.to(tl.float32, bitcast=True))
            pending = tl.sum(missing, axis=0)
            spins += 1
        if pending > 0:
            tl.atomic_add(err_ptr, 1)
        tl.store(out_ptr + t * out_stride_t + hh * out_stride_h + offs, acc, mask=mask)
        for p in tl.static_range(W):
            tl.store(
                row + p * recv_stride_src,
                tl.full([CHUNK], -1, tl.int32),
                mask=mask,
            )


class PageGather:
    """One-sided prefill page gather over symmetric staging buffers.

    ``stage[s, p]`` holds this rank's ``[ROWS, D]`` rows of the ``p``-th
    requested page; ``flags[src]`` holds the last gather epoch whose rows rank
    ``src`` has staged. Gather epoch ``e`` uses stage ``s = e % GATHER_STAGES``:

    1. ``_idxk_stage_pages_kernel`` copies this rank's rows of the pages into
       ``stage[s]``; its last program stores ``e`` into every rank's
       ``flags[rank]`` (release, system scope).
    2. ``_idxk_pull_pages_kernel`` waits until every ``flags[src] >= e``
       (acquire) and copies row ``i`` of every rank's ``stage[s, p]`` to row
       ``i * tp + src`` of full page ``p``: the bytes the NCCL all_gather +
       re-interleave produce.

    A rank restages ``stage[s]`` at epoch ``e + 2`` only after its pull of
    ``e + 1``, which waited for every peer's staging of ``e + 1``; a peer stages
    ``e + 1`` only after its pull of ``e`` has read ``stage[s]``. Every rank
    issues the same gathers (same steps and layers) in the same order.
    """

    def __init__(self, index_dim: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.row_words = index_dim // 4
        self.page_words = ROWS * self.row_words
        assert index_dim % 16 == 0
        self.device = torch.device("cuda", torch.cuda.current_device())
        name = group.device_group.group_name
        stage = symm_mem.empty(
            (GATHER_STAGES, GATHER_MAX_PAGES, self.page_words),
            dtype=torch.int32,
            device=self.device,
        )
        stage_handle = symm_mem.rendezvous(stage, name)
        flags = symm_mem.empty((self.world,), dtype=torch.int64, device=self.device)
        flags.zero_()
        flags_handle = symm_mem.rendezvous(flags, name)
        assert stage_handle is not None and flags_handle is not None
        assert int(stage_handle.buffer_ptrs[self.rank]) == stage.data_ptr()
        assert int(flags_handle.buffer_ptrs[self.rank]) == flags.data_ptr()
        self.stage, self.flags = stage, flags
        self.handles = (stage_handle, flags_handle)
        self.stage_bases = torch.tensor(
            [int(p) for p in stage_handle.buffer_ptrs],
            dtype=torch.int64,
            device=self.device,
        )
        self.flag_bases = torch.tensor(
            [int(p) for p in flags_handle.buffer_ptrs],
            dtype=torch.int64,
            device=self.device,
        )
        self.counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.epoch = GATHER_EPOCH0
        torch.cuda.synchronize(self.device)
        stage_handle.barrier()
        # Compile both kernels now (every rank, collectively), not mid-step.
        dummy = torch.zeros((2, ROWS, index_dim), dtype=torch.uint8, device=self.device)
        pages = torch.arange(2, dtype=torch.int64, device=self.device)
        for _ in range(GATHER_STAGES):
            self.gather(dummy, pages)
        torch.cuda.synchronize(self.device)
        stage_handle.barrier()
        logger.info_once(
            "MiniMax M3 index-K prefill pages gathered one-sided over symmetric "
            "memory: %d x %.1f MiB staging (up to %d pages per step)",
            GATHER_STAGES,
            stage[0].numel() * 4 / (1 << 20),
            GATHER_MAX_PAGES,
        )

    def gather(
        self, kv_shard: torch.Tensor, pages: torch.Tensor
    ) -> torch.Tensor | None:
        """Full ``[P, 128, D]`` pages, or None when the step does not fit."""
        num_pages = pages.numel()
        num_blocks, rows, dim = kv_shard.shape
        if (
            num_pages == 0
            or num_pages > GATHER_MAX_PAGES
            or rows != ROWS
            or dim != self.row_words * 4
            or kv_shard.stride(2) != 1
            or kv_shard.stride(1) != dim
            or kv_shard.stride(0) % 4
        ):
            return None
        kv_words = kv_shard.view(torch.uint8).view(torch.int32)
        self.epoch += 1
        stage = (self.epoch % GATHER_STAGES) * GATHER_MAX_PAGES * self.page_words
        _idxk_stage_pages_kernel[(num_pages,)](
            kv_words,
            kv_words.stride(0),
            pages,
            self.stage.view(-1)[stage:],
            self.counter,
            num_pages,
            self.flag_bases,
            self.rank,
            self.epoch,
            W=self.world,
            PAGE_WORDS=self.page_words,
            num_warps=4,
        )
        full = torch.empty(
            (num_pages, ROWS * self.world, self.row_words * 4),
            dtype=torch.uint8,
            device=kv_shard.device,
        )
        _idxk_pull_pages_kernel[(num_pages,)](
            self.flags,
            self.epoch,
            self.stage_bases,
            stage,
            full.view(torch.int32),
            W=self.world,
            ROWS=ROWS,
            ROW_WORDS=self.row_words,
            num_warps=8,
        )
        return full.view(kv_shard.dtype)


# Runtime integers are not value-specialized: a new divisibility class would
# otherwise JIT-compile mid-step on one rank while its peers wait in the pull.
@triton.jit(do_not_specialize=["kv_page_stride", "num_programs", "rank", "epoch"])
def _idxk_stage_pages_kernel(
    kv_ptr,
    kv_page_stride,
    pages_ptr,
    stage_ptr,
    counter_ptr,
    num_programs,
    flag_bases_ptr,
    rank,
    epoch,
    W: tl.constexpr,
    PAGE_WORDS: tl.constexpr,
):
    p = tl.program_id(0)
    page = tl.load(pages_ptr + p).to(tl.int64)
    offs = tl.arange(0, PAGE_WORDS)
    vals = tl.load(kv_ptr + page * kv_page_stride + offs)
    tl.store(stage_ptr + p.to(tl.int64) * PAGE_WORDS + offs, vals)
    # Every thread's rows are stored before this program counts itself done;
    # the last one resets the counter and publishes the epoch to every rank.
    tl.debug_barrier()
    done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
    if done == num_programs - 1:
        tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
        for r in tl.static_range(W):
            peer = tl.load(flag_bases_ptr + r).to(tl.pointer_type(tl.int64))
            tl.atomic_xchg(peer + rank, epoch, sem="release", scope="sys")


@triton.jit(do_not_specialize=["epoch", "stage_off"])
def _idxk_pull_pages_kernel(
    flags_ptr,
    epoch,
    stage_bases_ptr,
    stage_off,
    out_ptr,
    W: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    # No timeout: reading a peer's staging buffer before it publishes would
    # return stale rows, so a stalled peer stalls this rank instead.
    for r in tl.static_range(W):
        seen = tl.atomic_add(flags_ptr + r, 0, sem="acquire", scope="sys")
        while seen < epoch:
            seen = tl.atomic_add(flags_ptr + r, 0, sem="acquire", scope="sys")
    tl.debug_barrier()
    p = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, ROWS * W)
    i = j // W
    src_rank = j % W
    c = tl.arange(0, ROW_WORDS)
    base = tl.load(stage_bases_ptr + src_rank).to(tl.pointer_type(tl.int32))
    row = stage_off + p * (ROWS * ROW_WORDS) + i * ROW_WORDS
    vals = tl.load(base[:, None] + row[:, None] + c[None, :])
    dst = out_ptr + p * (ROWS * W * ROW_WORDS) + j[:, None] * ROW_WORDS + c[None, :]
    tl.store(dst, vals)


def shard_slot_mapping(
    slot_mapping: torch.Tensor, rank: int, tp: int, out: torch.Tensor
) -> torch.Tensor:
    """Remap block-size-128 slots to rows of the ``[num_blocks, 128 // tp, D]``
    shard; slots of other ranks' tokens (and padding) become -1."""
    n = slot_mapping.numel()
    s = slot_mapping
    off = torch.remainder(s, BLOCK)
    own = (s >= 0) & (torch.remainder(off, tp) == rank)
    row = torch.div(s, BLOCK, rounding_mode="floor") * (BLOCK // tp) + off // tp
    out = out[:n]
    out.copy_(row.masked_fill_(~own, -1))
    return out


def gather_prefill_pages(kv_shard: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Full ``[P, 128, D]`` index-K pages of ``pages`` from every rank's shard
    (token ``i * tp + r`` of a block comes from rank ``r``'s row ``i``)."""
    from vllm.distributed.parallel_state import get_tp_group

    if _GATHER is not None:
        full = _GATHER.gather(kv_shard, pages)
        if full is not None:
            return full
    group = get_tp_group()
    tp = group.world_size
    num_pages = pages.numel()
    _, rows, dim = kv_shard.shape
    local = kv_shard.view(torch.uint8).index_select(0, pages)
    gathered = group.all_gather(local, dim=0)
    full = (
        gathered.view(tp, num_pages, rows, dim)
        .permute(1, 2, 0, 3)
        .reshape(num_pages, rows * tp, dim)
    )
    return full.view(kv_shard.dtype)


def decode_score(
    index_q: torch.Tensor,
    kv_shard: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor,
    ordinal: int,
) -> None:
    """Exact per-block decode index scores of this rank's heads from the
    sharded cache. ``index_q`` is ``[nd, num_heads, D]`` (every index head);
    ``score_out`` is the ``[nd, hpr, tiles]`` unified-score view; ``ordinal``
    is the layer's ``register_layer`` ordinal."""
    from vllm.models.minimax_m3.nvidia.ops.index_decode_score_shard import (
        DEFAULT_SPLIT_K,
        minimax_m3_index_decode_score_shard_cutedsl,
    )

    comm = get_comm()
    assert comm is not None
    nd = index_q.shape[0]
    slot_set = comm.slot_set(ordinal)
    split_k = comm.decode_split(block_table.shape[0]) or DEFAULT_SPLIT_K
    minimax_m3_index_decode_score_shard_cutedsl(
        index_q,
        kv_shard,
        block_table,
        seq_lens,
        comm.rank,
        decode_query_len,
        max_decode_query_len,
        comm.recv_bases,
        slot_set,
        comm.max_decode_tokens,
        split_k,
    )
    comm.reduce_max(nd, seq_lens, decode_query_len, slot_set, score_out)


def shard_factor_from_spec(kv_cache_spec) -> int:
    """Shard factor encoded in an index-cache spec (1 when replicated)."""
    content = getattr(kv_cache_spec, "state_content_bytes", None)
    if content is None:
        return 1
    full = kv_cache_spec.head_size * kv_cache_spec.dtype.itemsize
    return full // content
