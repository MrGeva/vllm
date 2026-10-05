# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused vocab-parallel lm_head GEMV + argmax for greedy draft heads.

``LogitsProcessor.get_top_tokens`` otherwise materializes the local logits of
every draft step: a bf16 GEMM over the vocab shard (split-K plus its reduce
for decode-sized batches), a ``max`` over the logits, an NCCL all-gather of
the (value, index) pairs, then an argmax and a gather. Here one Triton kernel
streams the shard once and keeps only the best candidate:

- Every CTA computes logit tiles with ``tl.dot`` (fp32 accumulation) and
  rounds them to bf16, as the stock GEMM rounds its output. Each logit becomes
  a 64-bit key ``(order-preserving bf16 bits, ~token id)`` whose maximum is
  the stock argmax: the largest bf16 value, the lowest token id on ties
  (``torch.max`` keeps the first maximum and the argmax across ranks the
  lowest rank, whose shard holds the lower ids). NaN is the largest value, as
  in ``torch.max``; -0 equals +0.
- Each CTA folds its keys into a rank-local slot with ``atomic_max``. The last
  CTA to arrive takes the rank's best key, pushes it into every TP peer's
  symmetric receive slot, polls its own slots for the peers' keys (the payload
  is the flag; 0 is never a key) and writes the global token ids.

Receive slots alternate between two sets by call parity. A rank pushes call
``n + 2`` (the set of ``n``) only after it read every peer's call ``n + 1``
push, which each peer issued after finishing (and resetting) call ``n``, so no
barrier is needed. The spinning CTA waits only for other GPUs, never for a CTA
of its own grid (all of them have arrived). TP ranks call this in lockstep, as
they call the stock all-gather. Without symmetric memory the rank keys are
all-gathered with NCCL instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

logger = init_logger(__name__)

# Largest number of rows (draft batch) on the fused path: up to 16 one
# 16-column dot per K step, up to 32 two (both read the weights once).
MAX_ROWS = 32
COLS = 16
NUM_SLOT_SETS = 2
SPIN_LIMIT = 1 << 26

# (BLOCK_N, BLOCK_K, num_warps, num_stages) of the streaming GEMV, its grid
# (CTAs per SM; 0 = one CTA per row tile) and whether W streams through TMA.
# B300, EAGLE3 head (6144 x 100032 / 50016 rows per TP2 / TP4 shard): 182 /
# 94 us at M = 1 without the exchange, 6.7 / 6.6 TB/s.
_DEFAULT_CONFIG = (64, 128, 4, 5)
# Above 16 rows (two dots, a wider x tile): one stage less keeps two CTAs/SM.
_WIDE_CONFIG = (64, 128, 4, 4)
_MAX_CTAS_PER_SM = 2
_USE_TMA = True

_STATE: _ArgmaxState | None = None
_STATE_FAILED = False


@triton.jit
def _logit_keys(acc, rows, vocab_start):
    """Keys of bf16-rounded logits: max key = max value, then lowest id."""
    logits = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    bits = logits.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    # One NaN that sorts above +inf; -0 compares equal to +0.
    bits = tl.where((bits & 0x7FFF) > 0x7F80, 0x7FC0, bits)
    bits = tl.where(bits == 0x8000, 0, bits)
    ordered = tl.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)
    # Low word: ~token id, so that the lowest id wins a tie.
    tie = 0xFFFFFFFF - (vocab_start + rows).to(tl.int64)
    return (ordered.to(tl.int64) << 32) | tie[:, None]


@triton.jit
def _key_token_ids(keys):
    return 0xFFFFFFFF - (keys & 0xFFFFFFFF)


@triton.jit
def _vocab_parallel_argmax_kernel(
    x_ptr,
    w_ptr,
    w_desc,
    out_ptr,
    best_ptr,
    count_ptr,
    epoch_ptr,
    err_ptr,
    recv_ptrs,
    M,
    N,
    n_valid,
    vocab_start,
    stride_xm,
    stride_wn,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    HALVES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_TMA: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EXCHANGE: tl.constexpr,
    ROWS: tl.constexpr,
    SPIN: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = tl.cdiv(n_valid, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    # Up to 16 rows of x: one 16-column dot per K step; above, a second
    # 16-column dot (the same MMA shape, so the same keys per row).
    COLS: tl.constexpr = M_PAD // HALVES
    offs_c = tl.arange(0, COLS)
    c0 = offs_c < M
    c1 = (COLS + offs_c) < M
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_c[None, :] * stride_xm + offs_k[:, None]
    # Exchange addresses, loaded while the weights stream.
    epoch = tl.load(epoch_ptr)
    src = tl.arange(0, WORLD)
    bases = tl.load(recv_ptrs + src).to(tl.pointer_type(tl.int64))

    best = tl.zeros([COLS], dtype=tl.int64)
    best1 = tl.zeros([COLS], dtype=tl.int64)
    for tile in range(pid, num_tiles, nprog):
        rows = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
        acc = tl.zeros([BLOCK_N, COLS], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_N, COLS], dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            if USE_TMA:
                w = w_desc.load([tile * BLOCK_N, k])
            else:
                w = tl.load(w_ptrs + k, mask=(rows < n_valid)[:, None], other=0.0)
            xt = tl.load(x_ptrs + k, mask=c0[None, :], other=0.0)
            acc = tl.dot(w, xt, acc)
            if HALVES > 1:
                xt1 = tl.load(x_ptrs + COLS * stride_xm + k, mask=c1[None, :], other=0.0)
                acc1 = tl.dot(w, xt1, acc1)
        keys = _logit_keys(acc, rows, vocab_start)
        keys = tl.where((rows < n_valid)[:, None] & c0[None, :], keys, 0)
        best = tl.maximum(best, tl.max(keys, axis=0))
        if HALVES > 1:
            keys1 = _logit_keys(acc1, rows, vocab_start)
            keys1 = tl.where((rows < n_valid)[:, None] & c1[None, :], keys1, 0)
            best1 = tl.maximum(best1, tl.max(keys1, axis=0))

    # The returned old values feed the arrival increment, so every fold has
    # been performed before this CTA is counted (no key is above 2**48).
    old = tl.atomic_max(best_ptr + offs_c, best, mask=c0, sem="relaxed")
    dep = tl.max(tl.where(c0, old, 0), axis=0)
    if HALVES > 1:
        old1 = tl.atomic_max(best_ptr + COLS + offs_c, best1, mask=c1, sem="relaxed")
        dep = tl.maximum(dep, tl.max(tl.where(c1, old1, 0), axis=0))
    arrive = tl.where(dep < 0, 2, 1)
    done = tl.atomic_add(count_ptr, arrive, sem="acq_rel")
    if done == nprog - 1:
        local = tl.atomic_xchg(
            best_ptr + offs_m, tl.zeros([M_PAD], dtype=tl.int64), mask=m_mask
        )
        tl.store(count_ptr, 0)
        if EXCHANGE:
            slot = (epoch % 2) * WORLD * ROWS
            mask = (src[:, None] < WORLD) & m_mask[None, :]
            # Push this rank's keys to every peer at once. The payload is the
            # flag, so no fence is needed: a slot set is rewritten only after
            # its reader's kernel that reset it has completed (see above).
            tl.atomic_xchg(
                bases[:, None] + slot + RANK * ROWS + offs_m[None, :],
                tl.broadcast_to(local[None, :], [WORLD, M_PAD]),
                mask=mask,
                sem="relaxed",
                scope="sys",
            )
            own = tl.load(recv_ptrs + RANK).to(tl.pointer_type(tl.int64))
            ptrs = own + slot + src[:, None] * ROWS + offs_m[None, :]
            keys = tl.load(ptrs, mask=mask, other=1, volatile=True)
            spins = tl.full([], 0, tl.int32)
            while (tl.min(keys) == 0) & (spins < SPIN):
                keys = tl.load(ptrs, mask=mask, other=1, volatile=True)
                spins += 1
            if spins >= SPIN:
                tl.atomic_add(err_ptr, 1)
            local = tl.max(tl.where(mask, keys, 0), axis=0)
            tl.store(ptrs, tl.zeros([WORLD, M_PAD], dtype=tl.int64), mask=mask)
            tl.store(epoch_ptr, epoch + 1)
            tl.store(out_ptr + offs_m, _key_token_ids(local), mask=m_mask)
        elif WORLD == 1:
            tl.store(out_ptr + offs_m, _key_token_ids(local), mask=m_mask)
        else:
            # Keys for the NCCL fallback exchange.
            tl.store(out_ptr + offs_m, local, mask=m_mask)


class _ArgmaxState:
    """Per-process scratch and the symmetric receive slots of the TP group."""

    def __init__(self, device: torch.device) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.device = device
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.best = torch.zeros(MAX_ROWS, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self.recv: torch.Tensor | None = None
        self.peer_ptrs = torch.zeros(1, dtype=torch.int64, device=device)
        self.exchange = False
        self._tma_descs: dict = {}
        if self.world == 1:
            return
        error = None
        try:
            self._init_symm_mem(group)
        except Exception as e:  # noqa: BLE001 - every rank falls back together
            error = e
        ok = torch.tensor([int(self.recv is not None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        self.exchange = bool(ok.item())
        if not self.exchange:
            logger.warning_once(
                "Fused draft-head argmax exchanges keys with NCCL: symmetric "
                "memory set-up failed (%r)",
                error,
            )

    def _init_symm_mem(self, group) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        recv = symm_mem.empty(
            (NUM_SLOT_SETS, self.world, MAX_ROWS),
            dtype=torch.int64,
            device=self.device,
        )
        recv.zero_()
        handle = symm_mem.rendezvous(recv, group.device_group.group_name)
        assert handle is not None
        bases = [int(p) for p in handle.buffer_ptrs]
        assert bases[self.rank] == recv.data_ptr()
        self.peer_ptrs = torch.tensor(bases, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        handle.barrier()
        self.recv, self.handle = recv, handle


def _get_state(device: torch.device) -> _ArgmaxState | None:
    """Process-wide state, created collectively on the first (eager) call."""
    global _STATE, _STATE_FAILED
    if _STATE is not None or _STATE_FAILED:
        return _STATE
    if torch.cuda.is_current_stream_capturing():
        # The set-up synchronizes and rendezvouses; every rank reaches the
        # same call, so all of them use the stock path for this capture.
        logger.warning_once(
            "Fused draft-head argmax first reached during CUDA graph capture; "
            "this graph keeps the stock path."
        )
        return None
    try:
        _STATE = _ArgmaxState(device)
    except Exception as e:  # noqa: BLE001
        logger.warning("Fused draft-head argmax disabled: %r", e)
        _STATE_FAILED = True
        return None
    if _STATE.world == 1:
        exchange = "no"
    else:
        exchange = "symmetric-memory" if _STATE.exchange else "NCCL"
    logger.info_once(
        "Fused vocab-parallel draft-head argmax enabled (TP%d, %s exchange)",
        _STATE.world,
        exchange,
    )
    return _STATE


def supports_fused_argmax(
    lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
) -> bool:
    """Shape-independent eligibility; identical on every TP rank."""
    from vllm import envs
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    cached = getattr(lm_head, "_fused_argmax_ok", None)
    if cached is not None:
        return cached
    weight = getattr(lm_head, "weight", None)
    shard = lm_head.shard_indices
    ok = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and not envs.VLLM_BATCH_INVARIANT
        and type(lm_head.quant_method) is UnquantizedEmbeddingMethod
        and isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.stride(1) == 1
        and weight.shape[1] % _DEFAULT_CONFIG[1] == 0
        and hidden_states.dtype == torch.bfloat16
        and shard.num_added_elements_padded == 0
        and lm_head.tp_size & (lm_head.tp_size - 1) == 0
    )
    lm_head._fused_argmax_ok = ok
    return ok


def _tma_desc(state: _ArgmaxState, weight: torch.Tensor, block_n: int, block_k: int):
    from triton.tools.tensor_descriptor import TensorDescriptor

    key = (weight.data_ptr(), tuple(weight.shape), block_n, block_k)
    desc = state._tma_descs.get(key)
    if desc is None:
        desc = TensorDescriptor.from_tensor(weight, [block_n, block_k])
        state._tma_descs[key] = desc
    return desc


def launch(
    state: _ArgmaxState,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    n_valid: int,
    vocab_start: int,
    out: torch.Tensor,
    config: tuple[int, int, int, int] | None = None,
    max_ctas: int = 0,
    use_tma: bool = False,
    exchange: bool | None = None,
) -> None:
    """Run the kernel; ``out`` gets token ids (int64, ``[M]``), or the rank
    keys when ``exchange`` is False on a TP group."""
    m, k = hidden_states.shape
    halves = 1 if m <= COLS else 2
    block_n, block_k, num_warps, num_stages = config or (
        _DEFAULT_CONFIG if halves == 1 else _WIDE_CONFIG
    )
    n = weight.shape[0]
    exchange = state.exchange if exchange is None else exchange
    num_tiles = triton.cdiv(n, block_n)
    grid = num_tiles if max_ctas <= 0 else min(num_tiles, max_ctas)
    w_desc = _tma_desc(state, weight, block_n, block_k) if use_tma else None
    _vocab_parallel_argmax_kernel[(grid,)](
        hidden_states,
        weight,
        w_desc,
        out,
        state.best,
        state.count,
        state.epoch,
        state.errors,
        state.peer_ptrs,
        m,
        n,
        n_valid,
        vocab_start,
        hidden_states.stride(0),
        weight.stride(0),
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=COLS * halves,
        HALVES=halves,
        NUM_STAGES=num_stages,
        USE_TMA=use_tma,
        RANK=state.rank,
        WORLD=state.world,
        EXCHANGE=exchange and state.world > 1,
        ROWS=MAX_ROWS,
        SPIN=SPIN_LIMIT,
        num_warps=num_warps,
    )


def fused_vocab_parallel_argmax(
    lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    """Global greedy token ids ``[M]`` (int64), or None to use the stock path."""
    from vllm.model_executor.layers import vocab_parallel_screen, vocab_parallel_subset

    m = hidden_states.shape[0]
    if m == 0 or m > MAX_ROWS or hidden_states.stride(-1) != 1:
        return None
    state = _get_state(hidden_states.device)
    if state is None:
        return None
    weight = lm_head.weight
    shard = lm_head.shard_indices
    out = torch.empty(m, dtype=torch.int64, device=hidden_states.device)
    # VLLM_MM3_DRAFT_VOCAB_SUBSET: the argmax over a frequency-ranked subset
    # of the vocabulary (this kernel's math on the subset rows).
    if vocab_parallel_subset.try_launch(state, lm_head, hidden_states, out):
        exchanged = state.world == 1 or state.exchange
    # Same tokens from an int8 screen plus an exact pass over the few rows
    # that can hold the argmax (large shards); this kernel otherwise.
    elif vocab_parallel_screen.try_launch(lm_head, hidden_states, out):
        comm = vocab_parallel_screen._COMM
        exchanged = comm.world == 1 or comm.exchange
    else:
        launch(
            state,
            hidden_states,
            weight,
            shard.num_org_elements,
            shard.org_vocab_start_index,
            out,
            None,
            _MAX_CTAS_PER_SM * state.num_sms,
            _USE_TMA and weight.data_ptr() % 16 == 0,
        )
        exchanged = state.world == 1 or state.exchange
    if exchanged:
        return out
    from vllm.distributed import tensor_model_parallel_all_gather

    keys = tensor_model_parallel_all_gather(out.unsqueeze(-1), dim=-1)
    best = keys.max(dim=-1).values
    return 0xFFFFFFFF - (best & 0xFFFFFFFF)
