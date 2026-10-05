# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-head argmax over a frequency-ranked vocabulary subset (FR-Spec).

``VLLM_MM3_DRAFT_VOCAB_SUBSET=N`` restricts the greedy draft head (the
``get_top_tokens`` path of EAGLE3 / MTP drafters) to the ``N`` most frequent
tokens of a ranking (``draft_vocab_ranking``, or ``PATH[:N]`` for a ranking
file of token ids, most frequent first). Only the draft's proposals change: the
target's lm_head, sampling and rejection sampling still cover the whole
vocabulary, so the output distribution is the target's; a proposal differs
only when the full-vocab draft argmax lies outside the subset.

The head then streams ``N`` rows instead of a ``vocab / tp`` row shard per
rank. The subset rows are sorted by token id and re-partitioned evenly over
the TP ranks (rank ``r`` holds the ``r``-th block, gathered once from the
ranks that own the rows), with a local-to-global id map. The kernel is the
fused draft-head kernel's (``vocab_parallel_argmax``): bf16 logits from
``tl.dot`` with fp32 accumulation, 64-bit keys ``(ordered bf16 bits, ~row)``
folded per rank; the last CTA swaps the local row in the rank's best key for
its global token id before the keys are exchanged, so the result is the
argmax of the subset logits with the lowest token id on ties, as the stock
argmax over those rows. Blocks hold increasing token ids, so the tie order
is the global one. A block shorter than the others repeats its last row,
which cannot change its argmax.

The subset is built on the head's first eager call (collectives; never
during CUDA graph capture, where the full head is used instead), and every
rank falls back together if it cannot be built. ``0``/unset disables it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import vocab_parallel_argmax as vpa
from vllm.model_executor.layers.vocab_parallel_argmax import (
    _key_token_ids,
    _logit_keys,
)
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

logger = init_logger(__name__)

ENV = "VLLM_MM3_DRAFT_VOCAB_SUBSET"
# Largest draft batch on the subset path (as the fused kernel).
MAX_ROWS = vpa.MAX_ROWS
COLS = vpa.COLS
# Rows per all-gather while the subset is built (bounds its scratch).
_GATHER_ROWS = 1024
# (BLOCK_N, BLOCK_K, num_warps, num_stages) of the streaming GEMV per x
# halves, and CTAs per SM (persistent grid).
_CONFIG = (64, 128, 4, 5)
_WIDE_CONFIG = (64, 128, 4, 4)
_MAX_CTAS_PER_SM = 2
_USE_TMA = True

# Python-level dispatch counts (capture-time calls count once per graph).
CALLS = {"subset": 0, "fallback": 0}


@triton.jit
def _subset_argmax_kernel(
    x_ptr,
    w_ptr,
    w_desc,
    ids_ptr,
    out_ptr,
    best_ptr,
    count_ptr,
    epoch_ptr,
    err_ptr,
    recv_ptrs,
    M,
    n_valid,
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
    COLS: tl.constexpr = M_PAD // HALVES
    offs_c = tl.arange(0, COLS)
    c0 = offs_c < M
    c1 = (COLS + offs_c) < M
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_c[None, :] * stride_xm + offs_k[:, None]
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
        # Keys of local rows (vocab_start 0): the lowest row wins a tie.
        keys = _logit_keys(acc, rows, 0)
        keys = tl.where((rows < n_valid)[:, None] & c0[None, :], keys, 0)
        best = tl.maximum(best, tl.max(keys, axis=0))
        if HALVES > 1:
            keys1 = _logit_keys(acc1, rows, 0)
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
        # Local row -> global token id in the low word (every rank holds at
        # least one row, so every valid column has a key).
        row = _key_token_ids(local)
        gid = tl.load(ids_ptr + row, mask=m_mask & (local > 0), other=0)
        local = ((local >> 32) << 32) | (0xFFFFFFFF - gid.to(tl.int64))
        if EXCHANGE:
            slot = (epoch % 2) * WORLD * ROWS
            mask = (src[:, None] < WORLD) & m_mask[None, :]
            # As the fused kernel: the payload is the flag (keys are never 0).
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
            # Global keys for the NCCL fallback exchange.
            tl.store(out_ptr + offs_m, local, mask=m_mask)


class VocabSubset:
    """This rank's block of the subset: weight rows and their token ids."""

    def __init__(
        self, weight: torch.Tensor, ids: torch.Tensor, size: int, source: str
    ) -> None:
        self.weight = weight
        self.ids = ids
        self.n_valid = weight.shape[0]
        self.size = size
        self.source = source


def parse_env(value: str | None) -> tuple[str | None, int | None] | None:
    """``(path or None, size or None)`` of the knob, or None when off."""
    value = (value or "").strip()
    if value.lower() in ("", "0", "off", "false", "none"):
        return None
    if value.isdigit():
        return None, int(value)
    path, sep, size = value.rpartition(":")
    if sep and size.isdigit():
        return path, int(size)
    return value, None


def load_ranking(path: str | None, vocab_size: int) -> tuple[list[int], str]:
    """Token ids, most frequent first, and a description of their source."""
    if path is None:
        from vllm.model_executor.layers.draft_vocab_ranking import get_ranking

        ranking = get_ranking(vocab_size)
        if ranking is None:
            raise ValueError(f"no bundled draft vocab ranking for vocab {vocab_size}")
        return ranking
    with open(path) as f:
        return [int(t) for t in f.read().split()], path


def select_ids(ranking: list[int], size: int | None, vocab_size: int) -> list[int]:
    """The first ``size`` distinct in-vocab ids of the ranking."""
    seen: set[int] = set()
    ids: list[int] = []
    for t in ranking:
        if 0 <= t < vocab_size and t not in seen:
            seen.add(t)
            ids.append(t)
            if size is not None and len(ids) == size:
                break
    if size is not None and len(ids) < size:
        raise ValueError(f"ranking has {len(ids)} in-vocab ids, {size} requested")
    return ids


def _all_ok(group, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank of the TP group."""
    if group.world_size == 1:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def build(
    weight: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    ids: list[int] | None,
    source: str = "",
) -> VocabSubset | None:
    """Gather this rank's block of the subset rows from the ranks that own
    them. Every TP rank calls this with the same ids (None where they could
    not be loaded); None on every rank if any rank cannot build it."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    world, rank = group.world_size, group.rank_in_group
    dev = weight.device
    k = weight.shape[1]
    ok = ids is not None and len(set(ids)) >= world
    if not _all_ok(group, ok):
        return None
    ordered = sorted(set(ids))
    per = -(-len(ordered) // world)
    # Every block gets `per` ids; a short block repeats its last id.
    blocks = []
    for r in range(world):
        blk = ordered[r * per : (r + 1) * per] or ordered[-1:]
        blocks.append(blk + [blk[-1]] * (per - len(blk)))
    flat = torch.tensor([t for blk in blocks for t in blk], dtype=torch.int64)
    mine = flat[rank * per : (rank + 1) * per]
    if world == 1:
        rows = weight.index_select(0, (mine - vocab_start).to(dev)).contiguous()
        return VocabSubset(rows, mine.to(torch.int32).to(dev), len(ordered), source)
    ranges = [None] * world
    torch.distributed.all_gather_object(
        ranges, (vocab_start, vocab_end), group=group.cpu_group
    )
    owner = torch.full_like(flat, -1)
    for r, (lo, hi) in enumerate(ranges):
        owner[(flat >= lo) & (flat < hi)] = r
    try:
        out = torch.empty((per, k), dtype=weight.dtype, device=dev)
        send = torch.empty((_GATHER_ROWS, k), dtype=weight.dtype, device=dev)
        recv = torch.empty((world * _GATHER_ROWS, k), dtype=weight.dtype, device=dev)
        ok = not bool((owner < 0).any())
    except torch.OutOfMemoryError:
        ok = False
    if not _all_ok(group, ok):
        return None
    filled = torch.zeros(per, dtype=torch.bool)
    total = flat.numel()
    for c0 in range(0, total, _GATHER_ROWS):
        c1 = min(total, c0 + _GATHER_ROWS)
        ids_c, own_c = flat[c0:c1], owner[c0:c1]
        send.zero_()
        sel = (own_c == rank).nonzero().flatten()
        if sel.numel():
            send[sel.to(dev)] = weight.index_select(
                0, (ids_c[sel] - vocab_start).to(dev)
            )
        torch.distributed.all_gather_into_tensor(recv, send, group=group.device_group)
        # Rows of this chunk in this rank's block, copied from their owner.
        pos = torch.arange(c0, c1)
        keep = (pos >= rank * per) & (pos < (rank + 1) * per)
        if bool(keep.any()):
            src = own_c[keep] * _GATHER_ROWS + (pos[keep] - c0)
            dst = pos[keep] - rank * per
            out[dst.to(dev)] = recv.index_select(0, src.to(dev))
            filled[dst] = True
    del send, recv
    if not _all_ok(group, bool(filled.all())):
        return None
    return VocabSubset(out, mine.to(torch.int32).to(dev), len(ordered), source)


def launch(
    state: vpa._ArgmaxState,
    subset: VocabSubset,
    hidden_states: torch.Tensor,
    out: torch.Tensor,
    config: tuple[int, int, int, int] | None = None,
    max_ctas: int | None = None,
    use_tma: bool | None = None,
    exchange: bool | None = None,
) -> None:
    """Subset argmax into ``out``: token ids (int64, ``[M]``), or global keys
    when ``exchange`` is False on a TP group."""
    m, k = hidden_states.shape
    halves = 1 if m <= COLS else 2
    block_n, block_k, num_warps, num_stages = config or (
        _CONFIG if halves == 1 else _WIDE_CONFIG
    )
    w = subset.weight
    use_tma = (_USE_TMA if use_tma is None else use_tma) and w.data_ptr() % 16 == 0
    exchange = state.exchange if exchange is None else exchange
    num_tiles = triton.cdiv(subset.n_valid, block_n)
    max_ctas = _MAX_CTAS_PER_SM * state.num_sms if max_ctas is None else max_ctas
    grid = num_tiles if max_ctas <= 0 else min(num_tiles, max_ctas)
    w_desc = vpa._tma_desc(state, w, block_n, block_k) if use_tma else None
    _subset_argmax_kernel[(grid,)](
        hidden_states,
        w,
        w_desc,
        subset.ids,
        out,
        state.best,
        state.count,
        state.epoch,
        state.errors,
        state.peer_ptrs,
        m,
        subset.n_valid,
        hidden_states.stride(0),
        w.stride(0),
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
        ROWS=vpa.MAX_ROWS,
        SPIN=vpa.SPIN_LIMIT,
        num_warps=num_warps,
    )


def get_subset(
    lm_head: VocabParallelEmbedding, size: int | None, path: str | None = None
) -> VocabSubset | None:
    """The head's subset of ``size`` ranked ids, built on its first eager call
    (None while capturing, or if it cannot be built on every rank)."""
    cache = lm_head.__dict__.setdefault("_vocab_subsets", {})
    key = (path, size)
    if key in cache:
        return cache[key]
    if torch.cuda.is_current_stream_capturing():
        # Set-up uses collectives; this graph keeps the full head.
        logger.warning_once(
            "Draft vocab subset first reached during CUDA graph capture; "
            "this graph keeps the full draft head."
        )
        return None
    shard = lm_head.shard_indices
    ids, source, error = None, "", None
    try:
        ranking, source = load_ranking(path, lm_head.org_vocab_size)
        ids = select_ids(ranking, size, lm_head.org_vocab_size)
    except Exception as e:  # noqa: BLE001 - every rank falls back together
        error = e
    subset = build(
        lm_head.weight,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
        ids,
        source,
    )
    if subset is None:
        logger.warning(
            "Draft vocab subset %s disabled: %r", (path, size), error or "a rank failed"
        )
    else:
        logger.info(
            "Draft vocab subset active: %d of %d tokens (%s), %d rows x %d per "
            "rank (%.0f MB), local-to-global id map",
            subset.size,
            lm_head.org_vocab_size,
            subset.source,
            subset.n_valid,
            subset.weight.shape[1],
            subset.weight.numel() * subset.weight.element_size() / 1e6,
        )
    cache[key] = subset
    return subset


_SETTING: tuple | None = None


def _setting() -> tuple[str | None, int | None] | None:
    global _SETTING
    if _SETTING is None:
        _SETTING = (parse_env(os.environ.get(ENV)),)
    return _SETTING[0]


def try_launch(
    state: vpa._ArgmaxState,
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Subset argmax into ``out`` when the knob is set (token ids, or global
    keys without symmetric memory); False to use the full head instead."""
    setting = _setting()
    if setting is None:
        return False
    m = hidden_states.shape[0]
    subset = None
    if 0 < m <= MAX_ROWS:
        subset = get_subset(lm_head, setting[1], setting[0])
    if subset is None:
        CALLS["fallback"] += 1
        return False
    launch(state, subset, hidden_states, out)
    CALLS["subset"] += 1
    return True
