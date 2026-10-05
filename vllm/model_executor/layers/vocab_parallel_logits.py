# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel lm_head logits gathered through symmetric memory.

The stock target head runs the bf16 GEMM of the vocab shard (cuBLAS, split-K
plus its reduce kernel for decode-sized batches), an NCCL all-gather of the
``[M, shard]`` logits and a copy that moves the rank dimension behind the
rows. Both paths here return the same bf16 logits bit for bit:

- Push (one kernel). A Triton GEMV computes the shard's logits with the
  stock GEMM's own arithmetic: for decode-sized batches above a few rows
  cuBLAS runs one sequential K loop per output (no split-K), which the
  fused-kernel loop (16-wide tensor-core MMAs in K order, fp32 accumulation,
  bf16 rounding) reproduces bit for bit; split-K variants (contiguous or
  interleaved partial sums added in order) are tried for small batches. Each
  tile goes straight into every rank's symmetric ``[M, vocab]`` buffer. Every
  CTA then counts itself in on every rank (a system-scope release after a
  fence), and the last CTA of each rank waits for every rank's count. The
  variant that reproduces the stock GEMM is found per row count at the first
  eager call, on random inputs against the stock GEMM, and agreed on by all
  ranks.
- Pull. The stock GEMM writes the shard into the rank's symmetric slot; one
  kernel then flags it to every peer and copies the ``WORLD`` shards into the
  ``[M, vocab]`` output as each peer's flag arrives.

Where both paths are exact, the first eager call also times them on all ranks
in lockstep for buckets of row counts and keeps the faster (B300: push at TP2
up to 32 rows, pull at TP4, whose four destinations make the pushes slower).

Symmetric buffers alternate between two sets by call parity, and flags and
counts are monotonic in a per-process call counter. A rank writes set ``p``
again (call ``n + 2``) only after it saw every peer's call ``n + 1`` flag or
count, which each peer issued after its call ``n`` finished reading (pull) or
using (push) set ``p``. The push path returns a view of the buffer, valid
until the next call but one, as the stock logits are consumed within a step.
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

NUM_SETS = 2
SPIN_LIMIT = 1 << 26
# Upper limit on rows (logits per step) on these paths; the buffers are sized
# by the scheduler's batch (requests x tokens per request), capped here.
MAX_ROWS_CAP = 256
# K step of the stock split-K GEMM (64-element K tiles).
SPLIT_UNIT = 64
# (BLOCK_N, BLOCK_K, num_warps, num_stages) of the push GEMV per padded row
# count (M_PAD), and its CTAs per SM. Split variants use BLOCK_K 64.
# B300 TP4 shard: 93 / 97 / 108 / 119 us at M = 4 / 32 / 64 / 80 (cuBLAS ~101).
_GEMV_CONFIGS = {16: (64, 128, 4, 5), 32: (64, 128, 4, 4), 64: (128, 128, 4, 3),
                 128: (128, 128, 4, 3), 256: (128, 64, 8, 3)}
_PUSH_CTAS_PER_SM = 2
# CTAs of the pull gather per SM (split over the source ranks).
_PULL_CTAS_PER_SM = 2
# Split counts tried (contiguous and interleaved) for row counts up to
# SPLIT_MAX_ROWS, where cuBLAS picks split-K kernels.
_SPLITS = (3, 2, 4, 6)
SPLIT_MAX_ROWS = 16
# Row counts at which the first eager call times push against pull (both
# exact); each other row count follows the nearest bucket at or above it.
_TIME_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
_TIME_REPS = 8
ENABLED = True
PUSH_ENABLED = True

_STATE: _LogitsState | None = None
_STATE_FAILED = False


def split_variants(max_m_split: bool = True) -> list[tuple[int, bool]]:
    """Splits of K tried against the stock GEMM: none (its large-M kernels),
    then contiguous or interleaved (every NSPLIT-th K step) partial sums."""
    out = [(1, False)]
    if max_m_split:
        out += [(n, il) for n in _SPLITS for il in (False, True)]
    return out


@triton.jit
def _fence_sys(x):
    return tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gemv_logits_kernel(
    x_ptr,
    w_desc,
    out_ptr,
    buf_ptrs,
    cnt_ptrs,
    arrive_ptr,
    epoch_ptr,
    err_ptr,
    M,
    n_valid,
    col0,
    ld_out,
    set_offset,
    stride_xm,
    K: tl.constexpr,
    NSPLIT: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    PUSH: tl.constexpr,
    SPIN: tl.constexpr,
):
    """bf16 logits of the shard with the stock GEMM's arithmetic: NSPLIT
    partial sums over K (contiguous chunks, or with INTERLEAVE every NSPLIT-th
    BLOCK_K step), each accumulated by 16-wide tensor-core MMAs in K order and
    added in split order. PUSH: into every rank's buffer set (columns
    ``col0 + rows``), then the call's cross-rank count; else into ``out``."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = tl.cdiv(n_valid, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    epoch = tl.load(epoch_ptr)
    CHUNK: tl.constexpr = K // NSPLIT
    STEPS: tl.constexpr = K // (BLOCK_K * NSPLIT)

    for tile in range(pid, num_tiles, nprog):
        rows = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        total = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
        for s in tl.static_range(NSPLIT):
            acc = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
            if INTERLEAVE:
                for i in tl.range(0, STEPS, num_stages=NUM_STAGES):
                    k = (s + i * NSPLIT) * BLOCK_K
                    w = w_desc.load([tile * BLOCK_N, k])
                    xt = tl.load(x_ptrs + k, mask=m_mask[None, :], other=0.0)
                    acc = tl.dot(w, xt, acc)
            else:
                for k in tl.range(s * CHUNK, (s + 1) * CHUNK, BLOCK_K, num_stages=NUM_STAGES):
                    w = w_desc.load([tile * BLOCK_N, k])
                    xt = tl.load(x_ptrs + k, mask=m_mask[None, :], other=0.0)
                    acc = tl.dot(w, xt, acc)
            if NSPLIT == 1:
                total = acc
            else:
                # The reduce adds the split partials in order (fp32, from 0).
                total = total + acc
        logits = total.to(tl.bfloat16, fp_downcast_rounding="rtne")
        offs = set_offset + offs_m[None, :] * ld_out + col0 + rows[:, None]
        ok = (rows < n_valid)[:, None] & m_mask[None, :]
        if PUSH:
            for d in tl.static_range(WORLD):
                dst = tl.load(buf_ptrs + d).to(tl.pointer_type(tl.bfloat16))
                tl.store(dst + offs, logits, mask=ok)
        else:
            tl.store(out_ptr + offs, logits, mask=ok)

    if PUSH:
        # Count this CTA in on every rank once its stores are visible
        # system-wide: the CTA barrier, then a system-scope fence (cumulative
        # over the CTA's stores) ahead of relaxed counts, one fence per CTA
        # instead of one per release.
        tl.debug_barrier()
        f = _fence_sys(tl.full([], 0, tl.int32))
        for d in tl.static_range(WORLD):
            cbase = tl.load(cnt_ptrs + d).to(tl.pointer_type(tl.int64))
            tl.atomic_add(cbase + RANK, 1 + f.to(tl.int64), sem="relaxed", scope="sys")
        done = tl.atomic_add(arrive_ptr, 1, sem="acq_rel")
        if done == nprog - 1:
            # The last CTA waits for every rank's CTAs of this call.
            src = tl.arange(0, WORLD)
            own = tl.load(cnt_ptrs + RANK).to(tl.pointer_type(tl.int64))
            target = (epoch.to(tl.int64) + 1) * nprog
            got = tl.atomic_add(own + src, 0, sem="acquire", scope="sys")
            spins = tl.full([], 0, tl.int32)
            while (tl.min(got) < target) & (spins < SPIN):
                got = tl.atomic_add(own + src, 0, sem="acquire", scope="sys")
                spins += 1
            if spins >= SPIN:
                tl.atomic_add(err_ptr, 1)
            tl.store(arrive_ptr, 0)
            tl.store(epoch_ptr, epoch + 1)


@triton.jit
def _pull_logits_kernel(
    out_ptr,
    slot_ptrs,
    flag_ptrs,
    arrive_ptr,
    epoch_ptr,
    err_ptr,
    M,
    n,
    ld_out,
    set_offset,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    SPIN: tl.constexpr,
):
    """Copy every rank's [M, n] slot into ``out`` [M, WORLD * n] once its flag
    of this call arrived; flag this rank's slot to every peer first."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    per_src = nprog // WORLD
    p = pid // per_src
    c = pid % per_src
    epoch = tl.load(epoch_ptr)
    want = epoch.to(tl.int64) + 1
    src = tl.arange(0, WORLD)
    fbases = tl.load(flag_ptrs + src).to(tl.pointer_type(tl.int64))
    # Idempotent in every CTA (no CTA depends on another being resident).
    tl.atomic_max(
        fbases + RANK,
        tl.zeros([WORLD], dtype=tl.int64) + want,
        sem="release",
        scope="sys",
    )
    own = tl.load(flag_ptrs + RANK).to(tl.pointer_type(tl.int64))
    seen = tl.atomic_add(own + p, 0, sem="acquire", scope="sys")
    spins = tl.full([], 0, tl.int32)
    while (seen < want) & (spins < SPIN):
        seen = tl.atomic_add(own + p, 0, sem="acquire", scope="sys")
        spins += 1
    if spins >= SPIN:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()
    slot = tl.load(slot_ptrs + p).to(tl.pointer_type(tl.bfloat16)) + set_offset
    nb = tl.cdiv(n, BLOCK)
    cols = tl.arange(0, BLOCK)
    for idx in range(c, M * nb, per_src):
        r = idx // nb
        cc = (idx - r * nb) * BLOCK + cols
        ok = cc < n
        v = tl.load(slot + r * n + cc, mask=ok, cache_modifier=".cv")
        tl.store(out_ptr + r.to(tl.int64) * ld_out + p * n + cc, v, mask=ok)
    done = tl.atomic_add(arrive_ptr, 1, sem="acq_rel")
    if done == nprog - 1:
        tl.store(arrive_ptr, 0)
        tl.store(epoch_ptr, epoch + 1)


class _LogitsState:
    """Symmetric buffers, flags and counters of the TP group."""

    def __init__(self, device: torch.device, n: int, k: int, max_rows: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.device = device
        self.n, self.k, self.max_rows = n, k, max_rows
        self.vocab = n * self.world
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        name = group.device_group.group_name
        # Push: full [rows, vocab] per set; pull: the rank's [rows, n] per set.
        self.full = symm_mem.empty(
            (NUM_SETS, max_rows, self.vocab), dtype=torch.bfloat16, device=device
        )
        self.slot = symm_mem.empty(
            (NUM_SETS, max_rows, n), dtype=torch.bfloat16, device=device
        )
        # [push counts | pull flags] per source rank, monotonic per call.
        self.sig = symm_mem.empty((2, self.world), dtype=torch.int64, device=device)
        self.sig.zero_()
        handles = [
            symm_mem.rendezvous(t, name) for t in (self.full, self.slot, self.sig)
        ]
        assert all(h is not None for h in handles)
        ptrs = [[int(p) for p in h.buffer_ptrs] for h in handles]
        assert ptrs[0][self.rank] == self.full.data_ptr()
        self.full_ptrs = torch.tensor(ptrs[0], dtype=torch.int64, device=device)
        self.slot_ptrs = torch.tensor(ptrs[1], dtype=torch.int64, device=device)
        es = self.sig.element_size() * self.world
        self.cnt_ptrs = torch.tensor(ptrs[2], dtype=torch.int64, device=device)
        self.flag_ptrs = torch.tensor(
            [p + es for p in ptrs[2]], dtype=torch.int64, device=device
        )
        self.arrive = torch.zeros(2, dtype=torch.int32, device=device)
        self.push_epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.pull_epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        # Host call counts: the buffer set of each path alternates per call.
        self.push_calls = 0
        self.pull_calls = 0
        self._handles = handles
        torch.cuda.synchronize(device)
        handles[2].barrier()
        # (NSPLIT, INTERLEAVE) per variant id, and the exact variant per row
        # count (-1: pull path, -2: stock path).
        self.variants: list[tuple[int, bool]] = split_variants()
        self.table: dict[int, int] = {}
        self.timings: dict[int, tuple[float, float]] = {}
        self._descs: dict = {}
        per_src = max(1, _PULL_CTAS_PER_SM * self.num_sms // self.world)
        self.pull_grid = self.world * per_src

    def desc(self, weight: torch.Tensor, block_n: int, block_k: int):
        from triton.tools.tensor_descriptor import TensorDescriptor

        key = (weight.data_ptr(), tuple(weight.shape), block_n, block_k)
        d = self._descs.get(key)
        if d is None:
            d = TensorDescriptor.from_tensor(weight, [block_n, block_k])
            self._descs[key] = d
        return d


def max_rows_from_config(cfg) -> int:
    """Rows (logits per step) to size the buffers for: requests per step
    (capped by the active-sequence cap) times tokens per request."""
    if cfg is None or cfg.scheduler_config is None:
        return 0
    sched = cfg.scheduler_config
    reqs = sched.max_num_seqs
    if getattr(sched, "max_num_active_seqs", None):
        reqs = min(reqs, sched.max_num_active_seqs)
    spec = cfg.speculative_config
    per_req = 1 + ((spec.num_speculative_tokens or 0) if spec is not None else 0)
    return max(16, min(MAX_ROWS_CAP, reqs * per_req))


def supports_fused_logits(
    lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
) -> bool:
    """Shape-independent eligibility; identical on every TP rank."""
    from vllm import envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    cached = getattr(lm_head, "_fused_logits_ok", None)
    if cached is not None:
        return cached
    weight = getattr(lm_head, "weight", None)
    ok = (
        ENABLED
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and not envs.VLLM_BATCH_INVARIANT
        and lm_head.tp_size > 1
        # A quant config that excludes the head hands out the linear method.
        and type(lm_head.quant_method)
        in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        and isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and weight.shape[1] % SPLIT_UNIT == 0
        and hidden_states.dtype == torch.bfloat16
    )
    lm_head._fused_logits_ok = ok
    return ok


def _m_pad(m: int) -> int:
    return max(16, triton.next_power_of_2(m))


def launch_gemv(
    state: _LogitsState,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
    parity: int = 0,
    config: tuple[int, int, int, int] | None = None,
    ctas_per_sm: int | None = None,
) -> None:
    """The emulating GEMV: into ``out`` ([M, n], local only) or, without
    ``out``, pushed into buffer set ``parity`` of every rank."""
    m, k = hidden_states.shape
    nsplit, interleave = state.variants[variant]
    m_pad = _m_pad(m)
    block_n, block_k, num_warps, num_stages = config or _GEMV_CONFIGS[m_pad]
    if interleave or (k // nsplit) % block_k:
        block_k = SPLIT_UNIT
    cps = _PUSH_CTAS_PER_SM if ctas_per_sm is None else ctas_per_sm
    n = weight.shape[0]
    push = out is None
    grid = min(triton.cdiv(n, block_n), cps * state.num_sms)
    _gemv_logits_kernel[(grid,)](
        hidden_states,
        state.desc(weight, block_n, block_k),
        hidden_states if push else out,
        state.full_ptrs,
        state.cnt_ptrs,
        state.arrive,
        state.push_epoch,
        state.errors,
        m,
        n,
        state.rank * n if push else 0,
        state.vocab if push else out.stride(0),
        parity * state.max_rows * state.vocab if push else 0,
        hidden_states.stride(0),
        K=k,
        NSPLIT=nsplit,
        INTERLEAVE=interleave,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        NUM_STAGES=num_stages,
        RANK=state.rank,
        WORLD=state.world,
        PUSH=push,
        SPIN=SPIN_LIMIT,
        num_warps=num_warps,
    )


def launch_pull(
    state: _LogitsState,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    out: torch.Tensor,
    parity: int,
) -> None:
    """Stock GEMM into this rank's slot of set ``parity``, then the gather of
    every rank's slot into ``out`` [M, vocab]."""
    m = hidden_states.shape[0]
    slot = state.slot[parity, :m]
    torch.mm(hidden_states, lm_head.weight.t(), out=slot)
    _pull_logits_kernel[(state.pull_grid,)](
        out,
        state.slot_ptrs,
        state.flag_ptrs,
        state.arrive[1:],
        state.pull_epoch,
        state.errors,
        m,
        state.n,
        out.stride(0),
        parity * state.max_rows * state.n,
        RANK=state.rank,
        WORLD=state.world,
        BLOCK=2048,
        SPIN=SPIN_LIMIT,
        num_warps=4,
    )


def calibrate(
    state: _LogitsState,
    lm_head: VocabParallelEmbedding,
    rows: list[int] | None = None,
    push: bool = True,
) -> dict[int, int]:
    """Per row count, the split of K whose emulation equals the stock GEMM bit
    for bit on random inputs (local launches only), else whether the GEMM into
    the pull slot does; agreed by all ranks."""
    from vllm.distributed.parallel_state import get_tp_group

    weight = lm_head.weight
    rows = rows or list(range(1, state.max_rows + 1))
    gen = torch.Generator(device=state.device).manual_seed(20261002)
    x_all = torch.randn(
        max(rows), state.k, generator=gen, device=state.device, dtype=torch.float32
    ).to(torch.bfloat16)
    scratch = torch.empty(max(rows), state.n, dtype=torch.bfloat16, device=state.device)
    # Per row count: the exact push variant (or -1) and whether the stock
    # GEMM into the pull slot equals the stock one.
    found = torch.full((len(rows),), -1, dtype=torch.int32)
    pull_ok = torch.zeros(len(rows), dtype=torch.int32)
    prev = 0
    for i, m in enumerate(rows):
        x = x_all[:m]
        ref = lm_head.quant_method.apply(lm_head, x).view(torch.int16)
        if push:
            # Splits only where cuBLAS picks split-K kernels (small M), and
            # only those that cut K into whole 64-element steps.
            tried = [
                v
                for v, (nsplit, _) in enumerate(state.variants)
                if state.k % (nsplit * SPLIT_UNIT) == 0
                and (v == 0 or m <= SPLIT_MAX_ROWS)
            ]
            order = ([prev] if prev in tried else []) + [v for v in tried if v != prev]
            for v in order:
                out = scratch[:m]
                launch_gemv(state, x, weight, v, out=out)
                if torch.equal(out.view(torch.int16), ref):
                    found[i] = v
                    prev = v
                    break
        slot = state.slot[0, :m]
        torch.mm(x, weight.t(), out=slot)
        pull_ok[i] = int(torch.equal(slot.view(torch.int16), ref))
    group = get_tp_group()

    def agree(t: torch.Tensor) -> list[int]:
        lo, hi = t.clone(), t.clone()
        torch.distributed.all_reduce(
            lo, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        torch.distributed.all_reduce(
            hi, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        return [int(a) if int(a) == int(b) else -1 for a, b in zip(lo, hi)]

    push_v = agree(found)
    pull = [v == 1 for v in agree(pull_ok)]
    # Where both paths are exact, the faster one per bucket of row counts,
    # timed on all ranks in lockstep (the slower rank's time decides).
    use_push = [v >= 0 for v in push_v]
    both = [i for i in range(len(rows)) if push_v[i] >= 0 and pull[i]]
    if both and push:
        idx = {m: i for i, m in enumerate(rows)}
        buckets = [m for m in _TIME_BUCKETS if m in idx and idx[m] in both]
        times = torch.zeros(len(buckets), 2, dtype=torch.float64)
        for j, m in enumerate(buckets):
            times[j] = torch.tensor(_time_paths(state, lm_head, x_all[:m], push_v[idx[m]]))
        torch.distributed.all_reduce(
            times, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        state.timings = {m: tuple(round(float(t), 2) for t in times[j]) for j, m in enumerate(buckets)}
        for i in both:
            m = rows[i]
            # the nearest timed bucket at or above m (else the largest)
            j = next((j for j, b in enumerate(buckets) if b >= m), len(buckets) - 1)
            if buckets:
                use_push[i] = bool(times[j, 0] < times[j, 1])
    table = {}
    for i, m in enumerate(rows):
        if push_v[i] >= 0 and use_push[i]:
            table[m] = push_v[i]
        elif pull[i]:
            table[m] = -1
        else:
            table[m] = -2
    state.table.update(table)
    return table


def _time_paths(
    state: _LogitsState, lm_head: VocabParallelEmbedding, x: torch.Tensor, variant: int
) -> tuple[float, float]:
    """us per call of the push and the pull path on x (collective)."""
    m = x.shape[0]
    out = torch.empty(m, state.vocab, dtype=torch.bfloat16, device=x.device)

    def push_once() -> None:
        parity = state.push_calls % NUM_SETS
        state.push_calls += 1
        launch_gemv(state, x, lm_head.weight, variant, parity=parity)

    def pull_once() -> None:
        parity = state.pull_calls % NUM_SETS
        state.pull_calls += 1
        launch_pull(state, x, lm_head, out, parity)

    res = []
    for fn in (push_once, pull_once):
        for _ in range(2):
            fn()
        torch.cuda.synchronize(state.device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(_TIME_REPS):
            fn()
        e1.record()
        e1.synchronize()
        res.append(e0.elapsed_time(e1) * 1000.0 / _TIME_REPS)
    return res[0], res[1]


def _get_state(
    lm_head: VocabParallelEmbedding, max_rows: int
) -> _LogitsState | None:
    global _STATE, _STATE_FAILED
    weight = lm_head.weight
    if _STATE is not None:
        if (_STATE.n, _STATE.k) == tuple(weight.shape):
            return _STATE
        return None
    if _STATE_FAILED or max_rows <= 0 or torch.cuda.is_current_stream_capturing():
        return None
    from vllm.distributed.parallel_state import get_tp_group

    error = None
    state = None
    try:
        state = _LogitsState(weight.device, weight.shape[0], weight.shape[1], max_rows)
        calibrate(state, lm_head, push=PUSH_ENABLED)
    except Exception as e:  # noqa: BLE001 - every rank falls back together
        error = e
        state = None
    group = get_tp_group()
    ok = torch.tensor([int(state is not None)], dtype=torch.int32)
    torch.distributed.all_reduce(
        ok, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    if not ok.item():
        logger.warning("Symmetric-memory lm_head logits disabled: %r", error)
        _STATE_FAILED = True
        return None
    _STATE = state
    exact = sorted(m for m, v in state.table.items() if v >= 0)
    pull = sorted(m for m, v in state.table.items() if v == -1)
    logger.info(
        "lm_head logits gathered through symmetric memory (TP%d, up to %d rows; "
        "exact push GEMV for row counts %s, stock GEMM + pull for %s; push / "
        "pull us by rows: %s)",
        state.world,
        state.max_rows,
        _ranges(exact),
        _ranges(pull),
        state.timings,
    )
    return _STATE


def _ranges(values: list[int]) -> str:
    if not values:
        return "none"
    out, start, prev = [], values[0], values[0]
    for v in values[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ",".join(out)


def fused_vocab_parallel_logits(
    lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor, max_rows: int
) -> torch.Tensor | None:
    """Gathered logits ``[M, vocab]`` (bf16, as the stock GEMM + all-gather),
    or None to use the stock path. Every TP rank calls this in lockstep;
    ``max_rows`` sizes the buffers at the first call."""
    m = hidden_states.shape[0]
    if m == 0 or torch.cuda.is_current_stream_capturing():
        return None
    state = _get_state(lm_head, max_rows)
    if state is None or m > state.max_rows:
        return None
    x = hidden_states if hidden_states.stride(-1) == 1 else hidden_states.contiguous()
    variant = state.table.get(m, -2)
    if variant < -1:
        return None
    if variant >= 0:
        parity = state.push_calls % NUM_SETS
        state.push_calls += 1
        launch_gemv(state, x, lm_head.weight, variant, parity=parity)
        return state.full[parity, :m]
    parity = state.pull_calls % NUM_SETS
    state.pull_calls += 1
    out = torch.empty(m, state.vocab, dtype=torch.bfloat16, device=x.device)
    launch_pull(state, x, lm_head, out, parity)
    return out
