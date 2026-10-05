# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact int8-screened draft-head argmax: the fused kernel's tokens, fewer bytes.

The fused draft-head kernel (``vocab_parallel_argmax``) streams the whole bf16
vocab shard every draft step, although only its argmax is used. Here a screen
kernel streams a one-byte copy of the shard instead and proves which rows
cannot win; an exact kernel recomputes only the remaining rows:

1. Screen. An int8 copy of the shard with a per-row symmetric scale ``s``
   gives ``est = s * (q . x)`` (``tl.dot`` on the int8 values converted
   exactly to bf16, fp32 accumulation). For every row
   ``|a - est| <= c * ||x||_2``, where ``a`` is the fp32 logit the fused
   kernel accumulates before rounding, and
   ``c = ||w - s q||_2 + GAMMA * (||w||_2 + ||s q||_2)``: Cauchy-Schwarz on
   the quantization residual plus the worst-case accumulation error of both
   tensor-core dot products. ``c`` is precomputed in fp64 and rounded up.
   Every row's upper bound (and every 64-row tile's largest) goes to scratch
   per column of x; the rank's largest lower bound ``tau`` is folded with
   ``atomic_max``. The last CTA reads the rank's ``tau`` and exchanges it with
   the peers through symmetric memory: the largest bounds a row of the
   vocabulary from below, so every rank prunes with it.
2. Cut. A row whose upper bound is below ``tau`` minus two bf16 ulps of
   ``tau`` rounds to a bf16 logit strictly below that of the row attaining
   ``tau``: it can neither win nor tie.
3. Exact pass (a programmatic dependent launch). Each CTA recomputes, for
   each of its fused-kernel tiles (64 rows) that holds a candidate, only the
   candidate rows: they are prefetched into L2, the other rows' weights are
   masked off (not read) and their keys dropped. The tile math is the fused
   kernel's: the same 64 x 16 ``tl.dot`` along K in the same order (fp32
   accumulation, one 16-column dot per 16 columns of x), the same bf16
   rounding and 64-bit keys, so every key, and thus every token id, equals
   the fused kernel's bit for bit. The ranks then fold and exchange their
   keys as the fused kernel does (own receive slots for up to ``MAX_M`` rows
   of x; a rank without candidates sends 1, below every key). A non-finite
   bound (non-finite or huge hidden states) makes every row a candidate: a
   full exact pass.

For the B300 EAGLE3 head the screen reads 615 MB per draft step at TP2 instead
of 1.23 GB, and live draft hidden states leave a handful of candidate rows in
all (pruned with the global ``tau``, ranks without the argmax keep almost
none). Draft batches of 17-32 rows run here too, as two 16-column exact dots
per tile. Narrower shards (TP4: 615 MB of bf16) keep the fused kernel, which
streams them about as fast as the screen plus the exact pass.
"""

from __future__ import annotations

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

# Row tile of the exact (fused-kernel) pass; a screen tile covers whole ones.
TILE_N = 64
# Columns of x per exact dot: the fused kernel's M_PAD.
EXACT_COLS = 16
# Largest draft batch on the screened path (two exact dots per tile above 16).
MAX_M = 32
# Worst-case relative accumulation error of a K = 6144 bf16 tensor-core dot
# product (exact products; per 16-wide MMA, terms aligned to >= 24 bits and
# truncated, then the sum rounded): ~8.3e-4, rounded up.
GAMMA = 2.0**-10
# Slack for the fp32 arithmetic of the bound and of ||x||.
REL_SLACK = 2.0**-20
XN_SLACK = 1.0 + 2.0**-10
# Hidden states with a larger norm (or a non-finite one) take the full pass.
MAX_XNORM = 1e30
# Ordered int32 key of -inf (see _float_key).
NEG_INF_KEY = -2139095041
# (BLOCK_N, BLOCK_K, num_warps, num_stages) of the int8 screen and its CTAs
# per SM.
_SCREEN_CONFIG = (128, 128, 4, 4)
_SCREEN_CTAS_PER_SM = 4
_SCREEN_TMA = True
# (BLOCK_K, num_stages) of the exact pass per M_PAD (its own kernel, one CTA
# per SM).
# Its tile stays the fused kernel's 64 x 16 dot and its MMA sequence along K
# the fused kernel's own: any BLOCK_K issues the same 16-wide MMAs in the
# same order (64-bit keys identical on every row at TP2 and TP4). A few
# candidate rows make it latency-bound: a deep pipeline of narrow K steps.
_EXACT_CONFIGS = {16: (256, 4), 32: (128, 8)}
_EXACT_CTAS_PER_SM = 1
# Shards below this many rows (TP4 and narrower) keep the fused kernel: its
# stream is short enough that the screen plus the exact pass's dependent MMA
# chain (~15 us for one row) saves little (B300, EAGLE3 head).
MIN_SHARD_ROWS = 65536
# Pull each candidate row into L2 before its K loop (one bulk prefetch).
PREFETCH = True
# Launch the exact kernel as a programmatic dependent of the screen kernel.
USE_PDL = True
# Prune with the largest tau of all ranks (exchanged by the screen kernel's
# last CTA) rather than the rank's own: ranks without the argmax keep (almost)
# no candidates.
GLOBAL_TAU = True
# Rows per chunk of the int8 copy's preparation (bounds its fp64 scratch).
_PREP_ROWS = 1024
ENABLED = True


@triton.jit
def _float_key(f):
    """int32 whose signed order is the order of the fp32 ``f``."""
    bits = f.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _key_float(key):
    bits = tl.where(key >= 0, key, key ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _exact_keys(
    x_ptr,
    w_ptr,
    rows,
    hit,
    M,
    vocab_start,
    stride_xm,
    stride_wn,
    K: tl.constexpr,
    TILE: tl.constexpr,
    COLS: tl.constexpr,
    HALVES: tl.constexpr,
    E_BLOCK_K: tl.constexpr,
    E_STAGES: tl.constexpr,
):
    """Max fused-kernel key per column of x over the rows of one fused tile
    flagged in ``hit`` (the other rows are not read), for columns
    ``[0, COLS)`` and, with two halves, ``[COLS, 2 COLS)``: one 64 x 16 dot
    per 16 columns, the fused kernel's MMA sequence along K."""
    offs_c = tl.arange(0, COLS)
    c0 = offs_c < M
    c1 = (COLS + offs_c) < M
    offs_k = tl.arange(0, E_BLOCK_K)
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x0_ptrs = x_ptr + offs_c[None, :] * stride_xm + offs_k[:, None]
    x1_ptrs = x0_ptrs + COLS * stride_xm
    acc0 = tl.zeros([TILE, COLS], dtype=tl.float32)
    acc1 = tl.zeros([TILE, COLS], dtype=tl.float32)
    for k in tl.range(0, K, E_BLOCK_K, num_stages=E_STAGES):
        w = tl.load(w_ptrs + k, mask=hit[:, None], other=0.0)
        x0 = tl.load(x0_ptrs + k, mask=c0[None, :], other=0.0)
        acc0 = tl.dot(w, x0, acc0)
        if HALVES > 1:
            x1 = tl.load(x1_ptrs + k, mask=c1[None, :], other=0.0)
            acc1 = tl.dot(w, x1, acc1)
    keys0 = _logit_keys(acc0, rows, vocab_start)
    keys0 = tl.where(hit[:, None] & c0[None, :], keys0, 0)
    keys1 = _logit_keys(acc1, rows, vocab_start)
    keys1 = tl.where(hit[:, None] & c1[None, :], keys1, 0)
    return tl.max(keys0, axis=0), tl.max(keys1, axis=0)


@triton.jit
def _prefetch_rows(w_ptr, rows, hit, stride_wn, NBYTES: tl.constexpr):
    """Pull the flagged weight rows into L2 (one bulk prefetch per row)."""
    addr = (w_ptr + rows.to(tl.int64) * stride_wn).to(tl.int64)
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; "
        "@p cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
        constraints="=r,l,r,r",
        args=[addr, hit.to(tl.int32), tl.full(rows.shape, NBYTES, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _screen_kernel(
    x_ptr,
    x_desc,
    q_ptr,
    q_desc,
    scale_ptr,
    coef_ptr,
    up_ptr,
    upt_ptr,
    lo_ptr,
    gtau_ptr,
    cnt_ptr,
    epoch_ptr,
    err_ptr,
    recv_ptrs,
    M,
    n_valid,
    stride_xm,
    stride_qn,
    up_stride,
    upt_stride,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE: tl.constexpr,
    M_PAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_TMA: tl.constexpr,
    X_TMA: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    GLOBAL_TAU: tl.constexpr,
    ROWS: tl.constexpr,
    SPIN: tl.constexpr,
    REL: tl.constexpr,
    XN_SCALE: tl.constexpr,
    MAX_XN: tl.constexpr,
    LO_RESET: tl.constexpr,
    PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if PDL:
        # The exact kernel's CTAs may take SMs as this grid drains.
        tl.extra.cuda.gdc_launch_dependents()
    num_tiles = tl.cdiv(n_valid, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    HALVES: tl.constexpr = BLOCK_N // TILE

    best_lo = tl.full([M_PAD], float("-inf"), tl.float32)
    xnorm = tl.zeros([M_PAD], dtype=tl.float32)
    for tile in range(pid, num_tiles, nprog):
        rows = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = rows < n_valid
        q_ptrs = q_ptr + rows[:, None].to(tl.int64) * stride_qn + offs_k[None, :]
        acc = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
        sq = tl.zeros([BLOCK_K, M_PAD], dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            if USE_TMA:
                q = q_desc.load([tile * BLOCK_N, k])
            else:
                q = tl.load(q_ptrs + k, mask=row_ok[:, None], other=0)
            if X_TMA:
                # [M_PAD, BLOCK_K] box, rows past M zero-filled
                xt = tl.trans(x_desc.load([0, k]))
            else:
                xt = tl.load(x_ptrs + k, mask=m_mask[None, :], other=0.0)
            xf = xt.to(tl.float32)
            sq += xf * xf
            acc = tl.dot(q.to(xt.dtype), xt, acc)
        xnorm = tl.sqrt(tl.sum(sq, axis=0)) * XN_SCALE
        s = tl.load(scale_ptr + rows, mask=row_ok, other=0.0)
        c = tl.load(coef_ptr + rows, mask=row_ok, other=0.0)
        est = acc * s[:, None]
        bound = c[:, None] * xnorm[None, :]
        bound += (tl.abs(est) + bound) * REL
        ok = row_ok[:, None] & m_mask[None, :]
        up = tl.where(ok, est + bound, float("-inf"))
        tl.store(up_ptr + offs_m[None, :] * up_stride + rows[:, None], up, mask=ok)
        # Largest upper bound of each fused tile ([M, tiles] layout).
        upt = tl.max(tl.reshape(up, [HALVES, TILE, M_PAD]), axis=1)
        t = tile * HALVES + tl.arange(0, HALVES)
        tl.store(
            upt_ptr + offs_m[None, :] * upt_stride + t[:, None], upt, mask=m_mask[None, :]
        )
        lo = tl.where(ok, est - bound, float("-inf"))
        best_lo = tl.maximum(best_lo, tl.max(lo, axis=0))
    # A NaN threshold makes every row a candidate.
    best_lo = tl.where(xnorm < MAX_XN, best_lo, float("nan"))
    tl.atomic_max(lo_ptr + offs_m, _float_key(best_lo), mask=m_mask, sem="relaxed")
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr, 1, sem="acq_rel")
    if arrived == nprog - 1:
        # The last CTA: the rank's tau, exchanged with the peers (the largest
        # bounds a row of the vocabulary from below), for the exact kernel.
        key = tl.atomic_xchg(
            lo_ptr + offs_m, tl.full([M_PAD], LO_RESET, tl.int32), mask=m_mask
        )
        key = tl.where(m_mask, key, LO_RESET)
        if GLOBAL_TAU:
            epoch = tl.load(epoch_ptr)
            src = tl.arange(0, WORLD)
            bases = tl.load(recv_ptrs + src).to(tl.pointer_type(tl.int64))
            tslot = 2 * WORLD * ROWS + (epoch % 2) * WORLD * ROWS
            xmask = (src[:, None] < WORLD) & m_mask[None, :]
            # Non-zero payload: the slot is its own flag (as the keys).
            payload = (key.to(tl.int64) & 0xFFFFFFFF) | (1 << 40)
            tl.atomic_xchg(
                bases[:, None] + tslot + RANK * ROWS + offs_m[None, :],
                tl.broadcast_to(payload[None, :], [WORLD, M_PAD]),
                mask=xmask,
                sem="relaxed",
                scope="sys",
            )
            own = tl.load(recv_ptrs + RANK).to(tl.pointer_type(tl.int64))
            tptrs = own + tslot + src[:, None] * ROWS + offs_m[None, :]
            taus = tl.load(tptrs, mask=xmask, other=1, volatile=True)
            spins = tl.full([], 0, tl.int32)
            while (tl.min(taus) == 0) & (spins < SPIN):
                taus = tl.load(tptrs, mask=xmask, other=1, volatile=True)
                spins += 1
            if spins >= SPIN:
                tl.atomic_add(err_ptr, 1)
            tl.store(tptrs, tl.zeros([WORLD, M_PAD], dtype=tl.int64), mask=xmask)
            k32 = (taus & 0xFFFFFFFF) ^ 0x80000000
            gk = tl.max(tl.where(xmask & (taus != 0), k32, 0), axis=0) - 0x80000000
            # A missing peer (timeout) leaves the rank's own tau.
            key = tl.maximum(key, gk.to(tl.int32))
        tl.store(gtau_ptr + offs_m, key, mask=m_mask)
        tl.store(cnt_ptr, 0)


@triton.jit
def _exact_kernel(
    x_ptr,
    w_ptr,
    up_ptr,
    upt_ptr,
    gtau_ptr,
    best_ptr,
    cnt_ptr,
    epoch_ptr,
    err_ptr,
    recv_ptrs,
    out_ptr,
    M,
    n_valid,
    vocab_start,
    stride_xm,
    stride_wn,
    up_stride,
    upt_stride,
    K: tl.constexpr,
    TILE: tl.constexpr,
    M_PAD: tl.constexpr,
    COLS: tl.constexpr,
    OWN: tl.constexpr,
    E_BLOCK_K: tl.constexpr,
    E_STAGES: tl.constexpr,
    PREFETCH: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EXCHANGE: tl.constexpr,
    ROWS: tl.constexpr,
    SPIN: tl.constexpr,
    REL: tl.constexpr,
    LO_RESET: tl.constexpr,
    PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = tl.cdiv(n_valid, TILE)
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    src = tl.arange(0, WORLD)
    bases = tl.load(recv_ptrs + src).to(tl.pointer_type(tl.int64))
    xmask = (src[:, None] < WORLD) & m_mask[None, :]
    if PDL:
        # The screen grid has finished (its bounds and tau are visible).
        tl.extra.cuda.gdc_wait()
    epoch = tl.load(epoch_ptr)
    slot = (epoch % 2) * WORLD * ROWS
    key = tl.load(gtau_ptr + offs_m, mask=m_mask, other=LO_RESET)
    tau = _key_float(key)
    expo = (tau.to(tl.int32, bitcast=True) >> 23) & 0xFF
    two_ulps = tl.where(
        expo > 8, ((expo - 6) << 23).to(tl.float32, bitcast=True), 1e-37
    )
    cut = tau - two_ulps - tl.abs(tau) * REL

    # Owned fused tiles (pid + i * nprog) that hold a candidate: some row's
    # upper bound reaches the cut (NaN bounds or cuts compare false).
    t = pid + tl.arange(0, OWN) * nprog
    tok = (t < num_tiles)[:, None] & m_mask[None, :]
    u = tl.load(upt_ptr + offs_m[None, :] * upt_stride + t[:, None], mask=tok, other=0.0)
    hit_t = tl.max(tl.where(tok & ~(u < cut[None, :]), 1, 0), axis=1)
    pos = tl.cumsum(hit_t, axis=0) - 1
    count = tl.sum(hit_t, axis=0)
    offs_t = tl.arange(0, TILE)
    best0 = tl.zeros([COLS], dtype=tl.int64)
    best1 = tl.zeros([COLS], dtype=tl.int64)
    for j in range(0, count):
        tile = tl.sum(tl.where((hit_t > 0) & (pos == j), t, 0), axis=0)
        rows = tile * TILE + offs_t
        rv = rows < n_valid
        ur = tl.load(
            up_ptr + offs_m[None, :] * up_stride + rows[:, None],
            mask=rv[:, None] & m_mask[None, :],
            other=float("-inf"),
        )
        hit = (tl.max(tl.where(m_mask[None, :] & ~(ur < cut[None, :]), 1, 0), axis=1) > 0) & rv
        if PREFETCH:
            _prefetch_rows(w_ptr, rows, hit, stride_wn, K * 2)
        k0, k1 = _exact_keys(
            x_ptr, w_ptr, rows, hit, M, vocab_start, stride_xm, stride_wn, K, TILE,
            COLS, M_PAD // COLS, E_BLOCK_K, E_STAGES,
        )
        best0 = tl.maximum(best0, k0)
        best1 = tl.maximum(best1, k1)

    # Fold and exchange the keys, as in the fused kernel.
    offs_c = tl.arange(0, COLS)
    old = tl.atomic_max(best_ptr + offs_c, best0, mask=offs_c < M, sem="relaxed")
    dep = tl.max(tl.where(offs_c < M, old, 0), axis=0)
    if M_PAD > COLS:
        old1 = tl.atomic_max(
            best_ptr + COLS + offs_c, best1, mask=COLS + offs_c < M, sem="relaxed"
        )
        dep = tl.maximum(dep, tl.max(tl.where(COLS + offs_c < M, old1, 0), axis=0))
    # The returned old values feed the arrival increment, so every fold has
    # been performed before this CTA is counted (no key is above 2**48).
    arrive = tl.where(dep < 0, 2, 1)
    done = tl.atomic_add(cnt_ptr + 1, arrive, sem="acq_rel")
    if done == nprog - 1:
        local = tl.atomic_xchg(
            best_ptr + offs_m, tl.zeros([M_PAD], dtype=tl.int64), mask=m_mask
        )
        # 1 (below every key) when no row of this rank was a candidate.
        local = tl.maximum(local, 1)
        tl.store(cnt_ptr + 1, 0)
        tl.store(epoch_ptr, epoch + 1)
        if EXCHANGE:
            # Push this rank's keys to every peer at once. The payload is the
            # flag, so no fence is needed: a slot set is rewritten only after
            # its reader's kernel that reset it has completed.
            tl.atomic_xchg(
                bases[:, None] + slot + RANK * ROWS + offs_m[None, :],
                tl.broadcast_to(local[None, :], [WORLD, M_PAD]),
                mask=xmask,
                sem="relaxed",
                scope="sys",
            )
            own = tl.load(recv_ptrs + RANK).to(tl.pointer_type(tl.int64))
            ptrs = own + slot + src[:, None] * ROWS + offs_m[None, :]
            keys = tl.load(ptrs, mask=xmask, other=1, volatile=True)
            spins = tl.full([], 0, tl.int32)
            while (tl.min(keys) == 0) & (spins < SPIN):
                keys = tl.load(ptrs, mask=xmask, other=1, volatile=True)
                spins += 1
            if spins >= SPIN:
                tl.atomic_add(err_ptr, 1)
            local = tl.max(tl.where(xmask, keys, 0), axis=0)
            tl.store(ptrs, tl.zeros([WORLD, M_PAD], dtype=tl.int64), mask=xmask)
            tl.store(out_ptr + offs_m, _key_token_ids(local), mask=m_mask)
        elif WORLD == 1:
            tl.store(out_ptr + offs_m, _key_token_ids(local), mask=m_mask)
        else:
            # Keys for the NCCL fallback exchange.
            tl.store(out_ptr + offs_m, local, mask=m_mask)


class _Comm:
    """Per-process fold scratch and symmetric receive slots (keys and taus) for
    ``MAX_M`` rows (the fused kernel's own slots hold 16)."""

    def __init__(self, device: torch.device) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.best = torch.zeros(MAX_M, dtype=torch.int64, device=device)
        self.lo = torch.full((MAX_M,), NEG_INF_KEY, dtype=torch.int32, device=device)
        # The call's (exchanged) tau keys, from the screen to the exact kernel.
        self.gtau = torch.zeros(MAX_M, dtype=torch.int32, device=device)
        self.count = torch.zeros(2, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self.peer_ptrs = torch.zeros(1, dtype=torch.int64, device=device)
        self.exchange = False
        if self.world == 1:
            return
        error = None
        try:
            import torch.distributed._symmetric_memory as symm_mem

            # [keys | taus, slot set, source rank, row of x]
            recv = symm_mem.empty(
                (2, vpa.NUM_SLOT_SETS, self.world, MAX_M),
                dtype=torch.int64,
                device=device,
            )
            recv.zero_()
            handle = symm_mem.rendezvous(recv, group.device_group.group_name)
            assert handle is not None
            bases = [int(p) for p in handle.buffer_ptrs]
            assert bases[self.rank] == recv.data_ptr()
            self.peer_ptrs = torch.tensor(bases, dtype=torch.int64, device=device)
            torch.cuda.synchronize(device)
            handle.barrier()
            self.recv, self.handle = recv, handle
        except Exception as e:  # noqa: BLE001 - every rank falls back together
            error = e
        ok = torch.tensor([int(error is None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        self.exchange = bool(ok.item())
        if not self.exchange:
            logger.warning_once(
                "int8-screened draft-head argmax exchanges keys with NCCL: "
                "symmetric memory set-up failed (%r)",
                error,
            )


_COMM: _Comm | None = None


def _weight_key(weight: torch.Tensor) -> tuple:
    """Identity of the weight values the screen was built from."""
    try:
        version = weight._version
    except RuntimeError:  # inference tensors have no version counter
        version = -1
    return (weight.data_ptr(), tuple(weight.shape), version)


class Screen:
    """int8 copy of one lm_head shard, its bound coefficients and scratch."""

    def __init__(self, weight: torch.Tensor, n_valid: int) -> None:
        rows, k = weight.shape
        dev = weight.device
        self.key = _weight_key(weight)
        self.n_valid = n_valid
        self.q = torch.zeros((rows, k), dtype=torch.int8, device=dev)
        self.scale = torch.ones(rows, dtype=torch.float32, device=dev)
        self.coef = torch.zeros(rows, dtype=torch.float32, device=dev)
        inf = torch.tensor(float("inf"), dtype=torch.float32, device=dev)
        for r0 in range(0, n_valid, _PREP_ROWS):
            r1 = min(n_valid, r0 + _PREP_ROWS)
            w = weight[r0:r1].double()
            s = (w.abs().amax(dim=1) / 127.0).float()
            s = torch.where(s > 0, s, torch.ones_like(s))
            q = torch.round(w / s.double()[:, None]).clamp_(-127.0, 127.0)
            deq = q * s.double()[:, None]
            c = (w - deq).norm(dim=1) + GAMMA * (w.norm(dim=1) + deq.norm(dim=1))
            self.q[r0:r1] = q.to(torch.int8)
            self.scale[r0:r1] = s
            # Round the fp64 coefficient up to fp32.
            self.coef[r0:r1] = torch.nextafter((c * (1.0 + 2.0**-30)).float(), inf)
            del w, q, deq, c
        self.ok = bool(
            torch.isfinite(self.coef[:n_valid]).all()
            and torch.isfinite(self.scale[:n_valid]).all()
        )
        # Upper bound per (row of x, shard row) and per fused tile (64 rows;
        # room for screen tiles of up to 512 rows past the last one).
        self.up = torch.empty((MAX_M, rows), dtype=torch.float32, device=dev)
        n_up = triton.cdiv(rows, TILE_N) + 8
        self.upt = torch.empty((MAX_M, n_up), dtype=torch.float32, device=dev)
        self._q_descs: dict = {}

    def q_desc(self, block_n: int, block_k: int):
        from triton.tools.tensor_descriptor import TensorDescriptor

        desc = self._q_descs.get((block_n, block_k))
        if desc is None:
            desc = TensorDescriptor.from_tensor(self.q, [block_n, block_k])
            self._q_descs[(block_n, block_k)] = desc
        return desc


def _get_comm(device: torch.device) -> _Comm | None:
    global _COMM
    if _COMM is None:
        _COMM = _Comm(device)
    return _COMM


def get_screen(lm_head: VocabParallelEmbedding) -> Screen | None:
    """The lm_head's screen, built on its first eager call (None if it cannot)."""
    weight = lm_head.weight
    screen = getattr(lm_head, "_argmax_screen", None)
    if screen is not None and screen.key == _weight_key(weight):
        return screen
    if getattr(lm_head, "_argmax_screen_failed", False):
        return None
    if torch.cuda.is_current_stream_capturing():
        # Allocation and set-up are not capturable; this graph keeps the fused
        # kernel (same tokens) or, above 16 rows, the stock path.
        return None
    lm_head._argmax_screen = None
    k = weight.shape[1]
    if (
        not ENABLED
        or k % _SCREEN_CONFIG[1]
        or any(k % c[0] for c in _EXACT_CONFIGS.values())
        or _SCREEN_CONFIG[0] % TILE_N
    ):
        lm_head._argmax_screen_failed = True
        return None
    try:
        _get_comm(weight.device)
        screen = Screen(weight, lm_head.shard_indices.num_org_elements)
    except Exception as e:  # noqa: BLE001 - e.g. out of memory
        logger.warning("int8-screened draft-head argmax disabled: %r", e)
        lm_head._argmax_screen_failed = True
        return None
    if not screen.ok:
        logger.warning("int8-screened draft-head argmax disabled: non-finite weights")
        lm_head._argmax_screen_failed = True
        return None
    lm_head._argmax_screen = screen
    logger.info_once(
        "int8-screened draft-head argmax enabled (%d x %d shard, %.0f MB copy, "
        "up to %d rows)",
        weight.shape[0],
        k,
        screen.q.numel() / 1e6,
        MAX_M,
    )
    return screen


def _x_tma_ok(hidden_states: torch.Tensor) -> bool:
    return (
        hidden_states.data_ptr() % 16 == 0
        and hidden_states.stride(0) * hidden_states.element_size() % 16 == 0
    )


def launch(
    screen: Screen,
    comm: _Comm,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    vocab_start: int,
    out: torch.Tensor,
    config: tuple[int, int, int, int] | None = None,
    ctas_per_sm: int | None = None,
    exact_config: tuple[int, int] | None = None,
    use_tma: bool | None = None,
    exchange: bool | None = None,
    global_tau: bool | None = None,
    exact: bool = True,
    x_tma: bool | None = None,
    pdl: bool | None = None,
    prefetch: bool | None = None,
) -> None:
    """Screened argmax (screen kernel, then the exact kernel); ``out`` gets
    token ids (int64, ``[M]``), or the rank keys when ``exchange`` is False on
    a TP group. ``exact=False`` (diagnostics) skips the exact kernel."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    block_n, block_k, num_warps, num_stages = config or _SCREEN_CONFIG
    m_pad_ = EXACT_COLS if hidden_states.shape[0] <= EXACT_COLS else MAX_M
    e_block_k, e_stages = exact_config or _EXACT_CONFIGS[m_pad_]
    ctas_per_sm = _SCREEN_CTAS_PER_SM if ctas_per_sm is None else ctas_per_sm
    use_tma = _SCREEN_TMA if use_tma is None else use_tma
    exchange = comm.exchange if exchange is None else exchange
    exchange = exchange and comm.world > 1
    global_tau = (GLOBAL_TAU if global_tau is None else global_tau) and exchange
    pdl = USE_PDL if pdl is None else pdl
    prefetch = PREFETCH if prefetch is None else prefetch
    m, k = hidden_states.shape
    assert 0 < m <= MAX_M and block_n % TILE_N == 0
    m_pad = EXACT_COLS if m <= EXACT_COLS else MAX_M
    if x_tma is None:
        x_tma = _x_tma_ok(hidden_states)
    x_tma = use_tma and x_tma
    n_valid = screen.n_valid
    grid = min(triton.cdiv(n_valid, block_n), ctas_per_sm * comm.num_sms)
    _screen_kernel[(grid,)](
        hidden_states,
        TensorDescriptor.from_tensor(hidden_states, [m_pad, block_k])
        if x_tma
        else None,
        screen.q,
        screen.q_desc(block_n, block_k) if use_tma else None,
        screen.scale,
        screen.coef,
        screen.up,
        screen.upt,
        comm.lo,
        comm.gtau,
        comm.count,
        comm.epoch,
        comm.errors,
        comm.peer_ptrs,
        m,
        n_valid,
        hidden_states.stride(0),
        screen.q.stride(0),
        screen.up.stride(0),
        screen.upt.stride(0),
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        TILE=TILE_N,
        M_PAD=m_pad,
        NUM_STAGES=num_stages,
        USE_TMA=use_tma,
        X_TMA=x_tma,
        RANK=comm.rank,
        WORLD=comm.world,
        GLOBAL_TAU=global_tau,
        ROWS=MAX_M,
        SPIN=vpa.SPIN_LIMIT,
        REL=REL_SLACK,
        XN_SCALE=XN_SLACK,
        MAX_XN=MAX_XNORM,
        LO_RESET=NEG_INF_KEY,
        PDL=pdl,
        num_warps=num_warps,
    )
    if not exact:
        return
    tiles = triton.cdiv(n_valid, TILE_N)
    egrid = min(tiles, _EXACT_CTAS_PER_SM * comm.num_sms)
    _exact_kernel[(egrid,)](
        hidden_states,
        weight,
        screen.up,
        screen.upt,
        comm.gtau,
        comm.best,
        comm.count,
        comm.epoch,
        comm.errors,
        comm.peer_ptrs,
        out,
        m,
        n_valid,
        vocab_start,
        hidden_states.stride(0),
        weight.stride(0),
        screen.up.stride(0),
        screen.upt.stride(0),
        K=k,
        TILE=TILE_N,
        M_PAD=m_pad,
        COLS=EXACT_COLS,
        OWN=triton.next_power_of_2(triton.cdiv(tiles, egrid)),
        E_BLOCK_K=e_block_k,
        E_STAGES=e_stages,
        PREFETCH=prefetch,
        RANK=comm.rank,
        WORLD=comm.world,
        EXCHANGE=exchange,
        ROWS=MAX_M,
        SPIN=vpa.SPIN_LIMIT,
        REL=REL_SLACK,
        LO_RESET=NEG_INF_KEY,
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )


def candidate_rows(screen: Screen, comm: _Comm, m: int) -> torch.Tensor:
    """Diagnostics: the final candidate mask over shard rows of the last call
    (its bounds and the tau it pruned with)."""
    n = screen.n_valid
    key = comm.gtau[:m].clone()
    bits = torch.where(key >= 0, key, key ^ 0x7FFFFFFF)
    tau = bits.view(torch.float32)
    expo = (bits >> 23) & 0xFF
    two_ulps = torch.where(
        expo > 8, ((expo - 6) << 23).view(torch.float32), torch.full_like(tau, 1e-37)
    )
    cut = tau - two_ulps - tau.abs() * REL_SLACK
    up = screen.up[:m, :n]
    return (~(up < cut[:, None])).any(dim=0)


def try_launch(
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Screened argmax into ``out`` (token ids, or rank keys without symmetric
    memory); False to use the fused kernel or the stock path instead."""
    m = hidden_states.shape[0]
    if (
        m == 0
        or m > MAX_M
        or lm_head.shard_indices.num_org_elements < MIN_SHARD_ROWS
    ):
        return False
    screen = get_screen(lm_head)
    comm = _COMM
    if screen is None or comm is None:
        return False
    weight = lm_head.weight
    start = lm_head.shard_indices.org_vocab_start_index
    if not (_SCREEN_TMA and weight.data_ptr() % 16 == 0):
        return False
    x_tma = _x_tma_ok(hidden_states)
    if not getattr(screen, "launched", False):
        # The first (eager) call compiles both variants (up to 16 and up to
        # 32 rows of x) on rows of this call. A compile failure happens on
        # every rank before any key is pushed, so all ranks fall back
        # together.
        try:
            scratch = torch.empty(MAX_M, dtype=torch.int64, device=out.device)
            for rows in (EXACT_COLS, EXACT_COLS + 1):
                idx = torch.arange(rows, device=out.device) % m
                xs = hidden_states.index_select(0, idx)
                launch(screen, comm, xs, weight, start, scratch[:rows], x_tma=x_tma)
        except Exception as e:  # noqa: BLE001
            logger.warning("int8-screened draft-head argmax disabled: %r", e)
            lm_head._argmax_screen = None
            lm_head._argmax_screen_failed = True
            return False
        screen.launched = True
    launch(screen, comm, hidden_states, weight, start, out, x_tma=x_tma)
    return True
