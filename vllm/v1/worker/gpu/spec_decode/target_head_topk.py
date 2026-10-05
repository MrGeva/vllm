# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sort-free verify sampling from the target head's exact top-k candidates.

With every request on the sort-free top-k path (top_k <= 64), the verify
sampler only ever reads each row's logits at or above its k-th largest
(``topk_rejection_sampler``). The stock target head still computes all of
them: the exact push GEMV streams the bf16 vocab shard (1.23 GB per rank at
TP2) and gathers ``[M, vocab]`` logits on every rank. Opt-in with
VLLM_MM3_TARGET_HEAD_TOPK=1, read via os.environ.get, this path computes
only rows that can matter, bit for bit:

1. Screen (``_screen_kernel``). The int8 copy of the shard
   (``vocab_parallel_screen.Screen``, shared with the draft head's screen)
   gives ``est`` and the rigorous bound ``|a - est| <= c * ||x||`` per row of
   the shard and row of x, where ``a`` is the fp32 value the push GEMV rounds
   to bf16. Upper bounds go to scratch, and so does the largest lower bound
   of every 64-row tile. The last M CTAs (one per row of x, after every CTA
   arrived) find the row's k-th largest tile lower bound (bf16-rounded down),
   exchange their KSLOTS largest with the peers through symmetric memory (a
   bounded wait; on timeout the rank keeps its own), and take the k-th
   largest of the union as ``tau``: k distinct rows of the vocabulary reach
   it, so the k-th largest logit does. A row whose upper bound is below
   ``tau`` minus two bf16 ulps rounds strictly below k logits: it can be
   neither in the top k nor tied with its k-th. Non-finite or huge x makes
   every row a candidate.
2. Compaction (``_compact_kernel``): shard rows that are a candidate for any
   row of x, appended with an atomic counter (any order).
3. Exact pass (``_exact_kernel``). Gathered tiles of candidate rows recompute
   their logits with the push GEMV's arithmetic: the calibrated split of K,
   each split's 16-wide tensor-core MMAs in K order with fp32 accumulation,
   the partials added in split order from 0, RTNE bf16 rounding; so each
   logit equals the stock one bit for bit. Each split of a tile is its own
   work item (a short MMA chain per CTA) and the tile's last split to finish
   adds the partials. Tiles of fewer than 64 rows run as mma.sync, whose
   per-element results equal the GEMV's tcgen05 64 x 16 dot. Per row of x,
   every logit at or above its cut goes to every rank as a flagged (token,
   logit) word, each word its own ready flag (no fence); the last CTA
   publishes the flagged list lengths.
4. Sampler (``_sample_kernel``), one CTA per row of x once every rank's list
   arrived (bounded waits). The lists hold every logit at or above the k-th
   largest, so the exact k-th largest key, the logits above it and its ties
   are the stock ones. They fill the stock's 64 survivor slots in token order
   and the rest is the stock select kernel's code verbatim (temperature,
   top-p, the target log-prob of the next draft token, Gumbel residual and
   bonus samples with the stock per-token noise). When the lists exceed
   FAST_CAP entries or the ties do not fit the slots, the CTA writes the lists
   into a -inf row and runs the stock tile-max and select code on it: the
   stock top-k/top-p and its tie order only ever read tiles holding a logit
   at or above the k-th largest, so that gives the stock result too.

The stock rejection kernel then accepts and resamples as before: sampled and
accepted tokens equal the stock path's under the same seeds.

Slots alternate between two sets by a device call counter. A rank writes a
peer's set again only after that peer consumed and cleared it: the peer's
next call's lengths, published after its sampler cleared the set, precede the
writer's next-but-one call.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import vocab_parallel_logits as vpl
from vllm.model_executor.layers import vocab_parallel_screen as vps
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable, murmur3_uniform32
from vllm.v1.worker.gpu.spec_decode.topk_rejection_sampler import (
    _CAND_TILES,
    _SLOTS,
    _TILE,
    MAX_TOP_K,
    _kth_largest_key,
    _ordered_key,
    _topk_rejection_kernel,
)

logger = init_logger(__name__)

ENV_KNOB = "VLLM_MM3_TARGET_HEAD_TOPK"
# Rows of x (logits per step) on this path: one 16-column exact dot.
ROWS = 16
# Lower-bound keys per row each rank sends (>= MAX_TOP_K).
KSLOTS = 64
# List entries per row (both ranks) the sampler handles in registers; longer
# lists (or ties that overflow the slots) take the dense fallback.
FAST_CAP = 1024
# Shard rows per CTA of the compaction.
_COMPACT_ROWS = 512
# (BLOCK_N, BLOCK_K, num_warps, num_stages) and CTAs per SM of the screen.
_SCREEN_CONFIG = (128, 256, 4, 3)
_SCREEN_CTAS_PER_SM = 2
# CTAs per SM of the exact pass and its pipeline depth.
_EXACT_CTAS_PER_SM = 3
_EXACT_STAGES = 4
# Gathered rows per exact tile (below 64 rows the dot runs as mma.sync, whose
# per-element arithmetic equals the GEMV's tcgen05 64 x 16 dot).
_EXACT_TILE = 32
# Rows of the -inf fill / list scatter in the dense fallback.
_DENSE_BLOCK = 4096
_TILE_MAX_PER_ITER = 32
# Programmatic dependent launch of the compaction (after the screen) and of
# the sampler (after the exact pass).
USE_PDL = True
# Calls between the screened path's counter logs.
_LOG_EVERY = 4096
# Diagnostics: the sampler keeps a copy of the lists it read (dbg_tok/dbg_lg).
DEBUG = False

_INT32_MIN = tl.constexpr(-(2**31))
_INT32_MAX = tl.constexpr(2**31 - 1)


def enabled() -> bool:
    return os.environ.get(ENV_KNOB, "0") == "1"


@triton.jit
def _float_key(f):
    bits = f.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _key_float(key):
    bits = tl.where(key >= 0, key, key ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _screen_kernel(
    x_ptr,
    q_desc,
    scale_ptr,
    coef_ptr,
    up_ptr,
    lot_ptr,
    keys_ptr,
    cut_ptr,
    sync_ptr,
    tau_ptrs,
    epoch_ptr,
    err_ptr,
    expanded_idx_mapping_ptr,
    top_k_ptr,
    M,
    n_valid,
    stride_xm,
    up_stride,
    lot_stride,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE: tl.constexpr,
    M_PAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    TILES_PAD: tl.constexpr,
    KSLOTS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    ROWS: tl.constexpr,
    SPIN: tl.constexpr,
    REL: tl.constexpr,
    XN_SCALE: tl.constexpr,
    MAX_XN: tl.constexpr,
    KEY16_MIN: tl.constexpr,
    PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    num_blocks = tl.cdiv(n_valid, BLOCK_N)
    num_tiles = tl.cdiv(n_valid, TILE)
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    HALVES: tl.constexpr = BLOCK_N // TILE

    sq = tl.zeros([BLOCK_K, M_PAD], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        xf = tl.load(x_ptrs + k, mask=m_mask[None, :], other=0.0).to(tl.float32)
        sq += xf * xf
    xnorm = tl.sqrt(tl.sum(sq, axis=0)) * XN_SCALE

    for blk in range(pid, num_blocks, nprog):
        rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = rows < n_valid
        acc = tl.zeros([BLOCK_N, M_PAD], dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            q = q_desc.load([blk * BLOCK_N, k])
            xt = tl.load(x_ptrs + k, mask=m_mask[None, :], other=0.0)
            acc = tl.dot(q.to(xt.dtype), xt, acc)
        s = tl.load(scale_ptr + rows, mask=row_ok, other=0.0)
        c = tl.load(coef_ptr + rows, mask=row_ok, other=0.0)
        est = acc * s[:, None]
        bound = c[:, None] * xnorm[None, :]
        bound += (tl.abs(est) + bound) * REL
        ok = row_ok[:, None] & m_mask[None, :]
        up = tl.where(ok, est + bound, float("-inf"))
        lo = tl.where(ok, est - bound, float("-inf"))
        tl.store(up_ptr + offs_m[None, :] * up_stride + rows[:, None], up, mask=ok)
        lot = tl.max(tl.reshape(lo, [HALVES, TILE, M_PAD]), axis=1)
        t = blk * HALVES + tl.arange(0, HALVES)
        tl.store(
            lot_ptr + offs_m[None, :] * lot_stride + t[:, None],
            lot,
            mask=(t < num_tiles)[:, None] & m_mask[None, :],
        )

    # The last M CTAs to arrive find one row of x's tau each, once every CTA
    # has stored its bounds.
    tl.debug_barrier()
    arrived = tl.atomic_add(sync_ptr, 1, sem="acq_rel")
    first = nprog - M
    if arrived >= first:
        j = arrived - first
        got = tl.atomic_add(sync_ptr, 0, sem="acquire")
        spins = tl.full([], 0, tl.int32)
        while (got < nprog) & (spins < SPIN):
            got = tl.atomic_add(sync_ptr, 0, sem="acquire")
            spins += 1
        all_in = got >= nprog
        epoch = tl.load(epoch_ptr)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + j).to(tl.int64)
        k = tl.load(top_k_ptr + req_state_idx)
        tiles = tl.arange(0, TILES_PAD)
        tv = tiles < num_tiles
        # 16-bit keys (bf16 rounded down): a 16-step search, still a lower
        # bound of the k-th largest logit.
        lkey = _float_key(
            tl.load(lot_ptr + j * lot_stride + tiles, mask=tv, other=float("-inf"))
        ) >> 16
        lkey = tl.where(tv, lkey, KEY16_MIN)
        hi = tl.max(lkey)
        lo_key = tl.min(tl.where(tv, lkey, _INT32_MAX))
        kth = _kth_largest_key(lkey, k, lo_key, hi)
        # The rank's largest KSLOTS tile lower bounds at or above its k-th.
        is_top = tv & (lkey >= kth)
        rank = tl.cumsum(is_top.to(tl.int32), axis=0)
        sel = is_top & (rank <= KSLOTS)
        tl.store(keys_ptr + j * KSLOTS + rank - 1, lkey, mask=sel)
        nsel = tl.minimum(tl.sum(is_top.to(tl.int32)), KSLOTS)
        tl.debug_barrier()
        slots = tl.arange(0, KSLOTS)
        mine = tl.load(keys_ptr + j * KSLOTS + slots, mask=slots < nsel, other=KEY16_MIN)
        tau_key = kth
        if WORLD > 1:
            src = tl.arange(0, WORLD)
            bases = tl.load(tau_ptrs + src).to(tl.pointer_type(tl.int64))
            set_off = (epoch % 2) * WORLD * ROWS * KSLOTS
            # Non-zero payload: every slot is its own flag.
            payload = (mine.to(tl.int64) & 0xFFFFFFFF) | (1 << 40)
            tl.atomic_xchg(
                bases[:, None] + set_off + (RANK * ROWS + j) * KSLOTS + slots[None, :],
                tl.broadcast_to(payload[None, :], [WORLD, KSLOTS]),
                sem="relaxed",
                scope="sys",
            )
            own = tl.load(tau_ptrs + RANK).to(tl.pointer_type(tl.int64))
            rptrs = own + set_off + (src[:, None] * ROWS + j) * KSLOTS + slots[None, :]
            recv = tl.load(rptrs, volatile=True)
            spins = tl.full([], 0, tl.int32)
            while (tl.min(recv) == 0) & (spins < SPIN):
                recv = tl.load(rptrs, volatile=True)
                spins += 1
            tl.store(rptrs, tl.zeros([WORLD, KSLOTS], dtype=tl.int64))
            if spins >= SPIN:
                tl.atomic_add(err_ptr, 1)
            else:
                k32 = ((recv & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
                ukeys = tl.reshape(k32.to(tl.int32), [WORLD * KSLOTS])
                tau_key = _kth_largest_key(ukeys, k, kth, tl.max(ukeys))
        if not all_in:
            tl.atomic_add(err_ptr, 1)
        tau = _key_float(tau_key << 16)
        expo = (tau.to(tl.int32, bitcast=True) >> 23) & 0xFF
        two_ulps = tl.where(
            expo > 8, ((expo - 6) << 23).to(tl.float32, bitcast=True), 1e-37
        )
        cut = tau - two_ulps - tl.abs(tau) * REL
        xn = tl.sum(tl.where(offs_m == j, xnorm, 0.0))
        # Non-finite or huge x (or a missing arrival): every row is a candidate.
        cut = tl.where((xn < MAX_XN) & all_in, cut, float("-inf"))
        tl.store(cut_ptr + j, cut)
        done = tl.atomic_add(sync_ptr + 1, 1, sem="acq_rel")
        if done == M - 1:
            tl.store(sync_ptr, 0)
            tl.store(sync_ptr + 1, 0)


@triton.jit
def _compact_kernel(
    up_ptr,
    cut_ptr,
    lst_ptr,
    len_ptr,
    M,
    n_valid,
    up_stride,
    CH: tl.constexpr,
    M_PAD: tl.constexpr,
    PDL: tl.constexpr,
):
    """Shard rows that are a candidate for any row of x, appended (any order)."""
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs_m = tl.arange(0, M_PAD)
    mm = offs_m < M
    cut = tl.load(cut_ptr + offs_m, mask=mm, other=float("inf"))
    v = pid * CH + tl.arange(0, CH)
    vv = v < n_valid
    ok = mm[:, None] & vv[None, :]
    up = tl.load(
        up_ptr + offs_m[:, None] * up_stride + v[None, :], mask=ok, other=float("-inf")
    )
    # NaN bounds compare false: candidates.
    cand = tl.max((ok & ~(up < cut[:, None])).to(tl.int32), axis=0)
    base = tl.atomic_add(len_ptr, tl.sum(cand, axis=0), sem="relaxed")
    pos = base + tl.cumsum(cand, axis=0) - 1
    tl.store(lst_ptr + pos, v, mask=cand > 0)


@triton.jit
def _split_partial(
    x_ptrs,
    w_ptrs,
    ev,
    mm,
    s,
    K: tl.constexpr,
    NSPLIT: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE: tl.constexpr,
    M_PAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Partial s of the push GEMV's split of K for the gathered rows: 16-wide
    tensor-core MMAs along the split's K steps in order, fp32 accumulation."""
    STEPS: tl.constexpr = K // (BLOCK_K * NSPLIT)
    acc = tl.zeros([TILE, M_PAD], dtype=tl.float32)
    for jj in tl.range(0, STEPS, num_stages=NUM_STAGES):
        if INTERLEAVE:
            k = (s + jj * NSPLIT) * BLOCK_K
        else:
            k = s * (K // NSPLIT) + jj * BLOCK_K
        w = tl.load(w_ptrs + k, mask=ev[:, None], other=0.0)
        xt = tl.load(x_ptrs + k, mask=mm[None, :], other=0.0)
        acc = tl.dot(w, xt, acc)
    return acc


@triton.jit
def _exact_kernel(
    x_ptr,
    w_ptr,
    lst_ptr,
    len_ptr,
    cut_ptr,
    olen_ptr,
    arrive_ptr,
    part_ptr,
    tile_cnt_ptr,
    lst_ptrs,
    hdr_ptrs,
    epoch_ptr,
    stats_ptr,
    M,
    vocab_start,
    stride_xm,
    stride_wn,
    K: tl.constexpr,
    NSPLIT: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE: tl.constexpr,
    M_PAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    ROWS: tl.constexpr,
    NCAP: tl.constexpr,
):
    """Candidate logits with the push GEMV's arithmetic: per gathered tile of
    TILE rows, each split of K is its own work item (a CTA's 16-wide MMA chain
    along that split's K steps; the per-element arithmetic of these small
    tiles equals the GEMV's 64 x 16 dot bit for bit), and the last split to
    finish adds the partials in split order from 0 and rounds to bf16 (RTNE).
    Per row of x, every logit at or above its cut goes as a flagged (token,
    logit) word to every rank's list; the last CTA publishes the flagged list
    lengths on every rank."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.load(epoch_ptr)
    sset = epoch % 2
    list_off = sset.to(tl.int64) * WORLD * ROWS * NCAP + RANK * ROWS * NCAP
    offs_m = tl.arange(0, M_PAD)
    mm = offs_m < M
    cut = tl.load(cut_ptr + offs_m, mask=mm, other=float("inf"))
    E = tl.load(len_ptr)
    ntiles = tl.cdiv(E, TILE)
    i = tl.arange(0, TILE)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    part_off = i[:, None] * M_PAD + offs_m[None, :]
    for item in range(pid, ntiles * NSPLIT, nprog):
        t = item // NSPLIT
        s = item - t * NSPLIT
        e = t * TILE + i
        ev = e < E
        v = tl.load(lst_ptr + e, mask=ev, other=0)
        w_ptrs = w_ptr + v[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
        acc = _split_partial(
            x_ptrs, w_ptrs, ev, mm, s, K, NSPLIT, INTERLEAVE, BLOCK_K, TILE, M_PAD,
            NUM_STAGES,
        )
        last = True
        if NSPLIT > 1:
            tl.store(part_ptr + (t * NSPLIT + s) * TILE * M_PAD + part_off, acc)
            tl.debug_barrier()
            done = tl.atomic_add(tile_cnt_ptr + t, 1, sem="acq_rel")
            last = done == NSPLIT - 1
        if last:
            if NSPLIT == 1:
                total = acc
            else:
                tl.store(tile_cnt_ptr + t, 0)
                # The reduce adds the split partials in order (fp32, from 0).
                total = tl.zeros([TILE, M_PAD], dtype=tl.float32)
                for ss in tl.static_range(NSPLIT):
                    p = tl.load(
                        part_ptr + (t * NSPLIT + ss) * TILE * M_PAD + part_off,
                        cache_modifier=".cg",
                    )
                    total = total + p
            lg = total.to(tl.bfloat16, fp_downcast_rounding="rtne")
            # NaN logits (non-finite x, cut -inf) pass too.
            keep = ev[:, None] & mm[None, :] & ~(lg.to(tl.float32) < cut[None, :])
            ki = keep.to(tl.int32)
            c = tl.sum(ki, axis=0)
            base = tl.atomic_add(olen_ptr + offs_m, c, mask=mm & (c > 0), sem="relaxed")
            pos = base[None, :] + tl.cumsum(ki, axis=0) - 1
            off = list_off + offs_m[None, :].to(tl.int64) * NCAP + pos
            # (token, logit bits) with a flag bit: each entry is its own
            # ready flag, so no fence is needed before the peers read it.
            bits = lg.to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
            word = (1 << 48) | ((v + vocab_start).to(tl.int64)[:, None] << 16) | bits
            for d in tl.static_range(WORLD):
                lb = tl.load(lst_ptrs + d).to(tl.pointer_type(tl.int64))
                tl.store(lb + off, word, mask=keep)
    # The last CTA to arrive publishes the list lengths (flagged) on every
    # rank.
    tl.debug_barrier()
    arrived = tl.atomic_add(arrive_ptr, 1, sem="acq_rel")
    if arrived == nprog - 1:
        L = tl.atomic_xchg(olen_ptr + offs_m, tl.zeros([M_PAD], dtype=tl.int32), mask=mm)
        hword = (1 << 40) | L.to(tl.int64)
        for d in tl.static_range(WORLD):
            hb = tl.load(hdr_ptrs + d).to(tl.pointer_type(tl.int64))
            tl.atomic_xchg(
                hb + sset * WORLD * ROWS + RANK * ROWS + offs_m, hword, mask=mm,
                sem="relaxed", scope="sys",
            )
        tl.store(len_ptr, 0)
        tl.store(arrive_ptr, 0)
        tl.atomic_add(stats_ptr + 3, E.to(tl.int64), sem="relaxed")


@triton.jit
def _select_dense(
    logits_ptr,
    logits_stride,
    tile_max_ptr,
    tile_max_stride,
    cand_tile_ptr,
    slot_token_ptr,
    slot_logit_ptr,
    row,
    k,
    vocab_size,
    num_tiles,
    TILE: tl.constexpr,
    TILES_PAD: tl.constexpr,
    CAND_TILES: tl.constexpr,
    SLOTS: tl.constexpr,
    KEY_SHIFT: tl.constexpr,
    TILES_PER_ITER: tl.constexpr,
):
    """The stock tile-max and select kernels on one dense logits row (the
    survivor compaction of ``_topk_select_sample_kernel``, verbatim)."""
    for t0 in range(0, num_tiles, TILES_PER_ITER):
        tiles_i = t0 + tl.arange(0, TILES_PER_ITER)
        offs_i = tiles_i[:, None] * TILE + tl.arange(0, TILE)[None, :]
        lt = tl.load(
            logits_ptr + row * logits_stride + offs_i,
            mask=offs_i < vocab_size,
            other=float("-inf"),
        ).to(tl.float32)
        tl.store(
            tile_max_ptr + row * tile_max_stride + tiles_i,
            tl.max(lt, axis=1),
            mask=tiles_i < num_tiles,
        )
    tl.debug_barrier()

    tiles = tl.arange(0, TILES_PAD)
    tile_valid = tiles < num_tiles
    tile_key = _ordered_key(
        tl.load(
            tile_max_ptr + row * tile_max_stride + tiles,
            mask=tile_valid,
            other=float("-inf"),
        ),
        KEY_SHIFT,
    )
    max_key = tl.max(tl.where(tile_valid, tile_key, _INT32_MIN))
    min_key = tl.min(tl.where(tile_valid, tile_key, _INT32_MAX))
    tile_key = tl.where(tile_valid, tile_key, _INT32_MIN)
    tau = _kth_largest_key(tile_key, k, min_key, max_key)
    tile_gt = tile_key > tau
    tile_eq = tile_key == tau
    num_tile_gt = tl.sum(tile_gt.to(tl.int32))
    tile_eq_rank = tl.cumsum(tile_eq.to(tl.int32), axis=0)
    is_cand = tile_gt | (tile_eq & (tile_eq_rank <= CAND_TILES - num_tile_gt))
    cand_slot = tl.cumsum(is_cand.to(tl.int32), axis=0) - 1
    num_cand = tl.sum(is_cand.to(tl.int32))
    tl.store(cand_tile_ptr + row * CAND_TILES + cand_slot, tiles, mask=is_cand)
    tl.debug_barrier()

    flat = tl.arange(0, CAND_TILES * TILE)
    cand = flat // TILE
    cand_valid = cand < num_cand
    cand_tile = tl.load(
        cand_tile_ptr + row * CAND_TILES + cand, mask=cand_valid, other=0
    )
    token = cand_tile * TILE + flat % TILE
    valid = cand_valid & (token < vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + token, mask=valid, other=float("-inf")
    ).to(tl.float32)
    key = tl.where(valid, _ordered_key(logits, KEY_SHIFT), _INT32_MIN)

    kth = _kth_largest_key(key, k, tau, max_key)
    is_gt = key > kth
    is_tie = key == kth
    num_gt = tl.sum(is_gt.to(tl.int32))
    num_tie = tl.sum(is_tie.to(tl.int32))
    to_slot = is_gt | is_tie
    if num_gt + num_tie > SLOTS:
        tie_rank = tl.cumsum(is_tie.to(tl.int32), axis=0)
        to_slot = is_gt | (is_tie & (tie_rank <= SLOTS - num_gt))
    slot = tl.cumsum(to_slot.to(tl.int32), axis=0) - 1
    tl.store(slot_token_ptr + row * SLOTS + slot, token, mask=to_slot)
    tl.store(slot_logit_ptr + row * SLOTS + slot, logits, mask=to_slot)
    num_slots = tl.minimum(num_gt + num_tie, SLOTS)
    return kth, num_gt, num_slots


# num_logits changes with the batch; specializing on it would compile new
# variants mid-serving.
@triton.jit(do_not_specialize=["num_logits"])
def _sample_kernel(
    sample_ptr,
    next_logprob_ptr,
    lst_ptr,
    hdr_ptr,
    epoch_ptr,
    sync_ptr,
    err_ptr,
    tmp_token_ptr,
    tmp_logit_ptr,
    slot_token_ptr,
    slot_logit_ptr,
    dense_ptr,
    dense_stride,
    tile_max_ptr,
    tile_max_stride,
    cand_tile_ptr,
    path_ptr,
    stats_ptr,
    dbg_tok_ptr,
    dbg_lg_ptr,
    draft_sampled_ptr,
    pos_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    top_k_ptr,
    top_p_ptr,
    seed_ptr,
    num_logits,
    vocab_size,
    num_tiles,
    WORLD: tl.constexpr,
    ROWS: tl.constexpr,
    NCAP: tl.constexpr,
    FAST_CAP: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PAD: tl.constexpr,
    CAND_TILES: tl.constexpr,
    SLOTS: tl.constexpr,
    KEY_SHIFT: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
    SPIN: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    TILES_PER_ITER: tl.constexpr,
    PDL: tl.constexpr,
    DEBUG: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    epoch = tl.load(epoch_ptr)
    sset = epoch % 2
    # Every rank's list lengths of this call (flagged words), then the lists.
    src = tl.arange(0, WORLD)
    hptrs = hdr_ptr + sset * WORLD * ROWS + src * ROWS + row
    hw = tl.load(hptrs, volatile=True)
    spins = tl.full([], 0, tl.int32)
    while (tl.min(hw) == 0) & (spins < SPIN):
        hw = tl.load(hptrs, volatile=True)
        spins += 1
    if spins >= SPIN:
        tl.atomic_add(err_ptr, 1)
    tl.store(hptrs, tl.zeros([WORLD], dtype=tl.int64))
    lens = (hw & 0xFFFFFFFF).to(tl.int32)
    total_len = tl.sum(lens, axis=0)
    list_base = sset.to(tl.int64) * WORLD * ROWS * NCAP
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    k = tl.load(top_k_ptr + req_state_idx)

    kth = tl.full([], 0, tl.int32)
    num_gt = tl.full([], 0, tl.int32)
    num_slots = tl.full([], 0, tl.int32)
    slow = total_len > FAST_CAP
    if total_len <= FAST_CAP:
        f = tl.arange(0, FAST_CAP)
        ends = tl.cumsum(lens, axis=0)
        starts = ends - lens
        sf = tl.sum((f[:, None] >= ends[None, :]).to(tl.int32), axis=1)
        valid = f < total_len
        idx = f - tl.sum(tl.where(src[None, :] == sf[:, None], starts[None, :], 0), axis=1)
        off = list_base + (sf.to(tl.int64) * ROWS + row) * NCAP + idx
        word = tl.load(lst_ptr + off, mask=valid, other=1, volatile=True)
        spins = tl.full([], 0, tl.int32)
        while (tl.min(word) == 0) & (spins < SPIN):
            word = tl.load(lst_ptr + off, mask=valid, other=1, volatile=True)
            spins += 1
        if spins >= SPIN:
            tl.atomic_add(err_ptr, 1)
        tok = ((word >> 16) & 0xFFFFFFFF).to(tl.int32)
        lg = (word & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        lg = tl.where(valid, lg, float("-inf"))
        if DEBUG:
            tl.store(dbg_tok_ptr + row * FAST_CAP + f, tok, mask=valid)
            tl.store(dbg_lg_ptr + row * FAST_CAP + f, lg, mask=valid)
        key = tl.where(valid, _ordered_key(lg, KEY_SHIFT), _INT32_MIN)
        max_key = tl.max(key)
        min_key = tl.min(tl.where(valid, key, _INT32_MAX))
        kth_f = _kth_largest_key(key, k, min_key, max_key)
        is_gt = key > kth_f
        is_tie = key == kth_f
        ngt = tl.sum(is_gt.to(tl.int32))
        ntie = tl.sum(is_tie.to(tl.int32))
        overflow = ngt + ntie > SLOTS
        if ngt + ntie <= SLOTS:
            to_slot = is_gt | is_tie
            sl = tl.cumsum(to_slot.to(tl.int32), axis=0) - 1
            tl.store(tmp_token_ptr + row * SLOTS + sl, tok, mask=to_slot)
            tl.store(tmp_logit_ptr + row * SLOTS + sl, lg, mask=to_slot)
            tl.debug_barrier()
            ss = tl.arange(0, SLOTS)
            sv = ss < ngt + ntie
            ttok = tl.load(tmp_token_ptr + row * SLOTS + ss, mask=sv, other=_INT32_MAX)
            tlg = tl.load(tmp_logit_ptr + row * SLOTS + ss, mask=sv, other=0.0)
            # Survivors in token order, as the stock compaction leaves them.
            srank = tl.sum(((ttok[None, :] < ttok[:, None]) & sv[None, :]).to(tl.int32), axis=1)
            tl.store(slot_token_ptr + row * SLOTS + srank, ttok, mask=sv)
            tl.store(slot_logit_ptr + row * SLOTS + srank, tlg, mask=sv)
            kth = kth_f
            num_gt = ngt
            num_slots = ngt + ntie
            # Entries consumed: clear them for this set's next use.
            tl.store(lst_ptr + off, tl.zeros([FAST_CAP], dtype=tl.int64), mask=valid)
        slow = overflow
    if slow:
        # The lists as a -inf row, then the stock tile-max and select code.
        for c0 in range(0, vocab_size, DENSE_BLOCK):
            cc = c0 + tl.arange(0, DENSE_BLOCK)
            tl.store(
                dense_ptr + row * dense_stride + cc,
                tl.full([DENSE_BLOCK], float("-inf"), tl.bfloat16),
                mask=cc < vocab_size,
            )
        tl.debug_barrier()
        for s in tl.static_range(WORLD):
            ls = tl.sum(tl.where(src == s, lens, 0), axis=0)
            for c0 in range(0, ls, DENSE_BLOCK):
                cc = c0 + tl.arange(0, DENSE_BLOCK)
                cv = cc < ls
                off = list_base + (s * ROWS + row) * NCAP + cc
                word = tl.load(lst_ptr + off, mask=cv, other=1, volatile=True)
                spins = tl.full([], 0, tl.int32)
                while (tl.min(word) == 0) & (spins < SPIN):
                    word = tl.load(lst_ptr + off, mask=cv, other=1, volatile=True)
                    spins += 1
                if spins >= SPIN:
                    tl.atomic_add(err_ptr, 1)
                tl.store(lst_ptr + off, tl.zeros([DENSE_BLOCK], dtype=tl.int64), mask=cv)
                tok = ((word >> 16) & 0xFFFFFFFF).to(tl.int32)
                lgb = (word & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
                tl.store(dense_ptr + row * dense_stride + tok, lgb, mask=cv)
        tl.debug_barrier()
        kth, num_gt, num_slots = _select_dense(
            dense_ptr,
            dense_stride,
            tile_max_ptr,
            tile_max_stride,
            cand_tile_ptr,
            slot_token_ptr,
            slot_logit_ptr,
            row,
            k,
            vocab_size,
            num_tiles,
            TILE,
            TILES_PAD,
            CAND_TILES,
            SLOTS,
            KEY_SHIFT,
            TILES_PER_ITER,
        )
    tl.store(path_ptr + row, slow.to(tl.int32) + 2 * total_len)
    tl.atomic_add(stats_ptr + 1, 1, sem="relaxed")
    tl.atomic_add(stats_ptr + 2, slow.to(tl.int64), sem="relaxed")
    tl.atomic_add(stats_ptr + 4, total_len.to(tl.int64), sem="relaxed")
    tl.debug_barrier()

    # From here on: _topk_select_sample_kernel's code, verbatim.
    slots = tl.arange(0, SLOTS)
    slot_valid = slots < num_slots
    s_token = tl.load(slot_token_ptr + row * SLOTS + slots, mask=slot_valid, other=0)
    s_logit = tl.load(
        slot_logit_ptr + row * SLOTS + slots, mask=slot_valid, other=float("-inf")
    )
    s_key = tl.where(slot_valid, _ordered_key(s_logit, KEY_SHIFT), _INT32_MIN)
    # The top-k: every logit above the k-th, then its lowest-id ties.
    s_tie = s_key == kth
    s_in_topk = (s_key > kth) | (
        s_tie & (tl.cumsum(s_tie.to(tl.int32), axis=0) <= k - num_gt)
    )

    # Temperature, as the stock temperature kernel applies it.
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 1.0:
        s_logit = s_logit / temp
    max_logit = tl.max(s_logit)
    s_exp = tl.where(slot_valid, tl.exp(s_logit - max_logit), 0.0)
    kept = s_in_topk
    if TOPP_ENABLED:
        p = tl.load(top_p_ptr + req_state_idx)
        if p < 1.0:
            probs = s_exp / tl.sum(tl.where(s_in_topk, s_exp, 0.0))
            at_or_above = s_key[None, :] >= s_key[:, None]
            mass = tl.sum(tl.where(at_or_above, probs[None, :], 0.0), axis=1)
            level = tl.max(tl.where(slot_valid & (mass >= p), s_key, _INT32_MIN))
            at_level = slot_valid & (s_key == level)
            level_mass = tl.max(tl.where(at_level, mass, 0.0))
            level_prob = tl.max(tl.where(at_level, probs, 0.0))
            num_keep = tl.sum(at_level.to(tl.int32)) - (
                (level_mass - p) / level_prob
            ).to(tl.int32)
            same_before = (s_key[None, :] == s_key[:, None]) & (
                slots[None, :] <= slots[:, None]
            )
            level_rank = tl.sum(same_before.to(tl.int32), axis=1)
            kept = slot_valid & (
                (s_key > level) | (at_level & (level_rank <= num_keep))
            )
    kept_logit = tl.where(kept, s_logit, float("-inf"))
    lse = max_logit + tl.log(tl.sum(tl.where(kept, s_exp, 0.0)))

    next_token = tl.load(
        draft_sampled_ptr + row + 1, mask=row + 1 < num_logits, other=-1
    ).to(tl.int32)
    is_next = s_token == next_token
    next_logit = tl.max(tl.where(is_next, kept_logit, float("-inf")))
    tl.store(next_logprob_ptr + row, next_logit - lse)

    seed = tl.load(seed_ptr + req_state_idx)
    pos = tl.load(pos_ptr + row)
    u = murmur3_uniform32(seed, pos, s_token)
    noised = kept_logit + -tl.log(-_log1p_neg_stable(u))
    best, best_slot = tl.max(noised, axis=0, return_indices=True)
    sample = tl.sum(tl.where(slots == best_slot, s_token, 0))
    sample = tl.where(best == float("-inf"), 0, sample)
    noised = tl.where(is_next, float("-inf"), noised)
    best, best_slot = tl.max(noised, axis=0, return_indices=True)
    residual = tl.sum(tl.where(slots == best_slot, s_token, 0))
    residual = tl.where(best == float("-inf"), 0, residual)
    tl.store(sample_ptr + row * 2, sample.to(tl.int64))
    tl.store(sample_ptr + row * 2 + 1, residual.to(tl.int64))

    # The last CTA advances the call counter.
    done = tl.atomic_add(sync_ptr + 2, 1, sem="acq_rel")
    if done == num_logits - 1:
        tl.store(sync_ptr + 2, 0)
        tl.store(epoch_ptr, epoch + 1)
        tl.atomic_add(stats_ptr, 1, sem="relaxed")


class TopkHeadState:
    """Symmetric slots, scratch and the int8 screen of one target lm_head."""

    def __init__(self, lm_head, device: torch.device) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        self.world = group.world_size
        self.rank = group.rank_in_group
        weight = lm_head.weight
        self.weight = weight
        self.n, self.k = weight.shape
        self.n_valid = int(lm_head.shard_indices.num_org_elements)
        self.vocab_start = int(lm_head.shard_indices.org_vocab_start_index)
        self.vocab = self.n_valid * self.world
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # The draft head's screen of the same weight, when it shares the head.
        screen = vps.get_screen(lm_head)
        if screen is None:
            raise RuntimeError("int8 screen of the target head unavailable")
        self.screen = screen
        n, rows = self.n, ROWS
        self.num_tiles = triton.cdiv(self.n_valid, _TILE)
        self.tiles_pad = triton.next_power_of_2(self.num_tiles)
        self.vocab_tiles = triton.cdiv(self.vocab, _TILE)
        name = group.device_group.group_name
        self.tau = symm_mem.empty((2, self.world, rows, KSLOTS), dtype=torch.int64, device=device)
        # Flagged list words [set, source rank, row of x, entry] and lengths.
        self.lst64 = symm_mem.empty((2, self.world, rows, n), dtype=torch.int64, device=device)
        self.hdr64 = symm_mem.empty((2, self.world, rows), dtype=torch.int64, device=device)
        self.tau.zero_()
        self.lst64.zero_()
        self.hdr64.zero_()
        handles = [
            symm_mem.rendezvous(t, name) for t in (self.tau, self.lst64, self.hdr64)
        ]
        assert all(h is not None for h in handles)
        ptrs = [torch.tensor([int(p) for p in h.buffer_ptrs], dtype=torch.int64, device=device)
                for h in handles]
        self.tau_ptrs, self.lst_ptrs, self.hdr_ptrs = ptrs
        self._handles = handles
        f32, i32 = torch.float32, torch.int32
        self.up = torch.empty((rows, n), dtype=f32, device=device)
        self.lot = torch.empty((rows, self.tiles_pad), dtype=f32, device=device)
        self.keys = torch.empty((rows, KSLOTS), dtype=i32, device=device)
        self.cut = torch.empty(rows, dtype=f32, device=device)
        # Candidate shard rows (union over rows of x) and their count; the
        # exact pass's per-row list lengths.
        self.lst = torch.empty(n, dtype=i32, device=device)
        self.len = torch.zeros(1, dtype=i32, device=device)
        self.olen = torch.zeros(rows, dtype=i32, device=device)
        # Split-K partials of the exact pass and per-tile arrivals.
        max_tiles = triton.cdiv(n, _EXACT_TILE)
        self.part = torch.empty(max_tiles * 6 * _EXACT_TILE * rows, dtype=f32, device=device)
        self.tile_cnt = torch.zeros(max_tiles, dtype=i32, device=device)
        self.sync = torch.zeros(4, dtype=i32, device=device)
        self.epoch = torch.zeros(1, dtype=i32, device=device)
        self.errors = torch.zeros(1, dtype=i32, device=device)
        self.tmp_token = torch.empty((rows, _SLOTS), dtype=i32, device=device)
        self.tmp_logit = torch.empty((rows, _SLOTS), dtype=f32, device=device)
        self.slot_token = torch.empty((rows, _SLOTS), dtype=i32, device=device)
        self.slot_logit = torch.empty((rows, _SLOTS), dtype=f32, device=device)
        self.dense = torch.empty((rows, self.vocab), dtype=torch.bfloat16, device=device)
        self.tile_max = torch.empty((rows, self.vocab_tiles), dtype=f32, device=device)
        self.cand_tile = torch.empty((rows, _CAND_TILES), dtype=i32, device=device)
        # Per row: 2 * list entries + 1 if the dense fallback ran (last call).
        self.path = torch.zeros(rows, dtype=i32, device=device)
        # [calls, rows, rows on the dense fallback, union candidate rows of
        # this rank, list entries], cumulative.
        self.stats = torch.zeros(5, dtype=torch.int64, device=device)
        self.stats_host = torch.zeros(5, dtype=torch.int64, pin_memory=True)
        self.dbg_tok = torch.zeros((rows, FAST_CAP), dtype=i32, device=device)
        self.dbg_lg = torch.zeros((rows, FAST_CAP), dtype=f32, device=device)
        from triton.tools.tensor_descriptor import TensorDescriptor

        bn, bk, _, _ = _SCREEN_CONFIG
        self.q_desc = TensorDescriptor.from_tensor(screen.q, [bn, bk])
        self.screen_grid = min(triton.cdiv(self.n_valid, bn), _SCREEN_CTAS_PER_SM * self.num_sms)
        self.exact_grid = _EXACT_CTAS_PER_SM * self.num_sms
        self.compact_grid = triton.cdiv(self.n_valid, _COMPACT_ROWS)
        torch.cuda.synchronize(device)
        handles[0].barrier()


def launch(
    state: TopkHeadState,
    x: torch.Tensor,
    variant: tuple[int, bool, int],
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    topp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(sample [M, 2], next_logprob [M]) as ``_topk_select_sample_kernel``
    returns them for the gathered stock logits of ``x`` [M, K]. Every TP rank
    calls this in lockstep."""
    m, kdim = x.shape
    assert 0 < m <= ROWS and kdim == state.k and x.stride(1) == 1
    nsplit, interleave, block_k = variant
    screen = state.screen
    bn, bk, nw, ns = _SCREEN_CONFIG
    spin = vpl.SPIN_LIMIT
    _screen_kernel[(state.screen_grid,)](
        x,
        state.q_desc,
        screen.scale,
        screen.coef,
        state.up,
        state.lot,
        state.keys,
        state.cut,
        state.sync,
        state.tau_ptrs,
        state.epoch,
        state.errors,
        expanded_idx_mapping,
        top_k,
        m,
        state.n_valid,
        x.stride(0),
        state.up.stride(0),
        state.lot.stride(0),
        K=kdim,
        BLOCK_N=bn,
        BLOCK_K=bk,
        TILE=_TILE,
        M_PAD=ROWS,
        NUM_STAGES=ns,
        TILES_PAD=state.tiles_pad,
        KSLOTS=KSLOTS,
        RANK=state.rank,
        WORLD=state.world,
        ROWS=ROWS,
        SPIN=spin,
        REL=vps.REL_SLACK,
        XN_SCALE=vps.XN_SLACK,
        MAX_XN=vps.MAX_XNORM,
        KEY16_MIN=-(1 << 15),
        PDL=USE_PDL,
        num_warps=nw,
    )
    _compact_kernel[(state.compact_grid,)](
        state.up,
        state.cut,
        state.lst,
        state.len,
        m,
        state.n_valid,
        state.up.stride(0),
        CH=_COMPACT_ROWS,
        M_PAD=ROWS,
        PDL=USE_PDL,
        num_warps=4,
        launch_pdl=USE_PDL,
    )
    _exact_kernel[(state.exact_grid,)](
        x,
        state.weight,
        state.lst,
        state.len,
        state.cut,
        state.olen,
        state.sync[3:],
        state.part,
        state.tile_cnt,
        state.lst_ptrs,
        state.hdr_ptrs,
        state.epoch,
        state.stats,
        m,
        state.vocab_start,
        x.stride(0),
        state.weight.stride(0),
        K=kdim,
        NSPLIT=nsplit,
        INTERLEAVE=interleave,
        BLOCK_K=block_k,
        TILE=_EXACT_TILE,
        M_PAD=ROWS,
        NUM_STAGES=_EXACT_STAGES,
        RANK=state.rank,
        WORLD=state.world,
        ROWS=ROWS,
        NCAP=state.n,
        num_warps=4,
    )
    sample = x.new_empty(m, 2, dtype=torch.int64)
    next_logprob = x.new_empty(m, dtype=torch.float32)
    _sample_kernel[(m,)](
        sample,
        next_logprob,
        state.lst64,
        state.hdr64,
        state.epoch,
        state.sync,
        state.errors,
        state.tmp_token,
        state.tmp_logit,
        state.slot_token,
        state.slot_logit,
        state.dense,
        state.dense.stride(0),
        state.tile_max,
        state.tile_max.stride(0),
        state.cand_tile,
        state.path,
        state.stats,
        state.dbg_tok,
        state.dbg_lg,
        draft_sampled,
        pos,
        expanded_idx_mapping,
        temperature,
        top_k,
        top_p,
        seed,
        m,
        state.vocab,
        state.vocab_tiles,
        WORLD=state.world,
        ROWS=ROWS,
        NCAP=state.n,
        FAST_CAP=FAST_CAP,
        TILE=_TILE,
        TILES_PAD=triton.next_power_of_2(state.vocab_tiles),
        CAND_TILES=_CAND_TILES,
        SLOTS=_SLOTS,
        KEY_SHIFT=16,
        TOPP_ENABLED=topp_enabled,
        SPIN=spin,
        DENSE_BLOCK=_DENSE_BLOCK,
        TILES_PER_ITER=_TILE_MAX_PER_ITER,
        PDL=USE_PDL,
        DEBUG=DEBUG,
        num_warps=4,
        launch_pdl=USE_PDL,
    )
    return sample, next_logprob


def gemv_variant(state_logits, m: int, k: int) -> tuple[int, bool, int] | None:
    """(NSPLIT, INTERLEAVE, BLOCK_K) of the push GEMV the stock head runs for
    m rows, or None when the stock head is not that GEMV."""
    if state_logits is None:
        return None
    v = state_logits.table.get(m, -2)
    if v < 0:
        return None
    nsplit, interleave = state_logits.variants[v]
    block_k = vpl._GEMV_CONFIGS[vpl._m_pad(m)][1]
    if interleave or (k // nsplit) % block_k:
        block_k = vpl.SPLIT_UNIT
    return nsplit, interleave, block_k


def rejection_from_samples(
    sample: torch.Tensor,
    next_logprob: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    synthetic_conditional_rates: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stock rejection kernel of ``topk_rejection_sample``."""
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(num_reqs, num_speculative_steps + 1, dtype=torch.int64)
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _topk_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        sample,
        next_logprob,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        seed,
        synthetic_conditional_rates,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        num_warps=1,
    )
    return sampled, num_sampled


class TargetHeadTopk:
    """Verify-path dispatch: the screened head plus the stock rejection rule
    when every request is on the sort-free top-k path, else None (stock)."""

    def __init__(self, model) -> None:
        self.model = model
        self.state: TopkHeadState | None = None
        self.failed = False
        self.calls = 0
        self.last = None
        self.lm_head = None
        self.logits_processor = None

    def _head(self):
        if self.lm_head is not None:
            return self.lm_head, self.logits_processor
        lm = self.model
        if hasattr(lm, "get_language_model"):
            try:
                lm = lm.get_language_model()
            except Exception:  # noqa: BLE001
                lm = self.model
        self.lm_head = getattr(lm, "lm_head", None)
        self.logits_processor = getattr(lm, "logits_processor", None)
        return self.lm_head, self.logits_processor

    def _eligible_head(self, x: torch.Tensor) -> bool:
        lm_head, lp = self._head()
        if lm_head is None or lp is None:
            return False
        return bool(
            not lp.logits_as_input
            and lp.scale == 1.0
            and lp.soft_cap is None
            and lp.use_all_gather
            and lp.head_dtype in (None, x.dtype)
            and getattr(lm_head, "tp_size", 1) > 1
            and getattr(lm_head, "bias", None) is None
            and lp.org_vocab_size == lm_head.shard_indices.num_org_elements * lm_head.tp_size
            and lm_head.shard_indices.num_org_vocab_padding == 0
            and lm_head.weight.shape[0] == lm_head.shard_indices.num_org_elements
            and vpl.supports_fused_logits(lm_head, x)
        )

    def _setup(self, lm_head, device) -> bool:
        from vllm.distributed.parallel_state import get_tp_group

        error = None
        state = None
        try:
            state = TopkHeadState(lm_head, device)
        except Exception as e:  # noqa: BLE001 - every rank falls back together
            error = e
            state = None
        ok = torch.tensor([int(state is not None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
        )
        if not ok.item():
            logger.warning("Target-head top-k screen disabled: %r", error)
            self.failed = True
            return False
        self.state = state
        logger.info(
            "Target-head top-k screen: exact top-k candidates of the %d x %d "
            "vocab shard from its int8 screen (TP%d, up to %d rows)",
            state.n,
            state.k,
            state.world,
            ROWS,
        )
        return True

    def warmup(
        self,
        hidden_states: torch.Tensor,
        rows: int,
        token_dtype: torch.dtype,
        position_dtype: torch.dtype,
    ) -> None:
        """Set up the buffers and compile the kernels before serving, on
        ``rows`` dummy rows (every TP rank calls this in lockstep)."""
        if self.failed or torch.cuda.is_current_stream_capturing():
            return
        m = min(ROWS, rows)
        x = hidden_states.new_zeros(m, hidden_states.shape[-1])
        if not self._eligible_head(x):
            return
        lm_head = self.lm_head
        variant = gemv_variant(
            vpl._get_state(lm_head, self.logits_processor.max_logits_rows), m, x.shape[1]
        )
        if variant is None:
            return
        if self.state is None and not self._setup(lm_head, x.device):
            return
        dev = x.device
        draft_sampled = torch.zeros(m, dtype=token_dtype, device=dev)
        pos = torch.arange(m, dtype=position_dtype, device=dev)
        expanded = torch.zeros(m, dtype=torch.int32, device=dev)
        temperature = torch.ones(1, dtype=torch.float32, device=dev)
        top_k = torch.full((1,), MAX_TOP_K, dtype=torch.int32, device=dev)
        top_p = torch.full((1,), 0.95, dtype=torch.float32, device=dev)
        seed = torch.zeros(1, dtype=torch.int64, device=dev)
        # Zero rows make every shard row a candidate: this compiles the
        # dense fallback too.
        for topp_enabled in (False, True):
            launch(self.state, x, variant, draft_sampled, pos, expanded, temperature,
                   top_k, top_p, seed, topp_enabled)
        torch.cuda.synchronize(dev)

    def verify(self, x: torch.Tensor, input_batch, rejection_sampler, speculator):
        """SamplerOutput of the verify step from hidden states ``x`` (the
        logits rows), or None for the stock path. Identical on every rank."""
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
            RejectionSampler,
            gather_draft_sampled,
        )
        from vllm.v1.worker.gpu.spec_decode.topk_rejection_sampler import (
            topk_path_eligible_requests,
        )

        if self.failed or not isinstance(rejection_sampler, RejectionSampler):
            return None
        m = x.shape[0]
        if (
            m == 0
            or m > ROWS
            or input_batch.num_reqs == 0
            or input_batch.num_draft_tokens == 0
            or x.dim() != 2
            or x.stride(1) != 1
        ):
            return None
        rs = rejection_sampler
        sampler = rs.sampler
        draft_logits = speculator.draft_logits if speculator is not None else None
        if (
            sampler.compute_nans
            or draft_logits is not None
            or rs.use_block_verification
            or rs.watermark_key is not None
            or rs.enable_adaptive_verification
            or sampler.use_fp64_gumbel
            or input_batch.has_structured_output_reqs
        ):
            return None
        states = sampler.sampling_states
        idx_np = input_batch.idx_mapping_np
        if states.max_num_logprobs(idx_np) != NO_LOGPROBS:
            return None
        if not topk_path_eligible_requests(
            states.temperature.np[idx_np],
            states.top_k.np[idx_np],
            states.min_p.np[idx_np],
            sampler.uses_logits_processors[idx_np],
        ):
            return None
        if not self._eligible_head(x):
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        lm_head = self.lm_head
        variant = gemv_variant(
            vpl._get_state(lm_head, self.logits_processor.max_logits_rows), m, x.shape[1]
        )
        if variant is None:
            return None
        if self.state is None and not self._setup(lm_head, x.device):
            return None
        assert int(np.max(states.top_k.np[idx_np])) <= MAX_TOP_K
        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            sampler.req_states.prefill_len.gpu,
        )
        xc = x if x.is_contiguous() else x.contiguous()
        sample, next_logprob = launch(
            self.state,
            xc,
            variant,
            draft_sampled,
            pos,
            input_batch.expanded_idx_mapping,
            states.temperature.gpu,
            states.top_k.gpu,
            states.top_p.gpu,
            states.seeds.gpu,
            topp_enabled=bool(np.any(states.top_p.np[idx_np] != 1.0)),
        )
        # The last call's per-row samples (diagnostics).
        self.last = (sample, next_logprob)
        sampled, num_sampled = rejection_from_samples(
            sample,
            next_logprob,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            states.seeds.gpu,
            rs.num_speculative_steps,
            rs.synthetic_conditional_rates,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        self.calls += 1
        if self.calls % _LOG_EVERY == 0:
            # The counters copied _LOG_EVERY calls ago (no synchronization).
            v = self.state.stats_host.tolist()
            if v[0]:
                logger.info(
                    "Target-head top-k screen: %d calls, %d rows, %d on the dense "
                    "fallback, %.0f union candidate rows per call on this rank, "
                    "%.0f list entries per row",
                    v[0], v[1], v[2], v[3] / v[0], v[4] / max(1, v[1]),
                )
            self.state.stats_host.copy_(self.state.stats, non_blocking=True)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
