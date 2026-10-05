# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sort-free rejection sampling for batches whose requests all use a small top-k.

The generic verify path copies every logits row to FP32, masks it with the
full-vocab top-k/top-p kernel, reduces it again for the acceptance statistics,
and Gumbel-samples the resampled row over the whole vocab. With top_k <= 64
only a handful of tokens per row can ever be sampled, so this path:

1. reduces each row to per-tile maxima (one wide read of the logits);
2. per row, finds the k-th largest tile maximum, loads only the tiles that can
   hold the top-k, finds the exact k-th largest logit and compacts the
   survivors (at most ``_SLOTS`` tokens) in registers;
3. applies temperature and top-p over the survivors, computes the target
   log-probability of the next draft token, and draws the residual and bonus
   samples with the stock per-token Gumbel noise; and
4. runs the stock acceptance rule per request.

The masking semantics follow the stock top-k/top-p kernel: top-k keeps exactly
k tokens (ties at the k-th logit by lowest token id) and normalizes over them;
top-p finds the smallest probability level whose cumulative mass reaches p and
keeps every token above it plus the fewest ties at it (lowest token id first)
that reach p. Sampling reuses the stock Gumbel noise keyed by (seed, position,
token id), so the same masked distribution yields the same token.
"""

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _log1p_neg_stable,
    murmur3_uniform32,
    tl_rand32,
)

# Largest supported top_k. Every request in the batch must use 1 <= k <= this.
MAX_TOP_K = 64
# Vocab tile whose maximum bounds the tokens it can contribute to the top-k.
_TILE = 64
# Candidate tiles loaded per row (must be >= MAX_TOP_K).
_CAND_TILES = 64
# Tokens compacted per row: the top-k plus further ties at the k-th logit
# (must be >= MAX_TOP_K).
_SLOTS = 64
# Tiles reduced by one program of the tile-max kernel.
_TILES_PER_PROGRAM = 32

_INT32_MIN = tl.constexpr(-(2**31))
_INT32_MAX = tl.constexpr(2**31 - 1)


@triton.jit
def _ordered_key(x, KEY_SHIFT: tl.constexpr):
    # Order-preserving int32 key of an fp32 value (+0.0 and -0.0 map together).
    # KEY_SHIFT drops the low mantissa bits that are zero for 16-bit inputs.
    bits = (x + 0.0).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) & 0x7FFFFFFF)) >> KEY_SHIFT


@triton.jit
def _kth_largest_key(keys, k, lo, hi):
    # Largest t in [lo, hi] with count(keys >= t) >= k, given count(keys >= lo)
    # >= k.
    lo = lo.to(tl.int64)
    hi = hi.to(tl.int64)
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        count = tl.sum((keys >= mid.to(tl.int32)).to(tl.int32))
        if count >= k:
            lo = mid
        else:
            hi = mid - 1
    return lo.to(tl.int32)


@triton.jit
def _tile_max_kernel(
    logits_ptr,
    logits_stride,
    tile_max_ptr,
    tile_max_stride,
    vocab_size,
    num_tiles,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.program_id(1) * TILES_PER_PROGRAM + tl.arange(0, TILES_PER_PROGRAM)
    offs = tiles[:, None] * TILE + tl.arange(0, TILE)[None, :]
    logits = tl.load(
        logits_ptr + row * logits_stride + offs,
        mask=offs < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    tl.store(
        tile_max_ptr + row * tile_max_stride + tiles,
        tl.max(logits, axis=1),
        mask=tiles < num_tiles,
    )


# num_logits changes every batch; specializing on it (== 1, % 16) would compile
# new variants mid-serving instead of at warmup.
@triton.jit(do_not_specialize=["num_logits"])
def _topk_select_sample_kernel(
    # [num_logits, 2]: sample over the kept tokens, and the same sample with
    # the next row's draft token excluded (the residual after rejecting it).
    sample_ptr,
    # [num_logits]: target log-prob of the next row's draft token.
    next_logprob_ptr,
    # [num_logits, num_tiles]
    tile_max_ptr,
    tile_max_stride,
    # [num_logits, CAND_TILES] scratch
    cand_tile_ptr,
    # [num_logits, SLOTS] scratch
    slot_token_ptr,
    slot_logit_ptr,
    # [num_logits, V]
    logits_ptr,
    logits_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    top_k_ptr,
    top_p_ptr,
    seed_ptr,
    num_logits,
    vocab_size,
    num_tiles,
    TILE: tl.constexpr,
    TILES_PAD: tl.constexpr,
    CAND_TILES: tl.constexpr,
    SLOTS: tl.constexpr,
    KEY_SHIFT: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    k = tl.load(top_k_ptr + req_state_idx)

    # The k-th largest tile maximum tau lower-bounds the k-th largest logit, so
    # the top-k lies in tiles whose maximum is >= tau. Take every tile above
    # tau, then tiles equal to tau in token order: with CAND_TILES >= k these
    # contain every logit above tau and the lowest-id ties at tau the top-k
    # can need.
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

    # Load the candidate tiles as one flat vector, in token order.
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

    # Exact k-th largest logit. Compact every logit above it and the ties at
    # it (which the stock top-p search also counts) into SLOTS, in token
    # order; with more ties than fit, keep the lowest-id ones.
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
    tl.debug_barrier()

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
            # Probabilities are normalized over the top-k. The boundary is the
            # smallest logit level whose cumulative mass reaches p; keep every
            # logit above it and, of the ties at it, only as many (lowest token
            # id first) as reaching p needs.
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
            # Slots of one level are in token order.
            same_before = (s_key[None, :] == s_key[:, None]) & (
                slots[None, :] <= slots[:, None]
            )
            level_rank = tl.sum(same_before.to(tl.int32), axis=1)
            kept = slot_valid & (
                (s_key > level) | (at_level & (level_rank <= num_keep))
            )
    kept_logit = tl.where(kept, s_logit, float("-inf"))
    lse = max_logit + tl.log(tl.sum(tl.where(kept, s_exp, 0.0)))

    # The token this row verifies is the next row's input token.
    next_token = tl.load(
        draft_sampled_ptr + row + 1, mask=row + 1 < num_logits, other=-1
    ).to(tl.int32)
    is_next = s_token == next_token
    next_logit = tl.max(tl.where(is_next, kept_logit, float("-inf")))
    tl.store(next_logprob_ptr + row, next_logit - lse)

    # Gumbel-max with the stock per-token noise. When nothing survives, the
    # stock argmax over an all -inf row returns token 0.
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


@triton.jit
def _topk_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, 2]
    sample_ptr,
    # [num_logits]
    next_logprob_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)

    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # A -1 placeholder draft ends verification.
        verifying &= draft_sampled >= 0
        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
            if SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + i)
                accepted = u < rate
            else:
                # One-hot draft: accept iff p(x) > u.
                accepted = tl.load(next_logprob_ptr + logit_idx) > tl.log(u)
            verifying = accepted
            accepted_length += accepted
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                tl.maximum(0, draft_sampled),
            )

    # Resample the first rejected position from the residual, or sample the
    # bonus token. A rejected -1 placeholder resamples the full distribution.
    resample_idx = start_idx + accepted_length
    is_bonus = resample_idx == end_idx - 1
    rejected_draft = tl.load(
        draft_sampled_ptr + resample_idx + 1, mask=not is_bonus, other=-1
    )
    use_residual = rejected_draft >= 0
    token = tl.load(sample_ptr + resample_idx * 2 + use_residual.to(tl.int64))
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, token)
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


def topk_rejection_sample(
    # [num_logits, V]
    logits: torch.Tensor,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    topp_enabled: bool,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection sampling for one-hot drafts when every request has a positive
    temperature, 1 <= top_k <= MAX_TOP_K, and no other logits processing.

    Returns (sampled [num_reqs, num_speculative_steps + 1], num_sampled
    [num_reqs]) like ``rejection_sample``.
    """
    assert logits.ndim == 2 and logits.stride(-1) == 1
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = logits.shape
    key_shift = 16 if logits.dtype in (torch.bfloat16, torch.float16) else 0
    if logits.dtype == torch.float16:
        key_shift = 13
    num_tiles = triton.cdiv(vocab_size, _TILE)

    tile_max = logits.new_empty(num_logits, num_tiles, dtype=torch.float32)
    _tile_max_kernel[(num_logits, triton.cdiv(num_tiles, _TILES_PER_PROGRAM))](
        logits,
        logits.stride(0),
        tile_max,
        tile_max.stride(0),
        vocab_size,
        num_tiles,
        TILE=_TILE,
        TILES_PER_PROGRAM=_TILES_PER_PROGRAM,
    )

    sample = logits.new_empty(num_logits, 2, dtype=torch.int64)
    next_logprob = logits.new_empty(num_logits, dtype=torch.float32)
    cand_tile = logits.new_empty(num_logits, _CAND_TILES, dtype=torch.int32)
    slot_token = logits.new_empty(num_logits, _SLOTS, dtype=torch.int32)
    slot_logit = logits.new_empty(num_logits, _SLOTS, dtype=torch.float32)
    _topk_select_sample_kernel[(num_logits,)](
        sample,
        next_logprob,
        tile_max,
        tile_max.stride(0),
        cand_tile,
        slot_token,
        slot_logit,
        logits,
        logits.stride(0),
        draft_sampled,
        pos,
        expanded_idx_mapping,
        temperature,
        top_k,
        top_p,
        seed,
        num_logits,
        vocab_size,
        num_tiles,
        TILE=_TILE,
        TILES_PAD=triton.next_power_of_2(num_tiles),
        CAND_TILES=_CAND_TILES,
        SLOTS=_SLOTS,
        KEY_SHIFT=key_shift,
        TOPP_ENABLED=topp_enabled,
        num_warps=4,
    )

    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
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


def warmup_topk_rejection_sample(
    # [n, V]: rows with the served logits' dtype and row stride
    logits: torch.Tensor,
    num_speculative_steps: int,
    synthetic_conditional_rates: torch.Tensor | None,
    token_dtype: torch.dtype,
    position_dtype: torch.dtype,
) -> None:
    """Compile this path's kernels before serving: two dummy requests
    (temperature 1, top_k MAX_TOP_K) through both top-p variants."""
    num_reqs = 2
    num_logits = num_reqs * (num_speculative_steps + 1)
    device = logits.device
    if logits.shape[0] < num_logits:
        logits = logits.new_zeros(num_logits, logits.shape[1])
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(num_speculative_steps + 1)
    cu_num_logits = torch.arange(
        0,
        num_logits + 1,
        num_speculative_steps + 1,
        dtype=torch.int32,
        device=device,
    )
    draft_sampled = torch.zeros(num_logits, dtype=token_dtype, device=device)
    pos = torch.arange(num_logits, dtype=position_dtype, device=device)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=device)
    top_k = torch.full((num_reqs,), MAX_TOP_K, dtype=torch.int32, device=device)
    top_p = torch.full((num_reqs,), 0.95, dtype=torch.float32, device=device)
    seed = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    for topp_enabled in (False, True):
        topk_rejection_sample(
            logits[:num_logits],
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            temperature,
            top_k,
            top_p,
            seed,
            num_speculative_steps,
            topp_enabled=topp_enabled,
            synthetic_conditional_rates=synthetic_conditional_rates,
        )


def topk_path_eligible_requests(
    temperature_np: np.ndarray,
    top_k_np: np.ndarray,
    min_p_np: np.ndarray,
    uses_logits_processors_np: np.ndarray,
) -> bool:
    """Whether the requests' sampling reduces to temperature, top-k <=
    MAX_TOP_K and top-p (all arrays indexed by the batch's request states)."""
    return bool(
        np.all(temperature_np > 0.0)
        and np.all(top_k_np <= MAX_TOP_K)
        and np.all(min_p_np == 0.0)
        and not np.any(uses_logits_processors_np)
    )
