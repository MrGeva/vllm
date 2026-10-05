# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the MiniMax-M3 attention prep in the qkv_proj GEMM's epilogue.

Stock decode chain per attention layer:

    qkv_proj (small-M MXFP8 GEMM; bf16 [q | k | v | index_q | index_k] row to
    HBM) -> fused_minimax_m3_qknorm_rope_kv_insert (reloads the row: Gemma
    RMSNorm per head, partial NeoX RoPE, q / e4m3 q / e4m3 index_q stores,
    e4m3 K/V and index-K cache inserts) -> index decode score

With VLLM_MM3_QKV_EPILOGUE_PREP=1, calls of 1-8 tokens run the GEMM in prep
mode (``_small_m_kernel.PrepSpec``): each CTA's 128 weight rows are one head,
and its epilogue writes the prep kernel's outputs from the bf16-rounded head,
bit for bit, so the index decode score PDL-chains on the GEMM. Dense layers
get the normed and roped [q | k | v] row. Other calls keep the stock chain.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8 import small_m

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_MM3_QKV_EPILOGUE_PREP", "0") == "1"

_E4M3_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")


def applies(
    attn: torch.nn.Module, x: torch.Tensor, positions: torch.Tensor, sparse: bool
) -> bool:
    """Whether this layer's qkv_proj call takes the prep epilogue. Depends
    only on facts every TP rank shares."""
    qkv_proj = attn.qkv_proj
    if not (
        ENABLED
        and getattr(qkv_proj, "mxfp8_small_m_fused", False)
        and getattr(qkv_proj, "bias", None) is None
        and isinstance(x, torch.Tensor)
        and x.dim() == 2
        and small_m._small_m_ok(x, qkv_proj.weight)
        and positions.dtype == torch.int64
        and positions.is_contiguous()
        and positions.shape[0] >= x.shape[0]
        and attn.rotary_emb.cos_sin_cache.dtype == torch.bfloat16
        and attn.rotary_emb.cos_sin_cache.is_contiguous()
    ):
        return False
    if sparse:
        # The e4m3 K/V and index-K caches the epilogue inserts into.
        index_cache = attn.indexer.index_cache
        if not (
            attn.kv_cache_dtype in _E4M3_KV_CACHE_DTYPES
            and attn.kv_cache.dtype == torch.uint8
            and attn.kv_cache.dim() == 4
            and attn.kv_cache.stride(3) == 1
            and index_cache.dtype == torch.float8_e4m3fn
        ):
            return False
    k, n = qkv_proj.weight.shape
    return x.shape[1] == k and small_m.prep_supported(x.shape[0], n, k)


def _spec(attn: torch.nn.Module, num_index_heads: int, q_fp8: bool):
    from vllm.model_executor.kernels.linear.mxfp8._small_m_kernel import PrepSpec

    return PrepSpec(
        attn.num_heads,
        attn.num_kv_heads,
        num_index_heads,
        attn.rotary_emb.rotary_dim,
        q_fp8,
    )


def sparse_qkv_prep(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    q: torch.Tensor,
    query_fp8: torch.Tensor | None,
    index_q: torch.Tensor,
) -> None:
    """qkv_proj + fused_minimax_m3_qknorm_rope_kv_insert of a
    ``MiniMaxM3SparseAttention`` layer in one launch."""
    logger.info_once(
        "MiniMax-M3 qkv_proj epilogue prep: q/k/index norms, RoPE and cache "
        "inserts run in the qkv_proj GEMM epilogue for 1-8 tokens."
    )
    qkv_proj = attn.qkv_proj
    small_m.small_m_gemm_prep(
        hidden_states,
        qkv_proj.weight,
        qkv_proj.weight_scale,
        _spec(attn, attn.num_qkv_idx_heads, query_fp8 is not None),
        None,
        q,
        query_fp8,
        index_q,
        (
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.index_q_norm.weight,
            attn.index_k_norm.weight,
        ),
        attn.rotary_emb.cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        attn.kv_cache,
        attn.indexer.index_cache.kv_cache,
        attn.q_norm.variance_epsilon,
        attn._q_scale_float if query_fp8 is not None else 1.0,
    )


def dense_qkv_prep(
    attn: torch.nn.Module, hidden_states: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """qkv_proj + fused_minimax_m3_qknorm_rope_kv_insert of a
    ``MiniMaxM3Attention`` layer in one launch; returns the qkv row."""
    qkv_proj = attn.qkv_proj
    out = torch.empty(
        (hidden_states.shape[0], qkv_proj.weight.shape[1]),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    small_m.small_m_gemm_prep(
        hidden_states,
        qkv_proj.weight,
        qkv_proj.weight_scale,
        _spec(attn, 0, False),
        out,
        None,
        None,
        None,
        (attn.q_norm.weight, attn.k_norm.weight, None, None),
        attn.rotary_emb.cos_sin_cache,
        positions,
        None,
        None,
        None,
        None,
        attn.q_norm.variance_epsilon,
        1.0,
    )
    return out
