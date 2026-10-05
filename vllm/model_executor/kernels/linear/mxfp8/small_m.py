# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-sized MXFP8 linear: one CuTe DSL launch that quantizes the bf16
activation and splits K over a CTA cluster (SM100/SM103).

``FlashInferCutedslMxfp8LinearKernel`` otherwise quantizes the activation in
one launch and runs ``mm_mxfp8``, whose low-M tile puts one CTA on each 128
weight rows: a decode-sized qkv_proj streams its weight through ~20 of the
148 SMs. Here the weight rows are the MMA's M side and the tokens its N side
(8..48), K is split across a cluster so ~all SMs stream the weight, the
partials are summed through DSMEM, and the activation is quantized in the
kernel with FlashInfer's MXFP8 rounding.

``mxfp8_linear`` is one opaque op that picks this path for 1 <= M <= 48 and
keeps quantize + ``mm_mxfp8`` otherwise, so a compiled caller never branches
on the token count.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

SMALL_M_MAX = 48
_CTA_M = 128
_CTA_K = 128
# Shared memory one CTA may allocate on SM100 (227 KB), and the most it may
# use for two CTAs to share an SM.
_SMEM_LIMIT = 232448
_SMEM_TWO_PER_SM = 115712

# Measured on B300 (TP4 MiniMax-M3 projections): (N, K) -> {token tile:
# (split_k, a_stages, x_stages)}. Where back-to-back and cold-launch timings
# disagree, the cold-launch winner is kept: the predecessor kernel does not
# always overlap the launch.
_QKV_CONFIGS = {
    8: (8, 4, 4),
    16: (8, 4, 4),
    24: (6, 6, 4),
    32: (8, 3, 2),
    40: (5, 6, 4),
    48: (6, 6, 4),
}
# qkv_proj of the sparse layers with the TP-sharded index-K cache: all 4
# index_q heads on every rank, 23 weight tiles. A split-6 cluster at one CTA
# per SM no longer fits in one wave, so token tiles 24 and 48 change split.
_QKV_SHARD_CONFIGS = {
    8: (8, 4, 4),
    16: (8, 4, 4),
    24: (8, 3, 4),
    32: (8, 3, 2),
    40: (5, 6, 4),
    48: (5, 6, 4),
}
_TUNED: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (2560, 6144): _QKV_CONFIGS,  # qkv_proj, sparse-attention layers
    (2944, 6144): _QKV_SHARD_CONFIGS,  # qkv_proj, sparse layers, sharded index-K
    (2304, 6144): _QKV_CONFIGS,  # qkv_proj, dense-attention layers
    (6144, 2048): {  # o_proj
        8: (4, 3, 4),
        16: (5, 4, 4),
        24: (2, 6, 4),
        32: (2, 6, 4),
        40: (2, 4, 4),
        48: (2, 4, 4),
    },
}


@functools.cache
def is_available() -> bool:
    from vllm.platforms import current_platform

    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return False
    return True


def supports_shape(n: int, k: int) -> bool:
    """Whole 128-row weight tiles and 128-wide K tiles."""
    return n % _CTA_M == 0 and k % _CTA_K == 0 and n > 0 and k > 0


@functools.cache
def _num_sms() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


@functools.cache
def select_config(m: int, n: int, k: int):
    """Token tile, cluster split and pipeline depths for one problem."""
    from ._small_m_kernel import SmallMConfig, smem_bytes

    cta_n = max(8, -(-m // 8) * 8)
    tuned = _TUNED.get((n, k), {}).get(cta_n)
    if tuned is not None:
        return SmallMConfig(cta_n, *tuned)
    tiles = -(-n // _CTA_M)
    k_tiles = k // _CTA_K
    # ~1.2 CTAs per SM, at most 8 per cluster; staying within two CTAs per SM
    # keeps every cluster resident in one wave.
    split_k = max(1, min(8, k_tiles, round(1.2 * _num_sms() / tiles)))
    for limit in (_SMEM_TWO_PER_SM, _SMEM_LIMIT):
        for a_stages, x_stages in ((4, 4), (4, 2), (3, 2), (6, 2), (8, 2)):
            cfg = SmallMConfig(cta_n, split_k, a_stages, x_stages)
            if smem_bytes(cfg) <= limit:
                return cfg
    raise ValueError(f"no small-M MXFP8 config fits for M={m} N={n} K={k}")


_compiled: dict = {}


def _compile(cfg, use_pdl: bool, push=None, prep=None):
    key = (cfg, use_pdl, push, prep)
    fn = _compiled.get(key)
    if fn is None:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_ptr

        from ._small_m_kernel import (
            SmallMMxfp8Gemm,
            SmallMMxfp8GemmPrep,
            SmallMMxfp8GemmPush,
        )

        def fake(dtype):
            # Placeholder address, aligned like the real operands.
            return make_ptr(dtype, 1 << 20, cutlass.AddressSpace.gmem, assumed_align=16)

        args = [
            fake(cutlass.Float8E4M3FN),
            fake(cutlass.Uint8),
            fake(cutlass.BFloat16),
            fake(cutlass.BFloat16),
            cutlass.Int32(_CTA_M),
            cutlass.Int32(8),
            cutlass.Int32(_CTA_K),
        ]
        if push is not None:
            gemm = SmallMMxfp8GemmPush(cfg, push)
            args += [cutlass.Int64(0), cutlass.Int64(0)]
        elif prep is not None:
            gemm = SmallMMxfp8GemmPrep(cfg, prep)
            args += [cutlass.Int64(0)] * 17 + [cutlass.Float32(0.0)] * 2
        else:
            gemm = SmallMMxfp8Gemm(cfg, use_pdl)
        fn = cute.compile(gemm, *args, make_fake_stream())
        _compiled[key] = fn
        logger.debug(
            "Compiled small-M MXFP8 GEMM %s (pdl=%s, push=%s, prep=%s)",
            cfg,
            use_pdl,
            push,
            prep,
        )
    return fn


@functools.cache
def _use_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


def small_m_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cfg=None,
) -> torch.Tensor:
    """bf16 ``x`` [M, K] (contiguous, 1 <= M <= 48) times the MXFP8 weight
    held as a [K, N] view of a contiguous [N, K] e4m3 tensor with swizzled
    128x4 E8M0 ``weight_scale``. Returns bf16 [M, N]."""
    import cutlass
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    from vllm.utils.torch_utils import current_stream

    M, K = x.shape
    N = weight.shape[1]
    if cfg is None:
        cfg = select_config(M, N, K)
    fn = _compile(cfg, _use_pdl())
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    gmem = cutlass.AddressSpace.gmem
    fn(
        make_ptr(cutlass.Float8E4M3FN, weight.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, weight_scale.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=16),
        cutlass.Int32(N),
        cutlass.Int32(M),
        cutlass.Int32(K),
        CUstream(current_stream().cuda_stream),
    )
    return out


def small_m_gemm_push(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    push,
    mailbox_address: int,
    stage_address: int,
) -> None:
    """``small_m_gemm`` whose output is multicast into a Lamport mailbox
    instead of returned (``_small_m_kernel.PushSpec``): the mailbox's
    multicast address, and the device int32 stage words whose first holds the
    generation. Same tile config, so the same partials as ``small_m_gemm``."""
    import cutlass
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    from vllm.utils.torch_utils import current_stream

    M, K = x.shape
    N = weight.shape[1]
    assert push.capacity_m >= M
    fn = _compile(select_config(M, N, K), True, push)
    gmem = cutlass.AddressSpace.gmem
    fn(
        make_ptr(cutlass.Float8E4M3FN, weight.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, weight_scale.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), gmem, assumed_align=16),
        # Shapes the output view only; push mode never writes it.
        make_ptr(cutlass.BFloat16, x.data_ptr(), gmem, assumed_align=16),
        cutlass.Int32(N),
        cutlass.Int32(M),
        cutlass.Int32(K),
        cutlass.Int64(mailbox_address),
        cutlass.Int64(stage_address),
        CUstream(current_stream().cuda_stream),
    )


def prep_supported(m: int, n: int, k: int) -> bool:
    """Whether ``small_m_gemm_prep`` runs this problem: an 8-token tile whose
    K is split over at least two CTAs (one per token parity)."""
    if not 1 <= m <= 8 or not supports_shape(n, k):
        return False
    cfg = select_config(m, n, k)
    return cfg.cta_n == 8 and cfg.split_k >= 2


def small_m_gemm_prep(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    prep,
    out: torch.Tensor | None,
    q_out: torch.Tensor | None,
    q_fp8_out: torch.Tensor | None,
    index_q_out: torch.Tensor | None,
    norm_weights: tuple[torch.Tensor | None, ...],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    index_slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    eps: float,
    q_fp8_scale: float,
) -> None:
    """``small_m_gemm`` (same tile config, so the same bf16 row) whose
    epilogue applies MiniMax-M3's attention prep to it
    (``_small_m_kernel.PrepSpec``) instead of storing it: the outputs of
    ``fused_minimax_m3_qknorm_rope_kv_insert``, bit for bit. A sparse layer
    writes ``q_out``, ``q_fp8_out`` (with ``prep.q_fp8``), ``index_q_out``
    and the e4m3 K/V and index-K caches; a dense layer writes the normed and
    roped [q | k | v] row to ``out``. ``norm_weights`` are the q, k, index_q
    and index_k Gemma norm weights."""
    import cutlass
    import numpy as np
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    from vllm.utils.torch_utils import current_stream

    M, K = x.shape
    N = weight.shape[1]
    assert prep_supported(M, N, K) and N == prep.row_heads * _CTA_M
    fn = _compile(select_config(M, N, K), True, prep=prep)
    sparse = prep.num_index_heads > 0
    if sparse:
        assert kv_cache is not None and kv_cache.stride(3) == 1
        kv_strides = kv_cache.stride()[:3]
        block_size = kv_cache.shape[2]
    else:
        assert out is not None and out.shape == (M, N) and out.is_contiguous()
        kv_strides = (0, 0, 0)
        block_size = 1

    def addr(t: torch.Tensor | None):
        return cutlass.Int64(0 if t is None else t.data_ptr())

    gmem = cutlass.AddressSpace.gmem
    fn(
        make_ptr(cutlass.Float8E4M3FN, weight.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, weight_scale.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), gmem, assumed_align=16),
        # A sparse layer's output view is only shaped, never written.
        make_ptr(
            cutlass.BFloat16,
            (x if out is None else out).data_ptr(),
            gmem,
            assumed_align=16,
        ),
        cutlass.Int32(N),
        cutlass.Int32(M),
        cutlass.Int32(K),
        addr(q_out),
        addr(q_fp8_out),
        addr(index_q_out),
        *(addr(w) for w in norm_weights),
        addr(cos_sin_cache),
        addr(positions),
        addr(slot_mapping),
        addr(index_slot_mapping),
        addr(kv_cache),
        addr(index_cache),
        *(cutlass.Int64(s) for s in kv_strides),
        cutlass.Int64(block_size),
        cutlass.Float32(eps),
        # The prep kernel's 1.0f / (float)scale.
        cutlass.Float32(float(np.float32(1.0) / np.float32(q_fp8_scale))),
        CUstream(current_stream().cuda_stream),
    )


def _small_m_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    M = x.shape[0]
    return (
        1 <= M <= SMALL_M_MAX
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.stride(0) == 1
        and weight.data_ptr() % 16 == 0
    )


def _mxfp8_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if _small_m_ok(x, weight):
        return small_m_gemm(x, weight, weight_scale)

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.utils import flashinfer as vllm_flashinfer

    x_mxfp8, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    return vllm_flashinfer.mm_mxfp8(
        x_mxfp8, weight, x_scale, weight_scale, out_dtype=x.dtype, backend="cute-dsl"
    )


def _mxfp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[1]))


direct_register_custom_op(
    op_name="mxfp8_linear_small_m",
    op_func=_mxfp8_linear_impl,
    fake_impl=_mxfp8_linear_fake,
)


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """bf16 ``x`` [M, K] times the MXFP8 ``weight`` ([K, N] view of a
    contiguous [N, K] tensor, swizzled 128x4 ``weight_scale``) -> bf16
    [M, N]: the fused small-M kernel for 1 <= M <= 48, quantize +
    ``mm_mxfp8`` otherwise."""
    return torch.ops.vllm.mxfp8_linear_small_m(x, weight, weight_scale)
