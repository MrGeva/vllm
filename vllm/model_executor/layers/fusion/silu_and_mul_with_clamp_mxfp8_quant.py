# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiluAndMulWithClamp fused with the consumer's dynamic MXFP8 quantization.

One Triton launch replaces the pair the unfused path runs:

* ``torch.ops._C.silu_and_mul_with_clamp`` (the vectorized ``packed_compute``
  path of csrc/activation_kernels.cu), and
* FlashInfer's ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``
  (CuTe-DSL backend), the input quantization of the FlashInfer MXFP8 linear
  kernels (key ``kMxfp8Dynamic``: E4M3 values and UE8M0 scales per 32
  elements, scales in the F8_128x4 swizzled layout).

The result is bitwise identical to that pair. The activation keeps the CUDA
kernel's rounding steps: the clamped gate is rounded to bf16, the SiLU of it is
rounded to bf16 (precise ``expf`` and IEEE division, as csrc is not built with
fast math), and the product with ``up + beta`` (``up`` clamped in fp32) is
rounded to bf16. The quantization then follows FlashInfer's arithmetic on those
bf16 values: the block maximum times ``1 / 448``, its UE8M0 exponent rounded
up, the inverse scale ``2 ** (127 - exponent)``, and the scaled values clamped
to +-448 and converted to E4M3 with round-to-nearest-even. Scale rows past the
last token, up to the 128-row tile, and scale columns past the last block, up
to a multiple of 4, are written as zero, as FlashInfer writes them.
"""

import torch

from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
)
from vllm.triton_utils import tl, triton

# Rows and scale columns of one F8_128x4 swizzle tile.
_SF_TILE_ROWS = 128
_SF_TILE_COLS = 4


@triton.jit
def _sf_offset(r, c, padded_sf_cols):
    # Byte offset of scale (row r, column c) in the F8_128x4 swizzled layout.
    return (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128).to(tl.int64) * 128 * padded_sf_cols
    )


@triton.jit
def _silu_and_mul_with_clamp_mxfp8_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    num_tokens,
    d,
    limit,
    alpha,
    beta,
    num_sf_cols,
    padded_sf_cols,
    padded_rows,
    BLOCK_M: tl.constexpr,
    SF_PER_PROGRAM: tl.constexpr,
):
    # Program (pid_m, pid_k) covers rows [pid_m * BLOCK_M, +BLOCK_M) and scale
    # columns [pid_k * SF_PER_PROGRAM, +SF_PER_PROGRAM); the last row program
    # also zeroes the scale rows past the data, up to the 128-row tile.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    sf_cols = pid_k * SF_PER_PROGRAM + tl.arange(0, SF_PER_PROGRAM)
    cols = pid_k * SF_PER_PROGRAM * 32 + tl.arange(0, SF_PER_PROGRAM * 32)

    row_mask = rows < num_tokens
    col_mask = cols < d
    mask = row_mask[:, None] & col_mask[None, :]

    row_off = rows.to(tl.int64)[:, None] * (2 * d)
    gate = tl.load(x_ptr + row_off + cols[None, :], mask=mask, other=0.0)
    up = tl.load(x_ptr + row_off + d + cols[None, :], mask=mask, other=0.0)

    # silu_and_mul_with_clamp, packed path: the clamped gate rounds to bf16,
    # SiLU(gate) rounds to bf16, the product rounds to bf16; up stays fp32.
    g = tl.minimum(gate.to(tl.float32), limit).to(tl.bfloat16).to(tl.float32)
    u = tl.maximum(tl.minimum(up.to(tl.float32), limit), -limit)
    e = tl.extra.cuda.libdevice.exp(-g * alpha)
    s = tl.div_rn(g, 1.0 + e).to(tl.bfloat16).to(tl.float32)
    act = (s * (u + beta)).to(tl.bfloat16).to(tl.float32)

    # FlashInfer MXFP8: per 32-element block, UE8M0 of amax / 448 rounded up.
    blocks = tl.reshape(tl.abs(act), (BLOCK_M, SF_PER_PROGRAM, 32))
    amax = tl.max(blocks, axis=2)
    v = amax * (1.0 / 448.0)
    bits = v.to(tl.uint32, bitcast=True)
    exp_biased = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    bump = ((mantissa != 0) & ~((exp_biased == 0) & (mantissa <= 0x400000))).to(
        tl.uint32
    )
    ue8m0 = tl.minimum(exp_biased + bump, 254)
    ue8m0 = tl.where(v <= 0.0, 0, ue8m0)
    inv_exp = tl.maximum(254 - ue8m0.to(tl.int32), 0)
    inv = (inv_exp << 23).to(tl.float32, bitcast=True)
    inv = tl.where(ue8m0 == 0, 0.0, inv)

    scaled = tl.reshape(act, (BLOCK_M, SF_PER_PROGRAM, 32)) * inv[:, :, None]
    scaled = tl.maximum(tl.minimum(scaled, 448.0), -448.0)
    q = tl.reshape(scaled, (BLOCK_M, SF_PER_PROGRAM * 32)).to(tl.float8e4nv)
    tl.store(q_ptr + rows.to(tl.int64)[:, None] * d + cols[None, :], q, mask=mask)

    # Zero for the padding columns of the scale tile.
    sf_valid = row_mask[:, None] & (sf_cols < num_sf_cols)[None, :]
    sf = tl.where(sf_valid, ue8m0, 0).to(tl.uint8)
    c = sf_cols[None, :]
    tl.store(
        sf_ptr + _sf_offset(rows[:, None], c, padded_sf_cols),
        sf,
        mask=row_mask[:, None] & (c < padded_sf_cols),
    )
    if pid_m == tl.num_programs(0) - 1:
        pad = num_tokens + tl.arange(0, 128)[:, None]
        tl.store(
            sf_ptr + _sf_offset(pad, c, padded_sf_cols),
            tl.zeros((128, SF_PER_PROGRAM), dtype=tl.uint8),
            mask=(pad < padded_rows) & (c < padded_sf_cols),
        )


def silu_and_mul_with_clamp_mxfp8_supported(
    act_fn: torch.nn.Module, x: torch.Tensor, linear: LinearBase
) -> bool:
    """Return whether the fused producer handles x.

    The fused kernel follows the unfused activation's vectorized path, which
    that kernel takes for every row of a contiguous bf16 tensor whose half
    width is a multiple of the MXFP8 block (32).
    """
    d = x.shape[-1] // 2
    return (
        isinstance(act_fn, SiluAndMulWithClamp)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() in (2, 3)
        and x.is_contiguous()
        and d > 0
        and d % MXFP8_BLOCK_SIZE == 0
    )


def silu_and_mul_with_clamp_mxfp8_quant(
    act_fn: SiluAndMulWithClamp, x: torch.Tensor, linear: LinearBase
) -> QuantizedActivation:
    """SiluAndMulWithClamp + dynamic MXFP8 quantization (F8_128x4 scales)."""
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    x2d = x.view(-1, 2 * d)
    num_tokens = x2d.shape[0]

    num_sf_cols = d // MXFP8_BLOCK_SIZE
    padded_sf_cols = triton.cdiv(num_sf_cols, _SF_TILE_COLS) * _SF_TILE_COLS
    padded_rows = triton.cdiv(max(num_tokens, 1), _SF_TILE_ROWS) * _SF_TILE_ROWS

    q = torch.empty((num_tokens, d), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    sf = torch.empty(
        padded_rows * padded_sf_cols, dtype=MXFP8_SCALE_DTYPE, device=x.device
    )

    # Swept on GB300 at d = 768..6144: one row per program up to 64 tokens.
    block_m = 1 if num_tokens <= 64 else 4
    sf_per_program = 8
    grid = (
        triton.cdiv(max(num_tokens, 1), block_m),
        triton.cdiv(padded_sf_cols, sf_per_program),
    )
    _silu_and_mul_with_clamp_mxfp8_kernel[grid](
        x2d,
        q,
        sf,
        num_tokens,
        d,
        act_fn.swiglu_limit,
        act_fn.alpha,
        act_fn.beta,
        num_sf_cols,
        padded_sf_cols,
        padded_rows,
        BLOCK_M=block_m,
        SF_PER_PROGRAM=sf_per_program,
        num_warps=4,
    )
    return QuantizedActivation(
        data=q.view(out_shape),
        scale=sf,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=kMxfp8Dynamic,
    )
