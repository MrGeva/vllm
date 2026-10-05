# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernel behind ``small_m.py``: decode-sized MXFP8 linear with the
activation quantized inside the GEMM and K split over a CTA cluster.

Structure adapted from FlashInfer's ``LowLatencyBlockscaledGemmKernel``
(flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py,
BSD-3-Clause): swap-AB tcgen05 block-scaled MMA with the 128 weight rows as
the MMA M side and the tokens as its N side, TMA weight/scale loads, and a
DSMEM mailbox reduction of the split-K partials. Changes: the activation is
loaded as bf16 and quantized to MXFP8 by a warpgroup pair before the
mainloop consumes it (bit-identical to FlashInfer's CuTe DSL
``mxfp8_quantize``), the token tile spans 8..48, the split count need not
divide the token tile, and the partials move as 16-byte DSMEM stores.

Quantization matches FlashInfer step for step: bf16 abs-max of each 32-block,
times fp32(1/448), E8M0 rounded up, then x * 2^(127 - e) clamped to +-448 and
rounded to e4m3. The scaling and clamp run in bf16x2: the scale is a power of
two and |x * scale| <= 448, so the products are exact and the e4m3 rounding
sees the same values as FlashInfer's fp32 path.

Warp roles (352 threads):
  0-7   activation quantization (bf16 smem -> e4m3 B tile + E8M0 SFB);
        warps 0-3 then run the epilogue (TMEM -> registers -> split-K
        reduction -> bf16 store)
  8     TMA: weight tile + weight scales (issued before the PDL wait)
  9     tcgen05 MMA
  10    TMA: bf16 activation tile (after the PDL wait)

Push mode (``PushSpec``, a row-parallel linear whose all-reduce follows):
instead of the bf16 store, the epilogue multicasts the output over NVLink into
a Lamport mailbox ``[3, tp, capacity_m, N]`` of bf16 (-0.0 is the empty
sentinel, so a -0.0 output is sent as +0.0, as FlashInfer's one-shot
all-reduce does). The TMEM load gives each thread token pairs of one row and
lane ^ 4 the next row, so one shuffle forms each 4-byte (row, row + 1) store.
The generation is int32 word ``stage[0]``, which the consumer kernel advances:
the epilogue warps read it after their PDL wait and only then trigger the
dependent launch, so every CTA has read it before the consumer starts. The
consumer may therefore run, and finish, before this grid ends; it reads the
output from the mailbox.

Prep mode (``PrepSpec``, the MiniMax-M3 qkv_proj, token tile 8): each CTA's
128 weight rows are one 128-dim head, and the epilogue does the work of
``fused_minimax_m3_qknorm_rope_kv_insert`` on it instead of storing the bf16
row: per-head Gemma RMSNorm and partial NeoX RoPE, then the q / e4m3 q /
e4m3 index_q stores and the e4m3 K/V and index-K cache inserts (a dense
layer stores the normed and roped [q | k | v] row). Cluster ranks 0 and 1
reduce the even and odd tokens (all 128 rows each), so every element's
split-K sum keeps the stock order. The bf16-rounded head goes through shared
memory into the prep kernel's layout (a warp per token, 4 dims per lane), and
the fp32 math repeats that kernel's instructions with explicit rounding, so
every output byte equals the stock GEMM + prep kernel's.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import PipelineUserType, make_pipeline_state

CTA_M = 128
CTA_K = 128
SF_VEC = 32
# Bytes of one 128-row x 4-column E8M0 scale atom (one K tile of 128).
SF_ATOM_BYTES = 512
NUM_THREADS = 11 * 32
QUANT_WARPS = 8
QUANT_THREADS = QUANT_WARPS * 32
EPI_THREADS = 128
TMA_A_WARP = 8
MMA_WARP = 9
TMA_X_WARP = 10
TMEM_COLS = 128
# FlashInfer's quantizer multiplies the block amax by fp32(1 / 448).
_INV_E4M3_MAX = 1.0 / 448.0


class SmallMConfig(NamedTuple):
    cta_n: int  # token tile (MMA N), multiple of 8, <= 48
    split_k: int  # CTAs per cluster splitting K
    a_stages: int  # weight-tile pipeline depth
    x_stages: int  # activation-tile pipeline depth


class PushSpec(NamedTuple):
    tp: int  # ranks sharing the multicast mailbox
    rank: int  # this rank's slot
    capacity_m: int  # token rows per (generation, rank) slot


# Push-mode mailbox: generations, the empty bf16 (-0.0) and the stage-state
# word holding the current generation.
PUSH_GENERATIONS = 3
PUSH_SENTINEL = 0x8000
PUSH_STAGE_WORD = 0


class PrepSpec(NamedTuple):
    """Attention prep in the qkv_proj epilogue: this rank's head counts of the
    [q | k | v | index_q | index_k] row (``num_index_heads`` 0 for a dense
    [q | k | v] row), the RoPE width and whether the e4m3 q is stored too."""

    num_heads: int
    num_kv_heads: int
    num_index_heads: int
    rotary_dim: int
    q_fp8: bool

    @property
    def row_heads(self) -> int:
        idx = self.num_index_heads + 1 if self.num_index_heads else 0
        return self.num_heads + 2 * self.num_kv_heads + idx


class PrepArgs(NamedTuple):
    """Device operands of the prep epilogue (global addresses and scalars)."""

    q_out: Int64  # sparse: [M, nq * 128] bf16
    q_fp8_out: Int64  # sparse: [M, nq * 128] e4m3
    index_q_out: Int64  # sparse: [M, niq * 128] e4m3
    q_norm: Int64
    k_norm: Int64
    index_q_norm: Int64
    index_k_norm: Int64
    cos_sin: Int64  # [max_pos, rotary_dim] bf16
    positions: Int64  # [M] int64
    slot_mapping: Int64  # [M] int64
    index_slot_mapping: Int64  # [M] int64
    kv_cache: Int64  # e4m3 [blocks, nkv, block_size, 2 * 128]
    index_cache: Int64  # e4m3 [slots, 128]
    kv_stride_block: Int64
    kv_stride_head: Int64
    kv_stride_token: Int64
    block_size: Int64
    eps: Float32
    q_inv_scale: Float32


# Named barrier of the prep epilogue's 4 warps (0 is the CTA-wide barrier).
PREP_BARRIER = 1
# Prep-kernel float32 max of the e4m3 clamp, and its 1 / 128 head-dim mean.
_E4M3_MAX = 448.0
_INV_HEAD_DIM = 1.0 / 128.0


def shard_size(cta_n: int, split_k: int) -> int:
    """Token columns each cluster rank reduces and stores, a multiple of 4
    so the partials travel as 16-byte DSMEM stores."""
    per_rank = -(-cta_n // split_k)
    return -(-per_rank // 4) * 4


def smem_bytes(cfg: SmallMConfig) -> int:
    """Shared memory the kernel allocates for ``cfg`` (upper bound)."""
    mailbox = (cfg.split_k - 1) * CTA_M * shard_size(cfg.cta_n, cfg.split_k) * 4
    a_stage = CTA_M * CTA_K + SF_ATOM_BYTES
    x_stage = cfg.cta_n * CTA_K * 2 + cfg.cta_n * CTA_K + SF_ATOM_BYTES
    barriers = 8 * (2 * cfg.a_stages + 3 * cfg.x_stages + 4) + 16
    # 1 KB alignment for the swizzled A/B buffers, 128 B for the rest.
    return a_stage * cfg.a_stages + x_stage * cfg.x_stages + mailbox + barriers + 4096


# ---------------------------------------------------------------------------
# Inline PTX helpers. The quantization helpers reproduce FlashInfer's
# quantization_cute_dsl_utils (float_to_ue8m0_fast, ue8m0_to_inv_scale_fast,
# bfloat2_to_fp8x2_scaled, bf16x2 abs-max) instruction for instruction.
# ---------------------------------------------------------------------------


def _asm(ret, args, text, constraints, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _st_shared_u8(addr: Int32, val: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.u8 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4(
    addr: Int32, v0, v1, v2, v3, mbar: Int32, peer: Int32, *, loc=None, ip=None
):
    """16-byte store into cluster rank ``peer``'s shared memory at the same
    offset as local ``addr``, completing ``peer``'s copy of mbarrier ``mbar``."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            *[Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            Int32(mbar).ir_value(loc=loc, ip=ip),
            Int32(peer).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 ra, rb;
            mapa.shared::cluster.u32 ra, $0, $6;
            mapa.shared::cluster.u32 rb, $5, $6;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32
                [ra], {$1, $2, $3, $4}, [rb];
        }
        """,
        "r,f,f,f,f,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4_f32(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _bf16x16_absmax(v, *, loc=None, ip=None) -> Float32:
    """Max |x| over 16 bf16 (8 packed pairs), as f32."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip) for x in v],
            """
            {
                .reg .b32 a<8>, lo, hi;
                .reg .f32 f0, f1;
                and.b32 a0, $1, 0x7FFF7FFF;
                and.b32 a1, $2, 0x7FFF7FFF;
                and.b32 a2, $3, 0x7FFF7FFF;
                and.b32 a3, $4, 0x7FFF7FFF;
                and.b32 a4, $5, 0x7FFF7FFF;
                and.b32 a5, $6, 0x7FFF7FFF;
                and.b32 a6, $7, 0x7FFF7FFF;
                and.b32 a7, $8, 0x7FFF7FFF;
                max.bf16x2 a0, a0, a1;
                max.bf16x2 a2, a2, a3;
                max.bf16x2 a4, a4, a5;
                max.bf16x2 a6, a6, a7;
                max.bf16x2 a0, a0, a2;
                max.bf16x2 a4, a4, a6;
                max.bf16x2 a0, a0, a4;
                and.b32 lo, a0, 0xFFFF;
                shr.b32 hi, a0, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r,r,r,r,r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fmax(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _float_to_ue8m0(value: Float32, *, loc=None, ip=None) -> Uint32:
    """E8M0 of ``value`` rounded toward +inf, saturated to 254, 0 for <= 0."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;
                setp.le.f32 p_zero, $1, 0f00000000;
                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;
                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                setp.eq.u32 p_exp_zero, exp_biased, 0;
                setp.le.u32 p_tiny_sub, mantissa, 0x400000;
                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
                @p_tiny_sub mov.u32 bump, 0;
                add.u32 result, exp_biased, bump;
                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ue8m0_to_inv_scale_bf16x2(e: Uint32, *, loc=None, ip=None) -> Uint32:
    """2^(127 - e) as a bf16 pair (0 for e == 0 and e == 254), the value
    FlashInfer multiplies by in fp32."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(e).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 b, hi;
                .reg .pred p_zero;
                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 b, new_exp, 7;
                @p_zero mov.b32 b, 0;
                shl.b32 hi, b, 16;
                or.b32 $0, b, hi;
            }
            """,
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x8_to_e4m3x8(v0, v1, v2, v3, inv2, *, loc=None, ip=None):
    """8 bf16 (4 packed pairs) * inv2, clamped to +-448, rounded to e4m3 with
    element 0 in the low byte -> two 32-bit words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (v0, v1, v2, v3, inv2)],
        """
        {
            .reg .b32 t0, t1, t2, t3, cmax, cmin;
            .reg .b16 p0, p1, p2, p3;
            mov.b32 cmax, 0x43E043E0;
            mov.b32 cmin, 0xC3E0C3E0;
            mul.rn.bf16x2 t0, $2, $6;
            mul.rn.bf16x2 t1, $3, $6;
            mul.rn.bf16x2 t2, $4, $6;
            mul.rn.bf16x2 t3, $5, $6;
            min.bf16x2 t0, t0, cmax;
            min.bf16x2 t1, t1, cmax;
            min.bf16x2 t2, t2, cmax;
            min.bf16x2 t3, t3, cmax;
            max.bf16x2 t0, t0, cmin;
            max.bf16x2 t1, t1, cmin;
            max.bf16x2 t2, t2, cmin;
            max.bf16x2 t3, t3, cmin;
            cvt.rn.satfinite.e4m3x2.bf16x2 p0, t0;
            cvt.rn.satfinite.e4m3x2.bf16x2 p1, t1;
            cvt.rn.satfinite.e4m3x2.bf16x2 p2, t2;
            cvt.rn.satfinite.e4m3x2.bf16x2 p3, t3;
            mov.b32 $0, {p0, p1};
            mov.b32 $1, {p2, p3};
        }
        """,
        "=r,=r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _st_shared_v4(addr: Int32, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_volatile_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    """bf16 bits of ``value`` (the stock output store's rounding) in the low
    half, with -0.0 (the mailbox sentinel) sent as +0.0."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h;
                .reg .b32 w;
                .reg .pred p;
                cvt.rn.bf16.f32 h, $1;
                cvt.u32.u16 w, h;
                setp.eq.b32 p, w, 0x8000;
                selp.b32 $0, 0, w, p;
            }
            """,
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_xor_u32(value: Uint32, mask: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {mask}, 31, -1;",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_multicast_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.b32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Prep-mode helpers. The float ops carry explicit rounding so that ptxas emits
# exactly the FFMA / FMUL / FADD sequence of the prep kernel (nvcc's
# contraction of ``e * c - p * s`` is fma(e, c, -(p * s))); rsqrt.approx.f32 is
# rsqrtf without fast math (MUFU.RSQ with the denormal rescale).


def _side_load(ret, addr: Int64, text: str, constraints: str, loc, ip):
    # Volatile, so the loads stay ahead of the mainloop waits they overlap.
    return llvm.inline_asm(
        ret,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_nc_v2_u32(addr: Int64, *, loc=None, ip=None):
    res = _side_load(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        addr,
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def _ld_global_nc_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_side_load(T.i64(), addr, "ld.global.nc.s64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def _st_global_u32(addr: Int64, w, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v2_u32(addr: Int64, w0, w1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_u16(addr: Int32, val: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.u16 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v2_u32(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


def _f32_op(text: str, *args, loc, ip) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip) for a in args],
            text,
            "=f" + ",f" * len(args),
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    return _f32_op("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _fma_neg_rn(a, b, p, s, *, loc=None, ip=None) -> Float32:
    """fma(a, b, -(p * s))."""
    return _f32_op(
        "{ .reg .f32 t; mul.rn.f32 t, $3, $4; neg.f32 t, t; "
        "fma.rn.f32 $0, $1, $2, t; }",
        a,
        b,
        p,
        s,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_op("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_op("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _rsqrt_approx(a, *, loc=None, ip=None) -> Float32:
    return _f32_op("rsqrt.approx.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_to_f32(w, *, loc=None, ip=None):
    """A bf16 pair (element 0 in the low half) as two exact f32."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            shl.b32 lo, $2, 16;
            and.b32 hi, $2, 0xFFFF0000;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def _f32x2_to_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Round-to-nearest bf16 pair, ``a`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _f32_to_bf16_u16(a, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _f32x4_to_e4m3x4(v0, v1, v2, v3, clamp: bool, *, loc=None, ip=None) -> Uint32:
    """4 f32 -> 4 e4m3 bytes, element 0 lowest: satfinite rounding, after the
    prep kernel's fminf(fmaxf(x, -448), 448) when ``clamp`` (index_q, index-K
    and e4m3 q; the K/V cache converts directly)."""
    pre = (
        """
            max.f32 a0, $1, 0fC3E00000;
            max.f32 a1, $2, 0fC3E00000;
            max.f32 a2, $3, 0fC3E00000;
            max.f32 a3, $4, 0fC3E00000;
            min.f32 a0, a0, 0f43E00000;
            min.f32 a1, a1, 0f43E00000;
            min.f32 a2, a2, 0f43E00000;
            min.f32 a3, a3, 0f43E00000;
        """
        if clamp
        else """
            mov.f32 a0, $1;
            mov.f32 a1, $2;
            mov.f32 a2, $3;
            mov.f32 a3, $4;
        """
    )
    return Uint32(
        _asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{ .reg .f32 a0, a1, a2, a3; .reg .b16 h0, h1;"
            + pre
            + """
            cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;
            cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;
            mov.b32 $0, {h0, h1};
        }
            """,
            "=r,f,f,f,f",
            loc,
            ip,
        )
    )


class SmallMMxfp8Gemm:
    """out[M, N] = mxfp8(x)[M, K] @ w[N, K]^T, bf16 x quantized in-kernel.

    Grid: (ceil(N / 128), 1, split_k), cluster (1, 1, split_k). Cluster rank r
    owns K tiles [r * ceil(T / split_k), ...) of the T = K / 128 tiles, reduces
    output columns [r * shard, (r + 1) * shard) of the token tile through
    DSMEM, and stores them.
    """

    def __init__(
        self,
        cfg: SmallMConfig,
        use_pdl: bool,
        push: PushSpec | None = None,
        prep: PrepSpec | None = None,
    ):
        assert cfg.cta_n % 8 == 0 and 8 <= cfg.cta_n <= 48
        assert 1 <= cfg.split_k <= 8
        # Push mode pairs rows through lane ^ 4: the 16dp TMEM load layout of
        # token tiles 8 and 16.
        assert push is None or (use_pdl and cfg.cta_n <= min(16, push.capacity_m))
        # Prep mode splits the token tile 8's two token parities over cluster
        # ranks 0 and 1 (the stock owners of its two element halves).
        assert prep is None or (
            push is None and use_pdl and cfg.cta_n == 8 and cfg.split_k >= 2
        )
        assert prep is None or 0 < prep.rotary_dim <= CTA_M and prep.rotary_dim % 8 == 0
        self.cfg = cfg
        self.push = push
        self.prep = prep
        self.cta_n = cfg.cta_n
        self.split_k = cfg.split_k
        self.a_stages = cfg.a_stages
        self.x_stages = cfg.x_stages
        self.use_pdl = use_pdl
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.Float8E4M3FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.x_dtype = cutlass.BFloat16
        self.cta_group = tcgen05.CtaGroup.ONE
        self.shard = shard_size(self.cta_n, self.split_k)
        self.mailbox_elems = max((self.split_k - 1) * CTA_M * self.shard, 4)
        # Quantization work items: (row, 32-block, half) -> 16 elements each.
        self.quant_items = self.cta_n * (CTA_K // 16)
        self.quant_iters = -(-self.quant_items // QUANT_THREADS)

    def _setup(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (CTA_M, self.cta_n),
        )
        tiler = (CTA_M, self.cta_n, CTA_K)
        self.a_smem_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, tiler, self.ab_dtype, self.a_stages
        )
        self.b_smem_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, tiler, self.ab_dtype, self.x_stages
        )
        self.sfa_smem_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, tiler, SF_VEC, self.a_stages
        )
        # SFB spans a full 128-row scale atom regardless of the token tile.
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (CTA_M, 128),
        )
        self.sfb_smem_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, (CTA_M, 128, CTA_K), SF_VEC, self.x_stages
        )
        self.x_smem_staged = cute.make_layout(
            (self.cta_n, CTA_K, self.x_stages),
            stride=(CTA_K, 1, self.cta_n * CTA_K),
        )

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        n: Int32,
        m: Int32,
        k: Int32,
        stream: cuda.CUstream,
    ):
        self._launch(w_ptr, ws_ptr, x_ptr, out_ptr, n, m, k, Int64(0), Int64(0), stream)

    @cute.jit
    def _launch(
        self,
        w_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        n: Int32,
        m: Int32,
        k: Int32,
        push_mailbox: Int64,
        push_stage: Int64,
        stream: cuda.CUstream,
        prep_args: PrepArgs | None = None,
    ):
        self._setup()
        # Weight [N, K] (K-major), its swizzled 128x4 E8M0 scales, activation
        # [M, K] bf16, output [M, N] bf16 viewed as (N, M) with N contiguous.
        mA = cute.make_tensor(
            w_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        mSFA = cute.make_tensor(
            ws_ptr, blockscaled_utils.tile_atom_to_shape_SF(mA.shape, SF_VEC)
        )
        mX = cute.make_tensor(
            x_ptr, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        mC = cute.make_tensor(
            out_ptr, cute.make_ordered_layout((n, m, 1), order=(0, 1, 2))
        )
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(mC)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, self.split_k)), (self.tiled_mma.thr_id.shape,)
        )
        a_smem = cute.slice_(self.a_smem_staged, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id),
            mA,
            a_smem,
            (CTA_M, self.cta_n, CTA_K),
            self.tiled_mma,
            cluster_layout_vmnk.shape,
        )
        sfa_smem = cute.slice_(self.sfa_smem_staged, (None, None, None, 0))
        tma_atom_sfa, tma_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id),
            mSFA,
            sfa_smem,
            (CTA_M, self.cta_n, CTA_K),
            self.tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        x_smem = cute.slice_(self.x_smem_staged, (None, None, 0))
        tma_atom_x, tma_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            x_smem,
            (self.cta_n, CTA_K),
        )
        self.a_tx_bytes = cute.size_in_bytes(
            self.ab_dtype, a_smem
        ) + cute.size_in_bytes(self.sf_dtype, sfa_smem)
        self.x_tx_bytes = cute.size_in_bytes(self.x_dtype, x_smem)

        grid = (cute.ceil_div(n, CTA_M), 1, self.split_k)
        args = (
            self.tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_sfa,
            tma_sfa,
            tma_atom_x,
            tma_x,
            mC,
            self.a_smem_staged,
            self.b_smem_staged,
            self.sfa_smem_staged,
            self.sfb_smem_staged,
            self.x_smem_staged,
            push_mailbox,
            push_stage,
        )
        if cutlass.const_expr(prep_args is None):
            kernel = self.kernel(*args)
        else:
            kernel = self.kernel_prep(*args, *prep_args)
        kernel.launch(
            grid=grid,
            block=[NUM_THREADS, 1, 1],
            cluster=(1, 1, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        mC: cute.Tensor,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        x_smem_staged: cute.Layout,
        push_mailbox: Int64,
        push_stage: Int64,
    ):
        self._body(
            tiled_mma,
            tma_atom_a,
            mA,
            tma_atom_sfa,
            mSFA,
            tma_atom_x,
            mX,
            mC,
            a_smem_staged,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            x_smem_staged,
            push_mailbox,
            push_stage,
            None,
        )

    @cute.jit
    def _body(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        mC: cute.Tensor,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        x_smem_staged: cute.Layout,
        push_mailbox: Int64,
        push_stage: Int64,
        prep_args: PrepArgs | None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, _, _ = cute.arch.block_idx()

        if warp_idx == TMA_A_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
        if warp_idx == TMA_X_WARP:
            cpasync.prefetch_descriptor(tma_atom_x)

        # This CTA's K tiles.
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        total_k_tiles = cute.ceil_div(cute.size(mA, mode=[1]), CTA_K)
        per_split = cute.ceil_div(total_k_tiles, self.split_k)
        k_start = split_rank * per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        k_tile_count = k_end - k_start

        A_ST = self.a_stages
        X_ST = self.x_stages
        MAILBOX = self.mailbox_elems

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, A_ST]
            a_empty: cute.struct.MemRange[cutlass.Int64, A_ST]
            x_full: cute.struct.MemRange[cutlass.Int64, X_ST]
            b_full: cute.struct.MemRange[cutlass.Int64, X_ST]
            x_empty: cute.struct.MemRange[cutlass.Int64, X_ST]
            acc_full: cutlass.Int64
            tmem_bar: cutlass.Int64
            mailbox_bar: cutlass.Int64
            tmem_base: cutlass.Int32
            mailbox: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, MAILBOX], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        x_full = storage.x_full.data_ptr()
        b_full = storage.b_full.data_ptr()
        x_empty = storage.x_empty.data_ptr()
        acc_full = storage.acc_full.ptr
        tmem_bar = storage.tmem_bar.ptr
        mailbox_bar = storage.mailbox_bar.ptr
        tmem_base = storage.tmem_base.ptr
        smem_mailbox = storage.mailbox.data_ptr()

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(A_ST):
                    cute.arch.mbarrier_init(a_full + s, 1)
                    cute.arch.mbarrier_init(a_empty + s, 1)
                for s in cutlass.range_constexpr(X_ST):
                    cute.arch.mbarrier_init(x_full + s, 1)
                    cute.arch.mbarrier_init(b_full + s, QUANT_WARPS)
                    cute.arch.mbarrier_init(x_empty + s, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                # MMA warp (alloc / dealloc) + the epilogue threads.
                cute.arch.mbarrier_init(tmem_bar, 32 + EPI_THREADS)
                cute.arch.mbarrier_init(mailbox_bar, 1)
        cute.arch.mbarrier_init_fence()
        # Only the epilogue touches peer CTAs (its DSMEM mailbox stores), so
        # only it waits on the cluster barrier; everyone else starts after a
        # CTA-local sync.
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        cute.arch.barrier()

        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfa_smem_staged, byte_alignment=128
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfb_smem_staged, byte_alignment=128
        )
        sX = smem.allocate_tensor(
            element_type=self.x_dtype, layout=x_smem_staged, byte_alignment=128
        )

        if cutlass.const_expr(self.prep is not None):
            # The head as bf16 [4 tokens][128 dims] of this rank's parity.
            sHead = smem.allocate_tensor(
                element_type=cutlass.Uint16,
                layout=cute.make_layout(4 * CTA_M),
                byte_alignment=16,
            )

        if warp_idx < QUANT_WARPS:
            push_generation = Uint32(0)
            if cutlass.const_expr(self.push is not None):  # noqa: SIM102
                if warp_idx < EPI_THREADS // 32:
                    # The generation the previous consumer set (complete once
                    # the PDL wait returns), read before the consumer may
                    # launch and advance it.
                    cute.arch.griddepcontrol_wait()
                    push_generation = _ld_volatile_u32(push_stage + 4 * PUSH_STAGE_WORD)
                    cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.prep is None):
                self.quant_warps(x_full, b_full, sX, sB, sSFB, tidx, k_tile_count)
                if warp_idx < EPI_THREADS // 32:
                    self.epilogue(
                        tiled_mma,
                        mC,
                        acc_full,
                        tmem_bar,
                        tmem_base,
                        mailbox_bar,
                        smem_mailbox,
                        tidx,
                        m_tile,
                        split_rank,
                        k_tile_count,
                        push_mailbox,
                        push_generation,
                    )
            else:
                side = self.prep_load(prep_args, mC, m_tile, split_rank)
                self.quant_warps(x_full, b_full, sX, sB, sSFB, tidx, k_tile_count)
                if warp_idx < EPI_THREADS // 32:
                    self.prep_epilogue(
                        tiled_mma,
                        mC,
                        acc_full,
                        tmem_bar,
                        tmem_base,
                        mailbox_bar,
                        smem_mailbox,
                        tidx,
                        m_tile,
                        split_rank,
                        k_tile_count,
                        prep_args,
                        side,
                        sHead,
                    )
        elif warp_idx == TMA_A_WARP:
            self.tma_a_warp(
                tiled_mma,
                a_full,
                a_empty,
                tma_atom_a,
                mA,
                sA,
                tma_atom_sfa,
                mSFA,
                sSFA,
                m_tile,
                k_start,
                k_tile_count,
            )
        elif warp_idx == MMA_WARP:
            self.mma_warp(
                tiled_mma,
                a_full,
                a_empty,
                b_full,
                x_empty,
                acc_full,
                tmem_bar,
                tmem_base,
                sA,
                sB,
                sSFA,
                sSFB,
                sfa_smem_staged,
                sfb_smem_staged,
                k_tile_count,
            )
        elif warp_idx == TMA_X_WARP:
            self.tma_x_warp(x_full, x_empty, tma_atom_x, mX, sX, k_start, k_tile_count)

        cute.arch.barrier()

    @cute.jit
    def tma_a_warp(
        self,
        tiled_mma: cute.TiledMma,
        a_full,
        a_empty,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        sA: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        sSFA: cute.Tensor,
        m_tile: Int32,
        k_start: Int32,
        k_tile_count: Int32,
    ):
        thr_mma = tiled_mma.get_slice(0)
        gA = cute.local_tile(mA, (CTA_M, CTA_K), (None, None, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tAgA = tAgA[(None, m_tile, None, 0)]
        gSFA = cute.local_tile(mSFA, (CTA_M, CTA_K), (None, None, None))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA), 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tAgSFA = tAgSFA[(None, m_tile, None, 0)]

        state = make_pipeline_state(PipelineUserType.Producer, self.a_stages)
        for kt in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(a_empty + state.index, state.phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    a_full + state.index, self.a_tx_bytes
                )
            cute.copy(
                tma_atom_a,
                tAgA[(None, kt + k_start)],
                tAsA[(None, state.index)],
                tma_bar_ptr=a_full + state.index,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, kt + k_start)],
                tAsSFA[(None, state.index)],
                tma_bar_ptr=a_full + state.index,
            )
            state.advance()
        # Prep mode triggers at the end of its epilogue instead (prep_epilogue).
        if cutlass.const_expr(self.use_pdl and self.push is None and self.prep is None):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def tma_x_warp(
        self,
        x_full,
        x_empty,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        sX: cute.Tensor,
        k_start: Int32,
        k_tile_count: Int32,
    ):
        gX = cute.local_tile(mX, (self.cta_n, CTA_K), (None, None, None))
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        tXgX = tXgX[(None, 0, None, 0)]
        # The activation is the previous kernel's output.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        state = make_pipeline_state(PipelineUserType.Producer, self.x_stages)
        for kt in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(x_empty + state.index, state.phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    x_full + state.index, self.x_tx_bytes
                )
            cute.copy(
                tma_atom_x,
                tXgX[(None, kt + k_start)],
                tXsX[(None, state.index)],
                tma_bar_ptr=x_full + state.index,
            )
            state.advance()

    @cute.jit
    def quant_warps(
        self,
        x_full,
        b_full,
        sX: cute.Tensor,
        sB: cute.Tensor,
        sSFB: cute.Tensor,
        tidx: Int32,
        k_tile_count: Int32,
    ):
        """bf16 activation tile -> MXFP8 B tile and its E8M0 scales, in the
        operand layouts the MMA reads (SW128 K-major B, 128x4 scale atom).

        Each item is 16 consecutive elements of one row; the 2 threads of a
        32-block share its amax through one butterfly shuffle. All items' loads
        issue before any conversion so the items of one K tile overlap.
        """
        x_base = sX.iterator.toint()
        b_base = sB.iterator.toint()
        sf_base = sSFB.iterator.toint()
        x_stage_bytes = self.cta_n * CTA_K * 2
        b_stage_bytes = self.cta_n * CTA_K
        lane = cute.arch.lane_idx()
        inv_e4m3_max = Float32(_INV_E4M3_MAX)
        total = self.quant_items

        state = make_pipeline_state(PipelineUserType.Consumer, self.x_stages)
        for _kt in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(x_full + state.index, state.phase)
            xs = x_base + state.index * x_stage_bytes
            bs = b_base + state.index * b_stage_bytes
            ss = sf_base + state.index * SF_ATOM_BYTES
            rows = []
            cols = []
            vals = []
            for it in cutlass.range_constexpr(self.quant_iters):
                item = tidx + it * QUANT_THREADS
                if cutlass.const_expr((it + 1) * QUANT_THREADS > total):  # noqa: SIM102
                    # Partial last pass: idle warps (whole warps, since total is
                    # a multiple of 64) re-read a valid item and skip stores.
                    if item >= total:
                        item = total - 1
                row = item >> 3
                col = ((item >> 1) & 3) * 32 + (item & 1) * 16
                addr = xs + (row * CTA_K + col) * 2
                rows.append(row)
                cols.append(col)
                vals.append(_ld_shared_v4(addr) + _ld_shared_v4(addr + 16))
            for it in cutlass.range_constexpr(self.quant_iters):
                row = rows[it]
                col = cols[it]
                v = vals[it]
                amax = _bf16x16_absmax(v)
                amax = _fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
                e = _float_to_ue8m0(amax * inv_e4m3_max)
                inv2 = _ue8m0_to_inv_scale_bf16x2(e)
                w0, w1 = _bf16x8_to_e4m3x8(v[0], v[1], v[2], v[3], inv2)
                w2, w3 = _bf16x8_to_e4m3x8(v[4], v[5], v[6], v[7], inv2)
                # SW128: XOR address bits [7, 10) into [4, 7).
                b_addr = bs + row * CTA_K + col
                b_addr = b_addr ^ (((b_addr >> 7) & 7) << 4)
                sf_addr = ss + (row & 31) * 16 + (row >> 5) * 4 + (col >> 5)
                if cutlass.const_expr((it + 1) * QUANT_THREADS <= total):
                    _st_shared_v4(b_addr, w0, w1, w2, w3)
                    if (tidx & 1) == 0:
                        _st_shared_u8(sf_addr, e)
                else:
                    if tidx + it * QUANT_THREADS < total:
                        _st_shared_v4(b_addr, w0, w1, w2, w3)
                        if (tidx & 1) == 0:
                            _st_shared_u8(sf_addr, e)
            # Make the generic-proxy stores visible to the tensor core.
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(b_full + state.index)
            state.advance()

    def _s2t(self, sSF: cute.Tensor, tSF: cute.Tensor):
        """SMEM -> TMEM scale-factor copy, partitioned for one thread."""
        tCsSF = cute.filter_zeros(sSF)
        tCtSF = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled = tcgen05.make_s2t_copy(atom, tCtSF)
        thr = tiled.get_slice(0)
        src = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(tCsSF))
        dst = thr.partition_D(tCtSF)
        return tiled, src, dst

    @cute.jit
    def mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        a_full,
        a_empty,
        b_full,
        x_empty,
        acc_full,
        tmem_bar,
        tmem_base,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        k_tile_count: Int32,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((CTA_M, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        cute.arch.alloc_tmem(TMEM_COLS, tmem_base)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # TMEM columns: accumulator [0, cta_n), SFA 16, SFB 16.
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n, dtype=self.sf_dtype)
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                tiled_mma.shape_mnk,
                SF_VEC,
                cute.slice_(sfa_smem_staged, (None, None, None, 0)),
            ),
        )
        sfb_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n + 16, dtype=self.sf_dtype)
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                tiled_mma.shape_mnk,
                SF_VEC,
                cute.slice_(sfb_smem_staged, (None, None, None, 0)),
            ),
        )
        s2t_sfa, sfa_src, sfa_dst = self._s2t(sSFA, tCtSFA)
        s2t_sfb, sfb_src, sfb_dst = self._s2t(sSFB, tCtSFB)

        a_state = make_pipeline_state(PipelineUserType.Consumer, self.a_stages)
        b_state = make_pipeline_state(PipelineUserType.Consumer, self.x_stages)
        for kt in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(a_full + a_state.index, a_state.phase)
            cute.arch.mbarrier_wait(b_full + b_state.index, b_state.phase)
            cute.copy(
                s2t_sfa, sfa_src[(None, None, None, None, a_state.index)], sfa_dst
            )
            cute.copy(
                s2t_sfb, sfb_src[(None, None, None, None, b_state.index)], sfb_dst
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kt != 0)
            cute.gemm(
                tiled_mma,
                tCtAcc,
                [tCrA[(None, None, None, a_state.index)], tCtSFA],
                [tCrB[(None, None, None, b_state.index)], tCtSFB],
                tCtAcc,
            )
            with cute.arch.elect_one():
                tcgen05.commit(a_empty + a_state.index, None, self.cta_group)
                tcgen05.commit(x_empty + b_state.index, None, self.cta_group)
            a_state.advance()
            b_state.advance()

        with cute.arch.elect_one():
            tcgen05.commit(acc_full, None, self.cta_group)

        # The epilogue threads join the allocation phase only after quantizing,
        # so close that phase before arriving for the release phase.
        cute.arch.mbarrier_wait(tmem_bar, 0)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        mC: cute.Tensor,
        acc_full,
        tmem_bar,
        tmem_base,
        mailbox_bar,
        mailbox,
        tidx: Int32,
        m_tile: Int32,
        split_rank: Int32,
        k_tile_count: Int32,
        push_mailbox: Int64,
        push_generation: Uint32,
    ):
        thr_mma = tiled_mma.get_slice(0)
        gC = cute.local_tile(mC, (CTA_M, self.cta_n), (None, None, None))
        tCgC = thr_mma.partition_C(gC)
        acc_shape = tiled_mma.partition_shape_C((CTA_M, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        t2r_atom = sm100_utils.get_tmem_load_op(
            (CTA_M, self.cta_n, CTA_K),
            self.c_layout,
            mC.element_type,
            self.acc_dtype,
            (CTA_M, self.cta_n),
            False,
        )
        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc[((None, None), 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tDtAcc = thr_t2r.partition_S(tCtAcc[((None, None), 0, 0)])
        tDgC = thr_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])
        frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        tDrAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
        tDrC = cute.make_rmem_tensor(frag_shape, mC.element_type)
        tDgC = tDgC[(None, None, None, m_tile, 0, 0)]

        cC = cute.local_tile(
            cute.make_identity_tensor(mC.shape), (CTA_M, self.cta_n), (None, None, None)
        )
        tDcC = thr_t2r.partition_D(
            thr_mma.partition_C(cC)[((None, None), 0, 0, None, None, None)]
        )
        tDcC = tDcC[(None, None, None, m_tile, 0, 0)]
        tDpC = cute.make_rmem_tensor(tDcC.shape, cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(tDpC)):
            tDpC[i] = cute.elem_less(tDcC[i], mC.shape)

        cute.arch.mbarrier_wait(acc_full, 0)
        cute.copy(tiled_t2r, tDtAcc, tDrAcc)
        cute.arch.fence_view_async_tmem_load()
        if k_tile_count == 0:
            tDrAcc.fill(0.0)
        cute.arch.mbarrier_arrive(tmem_bar)

        if cutlass.const_expr(self.split_k > 1):
            shard = self.shard
            n_elems = cute.size(tDrAcc)
            mbox = mailbox.toint()
            mbar = mailbox_bar.toint()
            # Rank r reduces and stores elements [r * shard, (r + 1) * shard).
            # Mailbox: [sender slot][thread][shard] fp32, one 16-byte store per
            # 4 elements.
            my_count = n_elems - split_rank * shard
            if my_count > shard:
                my_count = shard
            if my_count < 0:
                my_count = 0
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_bar, (self.split_k - 1) * CTA_M * 4 * my_count
                )
            # Peers' mailbox barriers are initialized.
            cute.arch.cluster_wait()
            for peer in cutlass.range_constexpr(self.split_k):
                peer_count = min(max(n_elems - peer * shard, 0), shard)
                # Compile-time shard size, then the runtime rank test.
                if cutlass.const_expr(peer_count > 0):  # noqa: SIM102
                    if split_rank != peer:
                        slot = split_rank
                        if split_rank > peer:
                            slot = split_rank - 1
                        base = mbox + ((slot * CTA_M + tidx) * shard) * 4
                        for q in cutlass.range_constexpr(peer_count // 4):
                            j = peer * shard + 4 * q
                            _st_async_v4(
                                base + 16 * q,
                                tDrAcc[j],
                                tDrAcc[j + 1],
                                tDrAcc[j + 2],
                                tDrAcc[j + 3],
                                mbar,
                                Int32(peer),
                            )
            cute.arch.mbarrier_wait(mailbox_bar, 0)
            for r in cutlass.range_constexpr(self.split_k):
                r_count = min(max(n_elems - r * shard, 0), shard)
                if cutlass.const_expr(r_count > 0):  # noqa: SIM102
                    if split_rank == r:
                        for s in cutlass.range_constexpr(self.split_k - 1):
                            base = mbox + ((s * CTA_M + tidx) * shard) * 4
                            for q in cutlass.range_constexpr(r_count // 4):
                                p = _ld_shared_v4_f32(base + 16 * q)
                                for i in cutlass.range_constexpr(4):
                                    j = r * shard + 4 * q + i
                                    tDrAcc[j] = tDrAcc[j] + p[i]
            for i in cutlass.range_constexpr(n_elems):
                if split_rank != i // shard:
                    tDpC[i] = cutlass.Boolean(False)

        if cutlass.const_expr(self.push is None):
            tDrC.store(tDrAcc.load().to(mC.element_type))
            cute.basic_copy_if(tDpC, tDrC, tDgC)
        else:
            # Rows r (lane bit 2 clear) and r + 1 (lane ^ 4) of the same token
            # pair: the even-row lane stores both as one bf16x2 word, at the
            # (row, token) coordinate of its element.
            spec = self.push
            n = cute.size(mC, mode=[0])
            slot = (
                push_mailbox
                + Int64(
                    (Int64(push_generation) * spec.tp + spec.rank) * spec.capacity_m
                )
                * Int64(n)
                * 2
            )
            even_row = (cute.arch.lane_idx() & 4) == 0
            for i in cutlass.range_constexpr(cute.size(tDrAcc)):
                bits = _bf16_bits(tDrAcc[i])
                upper = _shfl_xor_u32(bits, 4)
                if tDpC[i] and even_row:
                    coord = tDcC[i]
                    _st_multicast_u32(
                        slot + (Int64(coord[1]) * Int64(n) + Int64(coord[0])) * 2,
                        bits | (upper << Uint32(16)),
                    )

    @cute.jit
    def prep_load(
        self, prep: PrepArgs, mC: cute.Tensor, m_tile: Int32, split_rank: Int32
    ):
        """What epilogue lane l of warp w needs besides the head: dims
        [4l, 4l + 4)'s norm weights and RoPE cos / sin at the position of token
        2w + rank, and that token's cache slots. Positions and slots are step
        inputs, written before this grid's predecessor ran (the score kernels
        read seq_lens the same way), so they load ahead of the PDL wait and
        the mainloop hides the latency."""
        spec = self.prep
        nq, nkv, niq = spec.num_heads, spec.num_kv_heads, spec.num_index_heads
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) & 3
        lane = cute.arch.lane_idx()
        tok = 2 * warp + (split_rank & 1)
        m = cute.size(mC, mode=[1])
        if tok > m - 1:
            tok = m - 1
        tok64 = Int64(tok)
        dim0 = 4 * lane
        pos = _ld_global_nc_s64(prep.positions + tok64 * 8)
        # Lanes past rotary_dim load in-row values they never use.
        half = spec.rotary_dim // 2
        cs_row = prep.cos_sin + pos * (2 * spec.rotary_dim)
        cs_col = dim0 % half
        cos01, cos23 = _ld_global_nc_v2_u32(cs_row + Int64(2 * cs_col))
        sin01, sin23 = _ld_global_nc_v2_u32(cs_row + Int64(2 * (cs_col + half)))
        norm = prep.q_norm
        if m_tile >= nq:
            norm = prep.k_norm
        slot = Int64(-1)
        index_slot = Int64(-1)
        if cutlass.const_expr(niq > 0):
            if m_tile >= nq + 2 * nkv:
                norm = prep.index_q_norm
            if m_tile >= nq + 2 * nkv + niq:
                norm = prep.index_k_norm
            slot = _ld_global_nc_s64(prep.slot_mapping + tok64 * 8)
            index_slot = _ld_global_nc_s64(prep.index_slot_mapping + tok64 * 8)
        w01, w23 = _ld_global_nc_v2_u32(norm + Int64(2 * dim0))
        return cos01, cos23, sin01, sin23, w01, w23, slot, index_slot

    @cute.jit
    def prep_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        mC: cute.Tensor,
        acc_full,
        tmem_bar,
        tmem_base,
        mailbox_bar,
        mailbox,
        tidx: Int32,
        m_tile: Int32,
        split_rank: Int32,
        k_tile_count: Int32,
        prep: PrepArgs,
        side,
        sHead: cute.Tensor,
    ):
        """Split-K reduce by token parity, then ``prep_head``. The 16dp TMEM
        load of the 8-token tile gives thread t rows t // 4 + 8k of tokens
        2 (t % 4) (element 2k) and 2 (t % 4) + 1 (element 2k + 1), so rank r
        reducing elements 2k + r holds all 128 rows of tokens 2j + r. The
        stock split gives ranks 0 and 1 elements 0-3 and 4-7, and an element's
        sum is its owner's partial plus the others in rank order: owners 0 and
        1 both give (p0 + p1) + p2 + ..., so the sums are unchanged."""
        thr_mma = tiled_mma.get_slice(0)
        gC = cute.local_tile(mC, (CTA_M, self.cta_n), (None, None, None))
        tCgC = thr_mma.partition_C(gC)
        acc_shape = tiled_mma.partition_shape_C((CTA_M, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        t2r_atom = sm100_utils.get_tmem_load_op(
            (CTA_M, self.cta_n, CTA_K),
            self.c_layout,
            mC.element_type,
            self.acc_dtype,
            (CTA_M, self.cta_n),
            False,
        )
        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc[((None, None), 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tDtAcc = thr_t2r.partition_S(tCtAcc[((None, None), 0, 0)])
        tDgC = thr_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])
        tDrAcc = cute.make_rmem_tensor(
            tDgC[(None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        assert cute.size(tDrAcc) == 8

        cute.arch.mbarrier_wait(acc_full, 0)
        cute.copy(tiled_t2r, tDtAcc, tDrAcc)
        cute.arch.fence_view_async_tmem_load()
        if k_tile_count == 0:
            tDrAcc.fill(0.0)
        cute.arch.mbarrier_arrive(tmem_bar)

        mbox = mailbox.toint()
        mbar = mailbox_bar.toint()
        count = Int32(0)
        if split_rank < 2:
            count = Int32(4)
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                mailbox_bar, (self.split_k - 1) * CTA_M * 4 * count
            )
        # Peers' mailbox barriers are initialized.
        cute.arch.cluster_wait()
        for peer in cutlass.range_constexpr(2):
            if split_rank != peer:
                slot = split_rank
                if split_rank > peer:
                    slot = split_rank - 1
                _st_async_v4(
                    mbox + ((slot * CTA_M + tidx) * 4) * 4,
                    tDrAcc[peer],
                    tDrAcc[peer + 2],
                    tDrAcc[peer + 4],
                    tDrAcc[peer + 6],
                    mbar,
                    Int32(peer),
                )
        cute.arch.mbarrier_wait(mailbox_bar, 0)
        warp = tidx // 32
        lane = tidx % 32
        head = sHead.iterator.toint()
        for r in cutlass.range_constexpr(2):
            if split_rank == r:
                for s in cutlass.range_constexpr(self.split_k - 1):
                    p = _ld_shared_v4_f32(mbox + ((s * CTA_M + tidx) * 4) * 4)
                    for i in cutlass.range_constexpr(4):
                        tDrAcc[2 * i + r] = tDrAcc[2 * i + r] + p[i]
                # The stock bf16 output rounding, transposed to [token][dim].
                for i in cutlass.range_constexpr(4):
                    row = warp * 32 + lane // 4 + 8 * i
                    _st_shared_u16(
                        head + ((lane % 4) * CTA_M + row) * 2,
                        _f32_to_bf16_u16(tDrAcc[2 * i + r]),
                    )
        if split_rank < 2:
            cute.arch.barrier(barrier_id=PREP_BARRIER, number_of_threads=EPI_THREADS)
            self.prep_head(prep, side, mC, head, warp, lane, m_tile, split_rank)
            # The dependent (the index decode score) launches once every CTA
            # has stored its outputs or exited, as it did after the prep
            # kernel's own end-of-kernel trigger: launched while the mainloop
            # still holds the SMs, its CTAs would land unevenly.
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def prep_head(
        self,
        prep: PrepArgs,
        side,
        mC: cute.Tensor,
        head: Int32,
        warp: Int32,
        lane: Int32,
        m_tile: Int32,
        split_rank: Int32,
    ):
        """One warp of fused_minimax_m3_qknorm_rope_kv_insert: warp w does
        token 2w + rank of this CTA's head, lane l its dims [4l, 4l + 4)."""
        spec = self.prep
        nq, nkv, niq = spec.num_heads, spec.num_kv_heads, spec.num_index_heads
        cos01, cos23, sin01, sin23, w01, w23, slot, index_slot = side
        tok = 2 * warp + split_rank
        if tok < cute.size(mC, mode=[1]):
            h01, h23 = _ld_shared_v2_u32(head + (warp * CTA_M + 4 * lane) * 2)
            e0, e1 = _bf16x2_to_f32(h01)
            e2, e3 = _bf16x2_to_f32(h23)
            if (m_tile < nq + nkv) | (m_tile >= nq + 2 * nkv):
                # Gemma RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w).
                ss = _fma_rn(e0, e0, Float32(0.0))
                ss = _fma_rn(e1, e1, ss)
                ss = _fma_rn(e2, e2, ss)
                ss = _fma_rn(e3, e3, ss)
                for mask in cutlass.range_constexpr(5):
                    ss = _add_rn(ss, cute.arch.shuffle_sync_bfly(ss, offset=16 >> mask))
                rcp = _rsqrt_approx(_fma_rn(ss, Float32(_INV_HEAD_DIM), prep.eps))
                w0, w1 = _bf16x2_to_f32(w01)
                w2, w3 = _bf16x2_to_f32(w23)
                one = Float32(1.0)
                e0 = _mul_rn(_add_rn(w0, one), _mul_rn(e0, rcp))
                e1 = _mul_rn(_add_rn(w1, one), _mul_rn(e1, rcp))
                e2 = _mul_rn(_add_rn(w2, one), _mul_rn(e2, rcp))
                e3 = _mul_rn(_add_rn(w3, one), _mul_rn(e3, rcp))
                # Partial NeoX RoPE: dim i pairs with i + half, a lane with
                # lane ^ (half / 4).
                half = spec.rotary_dim // 2
                p0 = cute.arch.shuffle_sync_bfly(e0, offset=half // 4)
                p1 = cute.arch.shuffle_sync_bfly(e1, offset=half // 4)
                p2 = cute.arch.shuffle_sync_bfly(e2, offset=half // 4)
                p3 = cute.arch.shuffle_sync_bfly(e3, offset=half // 4)
                if 4 * lane < spec.rotary_dim:
                    c0, c1 = _bf16x2_to_f32(cos01)
                    c2, c3 = _bf16x2_to_f32(cos23)
                    s0, s1 = _bf16x2_to_f32(sin01)
                    s2, s3 = _bf16x2_to_f32(sin23)
                    if 4 * lane < half:
                        e0 = _fma_neg_rn(e0, c0, p0, s0)
                        e1 = _fma_neg_rn(e1, c1, p1, s1)
                        e2 = _fma_neg_rn(e2, c2, p2, s2)
                        e3 = _fma_neg_rn(e3, c3, p3, s3)
                    else:
                        e0 = _fma_rn(e0, c0, _mul_rn(p0, s0))
                        e1 = _fma_rn(e1, c1, _mul_rn(p1, s1))
                        e2 = _fma_rn(e2, c2, _mul_rn(p2, s2))
                        e3 = _fma_rn(e3, c3, _mul_rn(p3, s3))
            dim0 = Int64(4 * lane)
            tok64 = Int64(tok)
            if cutlass.const_expr(niq == 0):
                # Dense layer: the normed and roped q / k (and v) row in place.
                off = tok64 * (spec.row_heads * CTA_M) + Int64(m_tile * CTA_M) + dim0
                _st_global_v2_u32(
                    mC.iterator.toint() + off * 2,
                    _f32x2_to_bf16x2(e0, e1),
                    _f32x2_to_bf16x2(e2, e3),
                )
            else:
                if m_tile < nq:
                    lo = _f32x2_to_bf16x2(e0, e1)
                    hi = _f32x2_to_bf16x2(e2, e3)
                    off = (tok64 * nq + Int64(m_tile)) * CTA_M + dim0
                    _st_global_v2_u32(prep.q_out + off * 2, lo, hi)
                    if cutlass.const_expr(spec.q_fp8):
                        # scaled_fp8_quant of the bf16 q.
                        r0, r1 = _bf16x2_to_f32(lo)
                        r2, r3 = _bf16x2_to_f32(hi)
                        inv = prep.q_inv_scale
                        _st_global_u32(
                            prep.q_fp8_out + off,
                            _f32x4_to_e4m3x4(
                                _mul_rn(r0, inv),
                                _mul_rn(r1, inv),
                                _mul_rn(r2, inv),
                                _mul_rn(r3, inv),
                                True,
                            ),
                        )
                elif m_tile < nq + 2 * nkv:
                    # K or V into the e4m3 K/V cache; padded tokens (slot < 0)
                    # are skipped.
                    if slot >= 0:
                        kv_head = m_tile - nq
                        kv_col = dim0
                        if m_tile >= nq + nkv:
                            kv_head = m_tile - nq - nkv
                            kv_col = dim0 + CTA_M
                        block = slot // prep.block_size
                        off = (
                            block * prep.kv_stride_block
                            + Int64(kv_head) * prep.kv_stride_head
                            + (slot - block * prep.block_size) * prep.kv_stride_token
                            + kv_col
                        )
                        _st_global_u32(
                            prep.kv_cache + off, _f32x4_to_e4m3x4(e0, e1, e2, e3, False)
                        )
                elif m_tile < nq + 2 * nkv + niq:
                    off = (tok64 * niq + Int64(m_tile - nq - 2 * nkv)) * CTA_M + dim0
                    _st_global_u32(
                        prep.index_q_out + off, _f32x4_to_e4m3x4(e0, e1, e2, e3, True)
                    )
                else:
                    if index_slot >= 0:
                        _st_global_u32(
                            prep.index_cache + index_slot * CTA_M + dim0,
                            _f32x4_to_e4m3x4(e0, e1, e2, e3, True),
                        )


class SmallMMxfp8GemmPush(SmallMMxfp8Gemm):
    """Push-mode entry point: the output goes to the multicast mailbox at
    ``push_mailbox`` in the generation held at ``push_stage`` (int32 words);
    ``out_ptr`` only shapes the output view and is never written."""

    def __init__(self, cfg: SmallMConfig, push: PushSpec):
        super().__init__(cfg, True, push)

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        n: Int32,
        m: Int32,
        k: Int32,
        push_mailbox: Int64,
        push_stage: Int64,
        stream: cuda.CUstream,
    ):
        self._launch(
            w_ptr, ws_ptr, x_ptr, out_ptr, n, m, k, push_mailbox, push_stage, stream
        )


class SmallMMxfp8GemmPrep(SmallMMxfp8Gemm):
    """Prep-mode entry point (``PrepSpec``): the epilogue writes the attention
    prep outputs. ``out_ptr`` is a dense layer's [M, N] qkv output; a sparse
    layer's is never written and only shapes the output view."""

    def __init__(self, cfg: SmallMConfig, prep: PrepSpec):
        super().__init__(cfg, True, prep=prep)

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        n: Int32,
        m: Int32,
        k: Int32,
        q_out: Int64,
        q_fp8_out: Int64,
        index_q_out: Int64,
        q_norm: Int64,
        k_norm: Int64,
        index_q_norm: Int64,
        index_k_norm: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        index_slot_mapping: Int64,
        kv_cache: Int64,
        index_cache: Int64,
        kv_stride_block: Int64,
        kv_stride_head: Int64,
        kv_stride_token: Int64,
        block_size: Int64,
        eps: Float32,
        q_inv_scale: Float32,
        stream: cuda.CUstream,
    ):
        self._launch(
            w_ptr,
            ws_ptr,
            x_ptr,
            out_ptr,
            n,
            m,
            k,
            Int64(0),
            Int64(0),
            stream,
            PrepArgs(
                q_out,
                q_fp8_out,
                index_q_out,
                q_norm,
                k_norm,
                index_q_norm,
                index_k_norm,
                cos_sin,
                positions,
                slot_mapping,
                index_slot_mapping,
                kv_cache,
                index_cache,
                kv_stride_block,
                kv_stride_head,
                kv_stride_token,
                block_size,
                eps,
                q_inv_scale,
            ),
        )

    @cute.kernel
    def kernel_prep(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        mC: cute.Tensor,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        x_smem_staged: cute.Layout,
        push_mailbox: Int64,
        push_stage: Int64,
        q_out: Int64,
        q_fp8_out: Int64,
        index_q_out: Int64,
        q_norm: Int64,
        k_norm: Int64,
        index_q_norm: Int64,
        index_k_norm: Int64,
        cos_sin: Int64,
        positions: Int64,
        slot_mapping: Int64,
        index_slot_mapping: Int64,
        kv_cache: Int64,
        index_cache: Int64,
        kv_stride_block: Int64,
        kv_stride_head: Int64,
        kv_stride_token: Int64,
        block_size: Int64,
        eps: Float32,
        q_inv_scale: Float32,
    ):
        self._body(
            tiled_mma,
            tma_atom_a,
            mA,
            tma_atom_sfa,
            mSFA,
            tma_atom_x,
            mX,
            mC,
            a_smem_staged,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
            x_smem_staged,
            push_mailbox,
            push_stage,
            PrepArgs(
                q_out,
                q_fp8_out,
                index_q_out,
                q_norm,
                k_norm,
                index_q_norm,
                index_k_norm,
                cos_sin,
                positions,
                slot_mapping,
                index_slot_mapping,
                kv_cache,
                index_cache,
                kv_stride_block,
                kv_stride_head,
                kv_stride_token,
                block_size,
                eps,
                q_inv_scale,
            ),
        )
