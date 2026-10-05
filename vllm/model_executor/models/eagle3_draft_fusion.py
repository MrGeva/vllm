# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode-sized forward of the one-layer EAGLE3 Llama draft.

At small M (the draft decode passes and the K+1-token draft prefill) the draft
layer is a chain of ~20 short kernels around four bf16 GEMMs. This path keeps
the operand precision (bf16 weights and activations, fp32 accumulation) and
the stock qkv/o/down/fc GEMMs, and removes the kernel boundaries around them:

* ``ar_norm``: the embedding, o_proj and down_proj all-reduces carry the
  following residual add + RMSNorm (FlashInfer's one-shot AR + RMSNorm, the
  kernel the target model already uses at these token counts);
* ``cat_norm``: hidden_norm writes its half of the qkv_proj input directly
  (one kernel instead of rms_norm + cat; at M = 1 the embedding norm writes
  the other half in place);
* ``gate_up``: gate_up_proj is a PDL GEMV with SiLU-and-mul in its epilogue
  (no separate activation kernel), pulling its first weight bytes into L2
  before waiting on the predecessor;
* ``rope_kv``: RoPE, the fp8 query quantization and the KV-cache insert are
  one kernel;
* ``fc``: the Eagle3 aux-hidden ``fc_norm`` (three chunk RMSNorms and two
  cats) is one kernel ahead of the stock fc GEMM;
* ``steps``: the speculator's fused decode loop sets the draft step once and
  update_draft_inputs advances it, instead of a fill kernel before each pass;
  the draft prefill's gathers and copies run as one kernel;
* ``seg_gemv``: projections whose shape is in ``_SEG_CONFIGS`` run as a PDL
  GEMV that streams several K segments per iteration instead of cuBLAS.

``VLLM_EAGLE3_DRAFT_FUSION``: unset or "1" enables every part, "0" disables
the path, or a comma list (e.g. "ar_norm,rope_kv") enables only those parts.
Any unsupported configuration falls back to the unfused model code.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton

logger = init_logger(__name__)

_PARTS = ("ar_norm", "cat_norm", "gate_up", "rope_kv", "fc", "steps", "seg_gemv")
# Largest token count that takes the fused path (decode-sized batches).
MAX_M = int(os.environ.get("VLLM_EAGLE3_DRAFT_FUSION_MAX_M", "16"))
_BLOCK_M = 16


def enabled_parts() -> frozenset[str]:
    raw = os.environ.get("VLLM_EAGLE3_DRAFT_FUSION", "1").strip().lower()
    if raw in ("", "1", "all", "true"):
        return frozenset(_PARTS)
    if raw in ("0", "none", "false"):
        return frozenset()
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    unknown = parts - set(_PARTS)
    if unknown:
        raise ValueError(f"VLLM_EAGLE3_DRAFT_FUSION: unknown parts {sorted(unknown)}")
    return parts


# --------------------------------------------------------------------------
# gate_up GEMV with SiLU-and-mul epilogue: out[M, I] = silu(x @ Wg^T) * (x @ Wu^T)
# for W = [Wg; Wu] ([2I, K]), bf16 in, fp32 accumulate, bf16 out.
# --------------------------------------------------------------------------


@triton.jit
def _l2_prefetch(w_ptr, rows, rmask, stride_w, NBYTES: tl.constexpr):
    """Pull the first NBYTES of each selected weight row into L2."""
    addr = (w_ptr + rows.to(tl.int64) * stride_w).to(tl.int64)
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; "
        "@p cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
        constraints="=r,l,r,r",
        args=[addr, rmask.to(tl.int32), tl.full(rows.shape, NBYTES, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gate_up_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_x,
    stride_w,
    stride_o,
    N_OUT: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    PF_BYTES: tl.constexpr,
    PDL: tl.constexpr,
):
    """Each CTA owns a contiguous, balanced range of output columns and
    streams the matching gate and up weight rows over the whole K (no split-K,
    no separate activation kernel). The epilogue rounds gate and up to bf16
    and applies silu-and-mul exactly as ``silu_and_mul`` rounds it. The first
    PF_BYTES of every owned row are pulled into L2 before
    griddepcontrol.wait: the weights do not depend on the predecessor."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    r0 = (pid * N_OUT) // nprog
    r1 = ((pid + 1) * N_OUT) // nprog
    rows = r0 + tl.arange(0, BLOCK_N)
    rmask = rows < r1
    if PF_BYTES > 0:
        _l2_prefetch(w_ptr, rows, rmask, stride_w, PF_BYTES)
        _l2_prefetch(w_ptr, rows + N_OUT, rmask, stride_w, PF_BYTES)
    offs_m = tl.arange(0, BLOCK_M)
    mmask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    wg = w_ptr + rows.to(tl.int64)[:, None] * stride_w + offs_k[None, :]
    wu = wg + N_OUT * stride_w
    xp = x_ptr + offs_m[:, None] * stride_x + offs_k[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    acc_g = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        x = tl.trans(tl.load(xp + k, mask=mmask[:, None], other=0.0))
        g = tl.load(wg + k, mask=rmask[:, None], other=0.0, eviction_policy="evict_first")
        acc_g = tl.dot(g, x, acc_g)
        u = tl.load(wu + k, mask=rmask[:, None], other=0.0, eviction_policy="evict_first")
        acc_u = tl.dot(u, x, acc_u)
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = tl.math.div_rn(g, 1.0 + tldevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    o_ptrs = out_ptr + offs_m[None, :] * stride_o + rows[:, None]
    tl.store(o_ptrs, (s * u).to(tl.bfloat16), mask=rmask[:, None] & mmask[None, :])


_NUM_SMS: dict[int, int] = {}

# (ctas_per_sm, block_k, num_stages, num_warps, l2_prefetch_mb): measured on
# B300 for the TP2 draft's gate_up_proj (9216 x 2 rows, K 6144) at M = 1 and 6,
# cold weights; l2_prefetch_mb is the weight budget pulled into L2 before
# griddepcontrol.wait.
_GATE_UP_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int, int]] = {
    (9216, 6144): (1, 128, 6, 8, 16),
}
_GATE_UP_DEFAULT = (1, 128, 4, 4, 16)


def _num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_SMS[idx]


def gate_up_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: tuple[int, int, int, int, int] | None = None,
    pdl: bool = True,
) -> torch.Tensor:
    """silu_and_mul(x @ weight.T) for M <= 16 tokens, weight = [gate; up]."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    m, k = x.shape
    rows, kw = weight.shape
    n_out = rows // 2
    assert kw == k and 0 < m <= _BLOCK_M and rows == 2 * n_out
    cps, block_k, stages, warps, pf_mb = config or _GATE_UP_CONFIGS.get(
        (n_out, k), _GATE_UP_DEFAULT
    )
    assert k % block_k == 0
    out = torch.empty(m, n_out, dtype=x.dtype, device=x.device)
    grid = min(n_out, cps * _num_sms(x.device))
    rows_per_cta = triton.cdiv(n_out, grid)
    block_n = max(16, triton.next_power_of_2(rows_per_cta))
    pf_bytes = min((pf_mb << 20) // (2 * rows_per_cta * grid) // 16 * 16, k * 2)
    _gate_up_kernel[(grid,)](
        x,
        weight,
        out,
        m,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        N_OUT=n_out,
        K=k,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_STAGES=stages,
        PF_BYTES=pf_bytes,
        PDL=pdl,
        num_warps=warps,
        launch_pdl=pdl,
    )
    return out


@triton.jit
def _gemv_seg_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_x,
    stride_w,
    stride_o,
    N_OUT: tl.constexpr,
    K: tl.constexpr,
    NSEG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    PF_BYTES: tl.constexpr,
    PDL: tl.constexpr,
):
    """out[M, N] = x[M, K] @ W[N, K]^T. Each CTA owns a balanced range of
    output rows; every iteration streams one BLOCK_K tile from each of NSEG
    K segments (more weight bytes in flight per SM), all into one fp32
    accumulator in a fixed order."""
    KS: tl.constexpr = K // NSEG
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    r0 = (pid * N_OUT) // nprog
    r1 = ((pid + 1) * N_OUT) // nprog
    rows = r0 + tl.arange(0, BLOCK_N)
    rmask = rows < r1
    if PF_BYTES > 0:
        for s in tl.static_range(NSEG):
            _l2_prefetch(w_ptr + s * KS, rows, rmask, stride_w, PF_BYTES)
    offs_m = tl.arange(0, BLOCK_M)
    mmask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    wp = w_ptr + rows.to(tl.int64)[:, None] * stride_w + offs_k[None, :]
    xp = x_ptr + offs_m[:, None] * stride_x + offs_k[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    for k in tl.range(0, KS, BLOCK_K, num_stages=NUM_STAGES):
        for s in tl.static_range(NSEG):
            w = tl.load(wp + s * KS + k, mask=rmask[:, None], other=0.0, eviction_policy="evict_first")
            x = tl.load(xp + s * KS + k, mask=mmask[:, None], other=0.0)
            acc = tl.dot(w, tl.trans(x), acc)
    o_ptrs = out_ptr + offs_m[None, :] * stride_o + rows[:, None]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=rmask[:, None] & mmask[None, :])


# Shapes (N, K) where the segmented GEMV beat the stock cuBLAS GEMM on B300 at
# M = 1 and 6 (cold weights): (ctas_per_sm, block_k, num_stages, num_warps,
# l2_prefetch_mb, nseg). Other shapes keep the stock GEMM.
_SEG_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}


def gemv_seg(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: tuple[int, int, int, int, int, int],
    pdl: bool = True,
) -> torch.Tensor:
    """x @ weight.T for M <= 16 tokens; config (ctas_per_sm, block_k,
    num_stages, num_warps, l2_prefetch_mb, nseg)."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    m, k = x.shape
    n_out, kw = weight.shape
    assert kw == k and 0 < m <= _BLOCK_M
    cps, block_k, stages, warps, pf_mb, nseg = config
    assert k % (nseg * block_k) == 0
    out = torch.empty(m, n_out, dtype=x.dtype, device=x.device)
    grid = min(n_out, cps * _num_sms(x.device))
    rows_per_cta = triton.cdiv(n_out, grid)
    block_n = max(16, triton.next_power_of_2(rows_per_cta))
    pf_bytes = min((pf_mb << 20) // (nseg * rows_per_cta * grid) // 16 * 16, (k // nseg) * 2)
    _gemv_seg_kernel[(grid,)](
        x,
        weight,
        out,
        m,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        N_OUT=n_out,
        K=k,
        NSEG=nseg,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_STAGES=stages,
        PF_BYTES=pf_bytes,
        PDL=pdl,
        num_warps=warps,
        launch_pdl=pdl,
    )
    return out


# --------------------------------------------------------------------------
# qkv_proj input: [embeds_normed | rms_norm(hidden)] without a cat.
# --------------------------------------------------------------------------


@triton.jit
def _cat_norm_kernel(
    e_ptr,
    h_ptr,
    w_ptr,
    out_ptr,
    stride_e,
    stride_h,
    stride_o,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    COPY_E: tl.constexpr,
    PDL: tl.constexpr,
):
    """Program per token: out[t, H:] = rms_norm(h[t], w) (x * rstd * w, one
    bf16 rounding, as rms_norm rounds it); out[t, :H] = e[t] when COPY_E (else
    the caller already wrote it there). The row stays in registers."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(h_ptr + t * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.math.div_rn(tl.sum(x * x, axis=0), H * 1.0) + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    orow = out_ptr + t * stride_o
    tl.store(orow + H + offs, (x * rstd * w).to(tl.bfloat16), mask=mask)
    if COPY_E:
        tl.store(orow + offs, tl.load(e_ptr + t * stride_e + offs, mask=mask), mask=mask)


def cat_norm(
    e: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    pdl: bool = True,
) -> torch.Tensor:
    m, hdim = h.shape
    copy_e = e.data_ptr() != out.data_ptr()
    _cat_norm_kernel[(m,)](
        e,
        h,
        weight,
        out,
        e.stride(0),
        h.stride(0),
        out.stride(0),
        eps,
        H=hdim,
        BLOCK=triton.next_power_of_2(hdim),
        COPY_E=copy_e,
        PDL=pdl,
        num_warps=16,
        launch_pdl=pdl,
    )
    return out


# --------------------------------------------------------------------------
# RoPE (neox) + fp8 query quantization + KV-cache insert.
# --------------------------------------------------------------------------


# How the fp32 rotation contracts into FMAs; 0 matches nvcc's contraction of
# rotary_embedding's ``x * cos - y * sin`` / ``y * cos + x * sin``.
ROPE_MODE = 0


@triton.jit
def _rotate(x, y, cos, sin, MODE: tl.constexpr):
    if MODE == 0:
        xr = tl.fma(x, cos, -(y * sin))
        yr = tl.fma(y, cos, x * sin)
    elif MODE == 1:
        xr = tl.fma(-y, sin, x * cos)
        yr = tl.fma(x, sin, y * cos)
    else:
        xr = x * cos - y * sin
        yr = y * cos + x * sin
    return xr.to(tl.bfloat16), yr.to(tl.bfloat16)


@triton.jit
def _rope_kv_kernel(
    qkv_ptr,
    pos_ptr,
    cs_ptr,
    slot_ptr,
    kc_ptr,
    vc_ptr,
    q8_ptr,
    qscale_ptr,
    kscale_ptr,
    vscale_ptr,
    n_slots,
    stride_qkv,
    stride_cs,
    stride_q8,
    blk_stride,
    page_stride,
    head_stride,
    block_size,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    WRITE_QBF16: tl.constexpr,
    WRITE_Q8: tl.constexpr,
    WRITE_KV: tl.constexpr,
    KV_FP8: tl.constexpr,
    ROPE: tl.constexpr,
    PDL: tl.constexpr,
):
    """One program per token. Rounds exactly as rotary_embedding (fp32 math,
    one bf16 rounding per output), scaled_fp8_quant (x * (1 / scale), clamp,
    satfinite) and reshape_and_cache_flash (x / scale, satfinite)."""
    t = tl.program_id(0)
    HALF: tl.constexpr = D // 2
    offs = tl.arange(0, HALF)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pos = tl.load(pos_ptr + t)
    cs = cs_ptr + pos * stride_cs
    cos = tl.load(cs + offs).to(tl.float32)[None, :]
    sin = tl.load(cs + HALF + offs).to(tl.float32)[None, :]
    row = qkv_ptr + t * stride_qkv

    hq = tl.arange(0, NQ)
    qp = row + hq[:, None] * D + offs[None, :]
    qx = tl.load(qp).to(tl.float32)
    qy = tl.load(qp + HALF).to(tl.float32)
    qxr, qyr = _rotate(qx, qy, cos, sin, ROPE)
    if WRITE_QBF16:
        tl.store(qp, qxr)
        tl.store(qp + HALF, qyr)
    if WRITE_Q8:
        inv = tl.math.div_rn(tl.full((), 1.0, tl.float32), tl.load(qscale_ptr))
        q8p = q8_ptr + t * stride_q8 + hq[:, None] * D + offs[None, :]
        q8x = tl.clamp(qxr.to(tl.float32) * inv, -448.0, 448.0)
        q8y = tl.clamp(qyr.to(tl.float32) * inv, -448.0, 448.0)
        tl.store(q8p, q8x.to(tl.float8e4nv))
        tl.store(q8p + HALF, q8y.to(tl.float8e4nv))

    if WRITE_KV:
        slot = tl.load(slot_ptr + t, mask=t < n_slots, other=-1)
        if slot >= 0:
            hk = tl.arange(0, NKV)
            kp = row + NQ * D + hk[:, None] * D + offs[None, :]
            kx = tl.load(kp).to(tl.float32)
            ky = tl.load(kp + HALF).to(tl.float32)
            kxr, kyr = _rotate(kx, ky, cos, sin, ROPE)
            offs_d = tl.arange(0, D)
            v = tl.load(row + (NQ + NKV) * D + hk[:, None] * D + offs_d[None, :])
            blk = slot // block_size
            off = slot % block_size
            base = blk * blk_stride + off * page_stride + hk[:, None] * head_stride
            if KV_FP8:
                ks = tl.load(kscale_ptr)
                vs = tl.load(vscale_ptr)
                kxs = tl.math.div_rn(kxr.to(tl.float32), ks).to(tl.float8e4nv)
                kys = tl.math.div_rn(kyr.to(tl.float32), ks).to(tl.float8e4nv)
                vv = tl.math.div_rn(v.to(tl.float32), vs).to(tl.float8e4nv)
            else:
                kxs = kxr
                kys = kyr
                vv = v
            tl.store(kc_ptr + base + offs[None, :], kxs)
            tl.store(kc_ptr + base + HALF + offs[None, :], kys)
            tl.store(vc_ptr + base + offs_d[None, :], vv)


def rope_kv(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    slot_mapping: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_scale: torch.Tensor | None,
    write_q_bf16: bool,
    pdl: bool = True,
    rope_mode: int | None = None,
) -> torch.Tensor | None:
    """In place on qkv's q part when write_q_bf16; returns the fp8 query
    ([M, num_q_heads * head_size]) when q_scale is given. k_cache / v_cache
    are the [num_blocks, block_size, num_kv_heads, head_size] views that
    reshape_and_cache_flash receives."""
    m = qkv.shape[0]
    q8 = None
    if q_scale is not None:
        q8 = torch.empty(
            m, num_q_heads * head_size, dtype=torch.float8_e4m3fn, device=qkv.device
        )
    write_kv = slot_mapping is not None and k_cache is not None
    kv_fp8 = write_kv and k_cache.dtype == torch.float8_e4m3fn
    _rope_kv_kernel[(m,)](
        qkv,
        positions,
        cos_sin_cache,
        slot_mapping if write_kv else positions,
        k_cache if write_kv else qkv,
        v_cache if write_kv else qkv,
        q8 if q8 is not None else qkv,
        q_scale if q_scale is not None else k_scale,
        k_scale,
        v_scale,
        slot_mapping.shape[0] if write_kv else 0,
        qkv.stride(0),
        cos_sin_cache.stride(0),
        q8.stride(0) if q8 is not None else 0,
        k_cache.stride(0) if write_kv else 0,
        k_cache.stride(1) if write_kv else 0,
        k_cache.stride(2) if write_kv else 0,
        k_cache.shape[1] if write_kv else 1,
        NQ=num_q_heads,
        NKV=num_kv_heads,
        D=head_size,
        WRITE_QBF16=write_q_bf16,
        WRITE_Q8=q8 is not None,
        WRITE_KV=write_kv,
        KV_FP8=kv_fp8,
        ROPE=ROPE_MODE if rope_mode is None else rope_mode,
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return q8


# --------------------------------------------------------------------------
# Eagle3 fc_norm: per-chunk RMSNorm of the aux hidden states, concatenated.
# --------------------------------------------------------------------------


@triton.jit
def _fc_norm_kernel(
    a0_ptr,
    a1_ptr,
    a2_ptr,
    w0_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_o,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """Program (token, chunk): out[t, c*H:(c+1)*H] = rms_norm(aux_c[t], w_c),
    rounded as rms_norm rounds (x * rstd * w, one bf16 rounding)."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if c == 0:
        a = a0_ptr + t * stride_a0
        w = w0_ptr
    elif c == 1:
        a = a1_ptr + t * stride_a1
        w = w1_ptr
    else:
        a = a2_ptr + t * stride_a2
        w = w2_ptr
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(a + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.math.div_rn(tl.sum(x * x, axis=0), H * 1.0) + eps)
    wv = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * stride_o + c * H + offs, (x * rstd * wv).to(tl.bfloat16), mask=mask)


def fc_norm(
    aux: list[torch.Tensor], weights: list[torch.Tensor], eps: float, pdl: bool = True
) -> torch.Tensor:
    m, h = aux[0].shape
    out = torch.empty(m, 3 * h, dtype=aux[0].dtype, device=aux[0].device)
    _fc_norm_kernel[(m, 3)](
        aux[0],
        aux[1],
        aux[2],
        weights[0],
        weights[1],
        weights[2],
        out,
        aux[0].stride(0),
        aux[1].stride(0),
        aux[2].stride(0),
        out.stride(0),
        eps,
        H=h,
        BLOCK=triton.next_power_of_2(h),
        PDL=pdl,
        num_warps=16,
        launch_pdl=pdl,
    )
    return out


# --------------------------------------------------------------------------
# Model integration.
# --------------------------------------------------------------------------


def _is_plain_bf16_linear(lin) -> bool:
    qm = getattr(lin, "quant_method", None)
    w = getattr(lin, "weight", None)
    return (
        type(qm).__name__ == "UnquantizedLinearMethod"
        and isinstance(w, torch.Tensor)
        and w.dtype == torch.bfloat16
        and w.is_cuda
        and w.is_contiguous()
        and getattr(lin, "bias", None) is None
    )


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class Eagle3DraftFusion:
    """Fused forward of a one-layer EAGLE3 ``LlamaModel`` (see module doc)."""

    def __init__(self, model, parts: frozenset[str]) -> None:
        self.model = model
        self.parts = parts
        self._checked = False
        self._ok = False
        self._zero_residual: torch.Tensor | None = None

    @classmethod
    def maybe_create(cls, model, vllm_config) -> "Eagle3DraftFusion | None":
        from vllm.config.compilation import CompilationMode
        from vllm.platforms import current_platform

        parts = enabled_parts()
        if not parts or not current_platform.is_cuda():
            return None
        if len(model.layers) != 1 or vllm_config.compilation_config.mode != (
            CompilationMode.NONE
        ):
            logger.info_once(
                "EAGLE3 draft fusion disabled: needs one draft layer and "
                "compilation mode NONE."
            )
            return None
        return cls(model, parts)

    # -- one-time structural check (weights are loaded and shared by now) --
    def _check(self) -> bool:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        model = self.model
        layer = model.layers[0]
        attn = layer.self_attn
        reasons = []
        tp = get_tensor_model_parallel_world_size()
        if "ar_norm" in self.parts and tp == 1:
            self.parts = self.parts - {"ar_norm"}
        if not all(
            _is_plain_bf16_linear(lin)
            for lin in (
                attn.qkv_proj,
                attn.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
            )
        ):
            reasons.append("draft linears are not unquantized bf16")
        if getattr(layer, "_residual_norm", None) != layer._norm_after_residual:
            reasons.append("norm_before_residual draft")
        if layer.layer_idx != 0:
            reasons.append("layer index")
        rope = attn.rotary_emb
        if not (
            type(rope) is RotaryEmbedding
            and rope.is_neox_style
            and not getattr(rope, "use_flashinfer", False)
            and rope.rotary_dim == rope.head_size == attn.head_dim
        ):
            reasons.append("rotary embedding is not full-dim neox")
        a = attn.attn
        if not isinstance(a, Attention) or a.kv_sharing_target_layer_name:
            reasons.append("attention layer")
        elif not (_pow2(attn.num_heads) and _pow2(attn.num_kv_heads)):
            reasons.append("head counts")
        elif a.kv_cache_dtype not in ("auto", "fp8", "fp8_e4m3"):
            reasons.append(f"kv cache dtype {a.kv_cache_dtype}")
        elif not hasattr(a.impl, "do_kv_cache_update"):
            reasons.append("attention backend without do_kv_cache_update")
        elif a._k_scale.numel() != 1 or a._v_scale.numel() != 1:
            reasons.append("per-head kv scales")
        elif getattr(a.impl, "is_kvcache_nvfp4", False):
            reasons.append("nvfp4 kv cache")
        if model.use_aux_hidden_state and "fc" in self.parts:
            if not (
                _is_plain_bf16_linear(model.fc)
                and model.fc_norm is not None
                and model.num_aux_hidden_states == 3
                and model.input_norm is None
            ):
                self.parts = self.parts - {"fc"}
        if reasons:
            logger.warning_once(
                "EAGLE3 draft fusion disabled: %s", "; ".join(reasons)
            )
            return False
        logger.info_once(
            "EAGLE3 draft fusion enabled (parts: %s, M <= %d)",
            ",".join(sorted(self.parts)),
            MAX_M,
        )
        return True

    def _active(self, m: int) -> bool:
        if not self._checked:
            self._ok = self._check()
            self._checked = True
        return self._ok and 0 < m <= MAX_M

    # -- pieces --
    def _ar_rms(self, partial, residual, norm, norm_out=None):
        """(norm(all_reduce(partial) + residual), all_reduce(partial) + residual)
        via the fused FlashInfer kernel, or None if it does not apply."""
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _AR_RESIDUAL_RMS_NORM,
            _can_use_flashinfer,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        tp = get_tensor_model_parallel_world_size()
        ok, max_token_num = _can_use_flashinfer(partial, tp)
        if not ok:
            return None
        if norm_out is None:
            norm_out = torch.empty_like(partial)
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=partial,
            residual=residual,
            rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon,
            world_size=tp,
            weight_bias=0.0,
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
            pattern_code=_AR_RESIDUAL_RMS_NORM,
            norm_out=norm_out,
        )
        return norm_out, partial

    def _embed_normed(
        self, input_ids: torch.Tensor, norm_out: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm import _custom_ops as ops

        model = self.model
        emb = model.embed_tokens
        norm = model.layers[0].input_layernorm
        if (
            "ar_norm" in self.parts
            and getattr(emb, "tp_size", 1) > 1
            and getattr(emb, "use_fused_embedding", False)
            and getattr(emb, "parallel_group", None) is None
        ):
            si = emb.shard_indices
            partial = ops.vocab_parallel_embedding(
                input_ids,
                emb.weight,
                si.org_vocab_start_index,
                si.org_vocab_end_index,
                si.num_org_vocab_padding,
                si.added_vocab_start_index,
                si.added_vocab_end_index,
            )
            m, h = partial.shape
            if self._zero_residual is None or self._zero_residual.shape[1] != h:
                self._zero_residual = torch.zeros(
                    MAX_M, h, dtype=partial.dtype, device=partial.device
                )
            # A zero residual: the kernel's residual output is the all-reduce,
            # its norm output the normalized embedding.
            res = self._ar_rms(partial, self._zero_residual[:m], norm, norm_out)
            if res is not None:
                return res[0]
            from vllm.distributed import tensor_model_parallel_all_reduce

            return norm(tensor_model_parallel_all_reduce(partial))
        return norm(model.embed_input_ids(input_ids))

    def _attention(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
            unified_attention_with_output,
        )

        attn = self.model.layers[0].self_attn
        a = attn.attn
        m = qkv.shape[0]
        q_size, kv_size = attn.q_size, attn.kv_size
        fwd = get_forward_context()
        if (
            "rope_kv" not in self.parts
            or fwd.attn_metadata is None
            or not isinstance(fwd.slot_mapping, dict)
        ):
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q, k = attn.rotary_emb(positions, q, k)
            return a(q, k, v)

        md, _, kv_cache, slot_mapping = get_attention_context(a.layer_name)
        k_cache = v_cache = None
        if slot_mapping is not None and kv_cache is not None and kv_cache.numel():
            # The views do_kv_cache_update hands to reshape_and_cache_flash.
            k_cache, v_cache = kv_cache.transpose(1, 2).split(a.head_size, dim=-1)
            if a.kv_cache_dtype != "auto":
                k_cache = k_cache.view(torch.float8_e4m3fn)
                v_cache = v_cache.view(torch.float8_e4m3fn)
        else:
            slot_mapping = None
        # The fp8 query can go straight to the decode kernel when the batch is
        # decode-only now and at every replay: eager calls see the real
        # metadata, FULL graphs are replayed only for uniform decode batches.
        # PIECEWISE segments may replay with prefills, so they keep bf16 q.
        mode = fwd.cudagraph_runtime_mode
        q8_ok = (
            mode != CUDAGraphMode.PIECEWISE
            and md is not None
            and getattr(md, "num_prefill_tokens", 1) == 0
            and getattr(md, "num_decode_tokens", 0) > 0
            and not getattr(md, "use_cascade", True)
            and getattr(md, "q_data_type_decode", None) == torch.float8_e4m3fn
            and a.query_quant is not None
            and a.impl.supports_quant_query_input
        )
        q8 = rope_kv(
            qkv,
            positions,
            # The cache dtype rotary_embedding uses (bf16 for a bf16 query).
            attn.rotary_emb._match_cos_sin_cache_dtype(qkv),
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            slot_mapping,
            k_cache,
            v_cache,
            a._k_scale,
            a._v_scale,
            a._q_scale if q8_ok else None,
            write_q_bf16=not q8_ok,
        )
        if q8 is None:
            # KV is cached; the layer quantizes q and runs attention as usual.
            return a(qkv[:, :q_size], None, None)
        output = torch.empty(m, q_size, dtype=qkv.dtype, device=qkv.device)
        unified_attention_with_output(
            q8.view(m, attn.num_heads, attn.head_dim),
            None,
            None,
            output.view(m, attn.num_heads, attn.head_dim),
            a.layer_name,
        )
        return output

    def _linear(self, lin, x: torch.Tensor) -> torch.Tensor:
        """The (un-reduced) projection: the segmented GEMV where it measured
        faster, else the same GEMM the layer runs."""
        if "seg_gemv" in self.parts:
            cfg = _SEG_CONFIGS.get(tuple(lin.weight.shape))
            if cfg is not None:
                return gemv_seg(x, lin.weight, cfg)
        return lin.quant_method.apply(lin, x, None)

    def _reduce_norm(self, partial, residual, norm):
        if "ar_norm" in self.parts:
            res = self._ar_rms(partial, residual, norm)
            if res is not None:
                return res
        from vllm.distributed import tensor_model_parallel_all_reduce

        reduced = tensor_model_parallel_all_reduce(partial)
        return norm(reduced, residual.clone())

    # -- forward --
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        m = input_ids.shape[0]
        if (
            input_ids.dim() != 1
            or hidden_states.dim() != 2
            or hidden_states.shape[0] != m
            or not self._active(m)
        ):
            return None
        model = self.model
        layer = model.layers[0]
        attn = layer.self_attn
        mlp = layer.mlp
        hidden_states = hidden_states.contiguous()
        h = hidden_states.shape[1]

        if "cat_norm" in self.parts:
            x = torch.empty(m, 2 * h, dtype=hidden_states.dtype, device=hidden_states.device)
            # One row: the embedding norm can write its half of x in place.
            e_n = self._embed_normed(input_ids, x[:, :h] if m == 1 else None)
            cat_norm(
                e_n,
                hidden_states,
                layer.hidden_norm.weight,
                layer.hidden_norm.variance_epsilon,
                x,
            )
        else:
            e_n = self._embed_normed(input_ids)
            x = torch.cat([e_n, layer.hidden_norm(hidden_states)], dim=-1)
        qkv = self._linear(attn.qkv_proj, x)

        attn_out = self._attention(qkv, positions)
        o_part = self._linear(attn.o_proj, attn_out)
        # residual = the layer input (norm after residual).
        a_n, r1 = self._reduce_norm(o_part, hidden_states, layer.post_attention_layernorm)

        if "gate_up" in self.parts:
            act = gate_up_silu(a_n, mlp.gate_up_proj.weight)
        else:
            act = mlp.act_fn(self._linear(mlp.gate_up_proj, a_n))
        d_part = self._linear(mlp.down_proj, act)
        h_n, prenorm = self._reduce_norm(d_part, r1, model.norm)
        return h_n, (h_n if model.norm_output else prenorm)

    def combine_aux(self, aux: list[torch.Tensor]) -> torch.Tensor | None:
        """fc(fc_norm(cat(aux))) for decode-sized batches, else None."""
        model = self.model
        if (
            len(aux) != 3
            or aux[0].dim() != 2
            or any(t.stride(-1) != 1 for t in aux)
            or not self._active(aux[0].shape[0])
            or "fc" not in self.parts
        ):
            return None
        norms = model.fc_norm
        x = fc_norm(aux, [n.weight for n in norms], norms[0].variance_epsilon)
        return self._linear(model.fc, x)
