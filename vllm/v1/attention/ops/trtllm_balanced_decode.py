# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-balanced split-KV decode on top of FlashInfer's trtllm-gen kernels.

trtllm-gen's generation kernels give every sequence of a decode batch the same
number of KV CTAs, ``min(ceil(L / 256), SMs // (batch * head CTAs))``, chosen on
the host from ``max_seq_len`` (``max_model_len`` under FULL CUDA graphs). Each
sequence's KV is then split evenly over its own CTAs, so with a mix of context
lengths the CTAs of the longest request set the kernel time while the SMs that
served shorter requests idle.

This path re-batches one decode call into page-aligned KV chunks of equal size,
planned on the GPU from the live ``seq_lens`` so a captured graph serves every
length mix. It runs the same trtllm-gen kernel on that virtual batch (one query
row per chunk, so the kernel never splits further) and merges the per-chunk bf16
outputs with their base-2 LSEs in fp32. That is the reduction trtllm-gen applies
to its own split-KV partials; only the split points differ, and the stock split
points already change with the batch size.

For speculative verification (``q_len > 1``) the ``q_len`` query tokens of a
request are folded into the head axis over the KV prefix that every one of them
sees (``seq_len - q_len + 1`` keys). The at most ``q_len - 1`` causally masked
tail keys are added exactly in the merge kernel.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

LOG2E = 1.4426950408889634


@triton.jit
def _chunk_plan(
    seq_lens_ptr,
    num_reqs,
    QLEN: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    target_chunks,
    min_chunk_pages,
    max_chunk_pages,
):
    """Per-request chunking shared by the plan and merge kernels.

    Returns (offs, lens, visible, n_chunks, first_chunk, chunk_pages): request r
    owns virtual rows [first_chunk[r], first_chunk[r] + n_chunks[r]), each
    covering ``chunk_pages[r]`` pages of its ``visible[r]`` prefix keys (the last
    one possibly fewer). The size bound C comes from ``target_chunks - num_reqs``
    so that sum(ceil(pages / C)) <= target_chunks: the virtual batch never spills
    into an extra wave. Each request's chunks are then equalized.
    """
    offs = tl.arange(0, BLOCK_R)
    lens = tl.load(seq_lens_ptr + offs, mask=offs < num_reqs, other=0)
    visible = tl.maximum(lens - (QLEN - 1), 0)
    pages = (visible + PAGE - 1) // PAGE
    total_pages = tl.sum(pages, axis=0)
    budget = tl.maximum(target_chunks - num_reqs, 1)
    bound = (total_pages + budget - 1) // budget
    bound = tl.minimum(tl.maximum(bound, min_chunk_pages), max_chunk_pages)
    n_chunks = (pages + bound - 1) // bound
    chunk_pages = (pages + tl.maximum(n_chunks, 1) - 1) // tl.maximum(n_chunks, 1)
    first_chunk = tl.cumsum(n_chunks, axis=0) - n_chunks
    return offs, lens, visible, n_chunks, first_chunk, chunk_pages


@triton.jit(do_not_specialize=["num_reqs", "max_chunk_pages"])
def _plan_gather_kernel(
    seq_lens_ptr,
    block_tables_ptr,
    stride_bt,
    q_ptr,
    vq_ptr,
    vbt_ptr,
    vlens_ptr,
    num_reqs,
    target_chunks,
    min_chunk_pages,
    max_chunk_pages,
    QLEN: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    Q_ROW_BYTES: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_P: tl.constexpr,
    FOLD: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    v = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    offs, lens, visible, n_chunks, first_chunk, chunk_pages = _chunk_plan(
        seq_lens_ptr,
        num_reqs,
        QLEN,
        PAGE,
        BLOCK_R,
        target_chunks,
        min_chunk_pages,
        max_chunk_pages,
    )
    used = tl.sum(n_chunks, axis=0)
    # Owner request: the last request whose first chunk is <= v.
    owner_mask = (offs < num_reqs) & (first_chunk <= v) & (n_chunks > 0)
    r = tl.max(tl.where(owner_mask, offs, -1), axis=0)
    if v < used:
        sel = offs == r
        start = tl.sum(tl.where(sel, first_chunk, 0), axis=0)
        vis = tl.sum(tl.where(sel, visible, 0), axis=0)
        cp = tl.sum(tl.where(sel, chunk_pages, 0), axis=0)
        c = v - start
        p0 = c * cp
        tok0 = p0 * PAGE
        ntok = tl.minimum(cp * PAGE, vis - tok0)
        npages = (ntok + PAGE - 1) // PAGE
        src_bt = block_tables_ptr + r.to(tl.int64) * stride_bt + p0
        # Folded: one virtual row per chunk with the q_len tokens as heads.
        # Unfolded: q_len rows per chunk (one per token) over the same pages.
        NROW: tl.constexpr = 1 if FOLD else QLEN
        for jr in tl.static_range(NROW):
            row = v.to(tl.int64) * NROW + jr
            tl.store(vlens_ptr + row, ntok)
            dst_bt = vbt_ptr + row * max_chunk_pages
            for k in range(0, npages, BLOCK_P):
                po = k + tl.arange(0, BLOCK_P)
                pm = po < npages
                tl.store(dst_bt + po, tl.load(src_bt + po, mask=pm), mask=pm)
        # Either way the chunk's query bytes are the request's q_len token rows.
        src_q = q_ptr + r.to(tl.int64) * Q_ROW_BYTES
        dst_q = vq_ptr + v.to(tl.int64) * Q_ROW_BYTES
        for k in tl.static_range(0, Q_ROW_BYTES, BLOCK_Q):
            qo = k + tl.arange(0, BLOCK_Q)
            qm = qo < Q_ROW_BYTES
            tl.store(dst_q + qo, tl.load(src_q + qo, mask=qm), mask=qm)
    else:
        NROW0: tl.constexpr = 1 if FOLD else QLEN
        for jr in tl.static_range(NROW0):
            tl.store(vlens_ptr + v.to(tl.int64) * NROW0 + jr, 0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["num_reqs", "max_chunk_pages"])
def _merge_kernel(
    seq_lens_ptr,
    block_tables_ptr,
    stride_bt,
    q_ptr,
    k_ptr,
    v_ptr,
    stride_kb,
    stride_kt,
    stride_vb,
    stride_vt,
    vout_ptr,
    vlse_ptr,
    out_ptr,
    stride_ot,
    stride_oh,
    num_reqs,
    target_chunks,
    min_chunk_pages,
    max_chunk_pages,
    bmm1_scale_log2,
    bmm2_scale,
    QLEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    TAIL_P: tl.constexpr,
    FOLD: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok = tl.program_id(0)
    h = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tok // QLEN
    j = tok - r * QLEN
    offs, lens, visible, n_chunks, first_chunk, chunk_pages = _chunk_plan(
        seq_lens_ptr,
        num_reqs,
        QLEN,
        PAGE,
        BLOCK_R,
        target_chunks,
        min_chunk_pages,
        max_chunk_pages,
    )
    sel = offs == r
    start = tl.sum(tl.where(sel, first_chunk, 0), axis=0)
    n = tl.sum(tl.where(sel, n_chunks, 0), axis=0)
    seq_len = tl.sum(tl.where(sel, lens, 0), axis=0)
    vis = tl.sum(tl.where(sel, visible, 0), axis=0)

    VROW: tl.constexpr = QLEN * NUM_HEADS if FOLD else NUM_HEADS
    hv = j * NUM_HEADS + h if FOLD else h
    d = tl.arange(0, HEAD_DIM)
    if QLEN > 1:
        # Token j also sees tail keys [vis, vis + j), which the folded chunks
        # exclude. Their loads do not depend on the partials, so issue them
        # first; they are scored with the kernel's scaling after the merge.
        tt = tl.arange(0, TAIL_P)
        tpos = vis + tt
        tmask = (tt < QLEN - 1) & (tpos < seq_len - (QLEN - 1) + j)
        bt_row = block_tables_ptr + r.to(tl.int64) * stride_bt
        tpage = tl.load(bt_row + tpos // PAGE, mask=tmask, other=0).to(tl.int64)
        tslot = tpos % PAGE
        tk = tl.load(
            k_ptr + (tpage * stride_kb + tslot * stride_kt)[:, None] + d[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        tv = tl.load(
            v_ptr + (tpage * stride_vb + tslot * stride_vt)[:, None] + d[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        qrow = tl.load(
            q_ptr + tok.to(tl.int64) * (NUM_HEADS * HEAD_DIM) + h * HEAD_DIM + d
        ).to(tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    den = tl.zeros([1], dtype=tl.float32)
    m = tl.full([1], float("-inf"), dtype=tl.float32)
    for cb in range(0, n, BLOCK_C):
        cs = cb + tl.arange(0, BLOCK_C)
        cm = cs < n
        rows = (start + cs).to(tl.int64)
        if not FOLD:
            rows = rows * QLEN + j
        lse = tl.load(vlse_ptr + rows * VROW + hv, mask=cm, other=float("-inf"))
        o = tl.load(
            vout_ptr + (rows * VROW + hv)[:, None] * HEAD_DIM + d[None, :],
            mask=cm[:, None],
            other=0.0,
        ).to(tl.float32)
        m_new = tl.maximum(m, tl.max(lse, axis=0))
        alpha = tl.exp2(m - m_new)
        w = tl.exp2(lse - m_new)
        acc = acc * alpha + tl.sum(w[:, None] * o, axis=0)
        den = den * alpha + tl.sum(w, axis=0)
        m = m_new
    if QLEN > 1:
        ts = tl.sum(tk * qrow[None, :], axis=1) * bmm1_scale_log2
        ts = tl.where(tmask, ts, float("-inf"))
        m_new = tl.maximum(m, tl.max(ts, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        w = tl.exp2(ts - m_safe)
        acc = acc * alpha + tl.sum(w[:, None] * tv, axis=0) * bmm2_scale
        den = den * alpha + tl.sum(w, axis=0)
    res = tl.where(den > 0, acc / tl.maximum(den, 1e-30), 0.0)
    tl.store(
        out_ptr + tok.to(tl.int64) * stride_ot + h * stride_oh + d,
        res.to(out_ptr.dtype.element_ty),
    )


class TrtllmBalancedDecode:
    """Static buffers and launch logic for one device.

    ``chunks_per_sm`` sets the virtual batch size (at most ``SMs * chunks_per_sm``
    chunks per call, at least one SM's worth so trtllm-gen never splits a chunk
    again); ``min_chunk_pages`` bounds chunk size from below for short contexts.
    """

    def __init__(
        self,
        device: torch.device,
        max_model_len: int,
        page_size: int,
        max_num_reqs: int,
        chunks_per_sm: int = 1,
        min_chunk_pages: int = 8,
        min_decodes: int = 4,
        max_virtual_heads: int = 64,
        head_dim: int = 128,
        num_sms: int | None = None,
        fold_tokens: bool = True,
        max_q_len: int = 4,
        use_pdl: bool | None = None,
    ) -> None:
        if num_sms is None:
            from vllm.utils.platform_utils import num_compute_units

            num_sms = num_compute_units(device.index or 0)
        if use_pdl is None:
            from vllm.platforms import current_platform

            use_pdl = current_platform.is_arch_support_pdl()
        self.use_pdl = bool(use_pdl)
        self._pdl_launch = {"launch_pdl": True} if self.use_pdl else {}
        self.device = device
        self.page_size = page_size
        self.max_pages_per_req = triton.cdiv(max_model_len, page_size)
        self.num_sms = num_sms
        self.target_chunks = max(1, chunks_per_sm) * self.num_sms
        self.min_chunk_pages = max(1, min_chunk_pages)
        self.min_decodes = max(1, min_decodes)
        self.max_num_reqs = max_num_reqs
        self.max_virtual_heads = max_virtual_heads
        self.head_dim = head_dim
        self.fold_tokens = fold_tokens
        self.max_q_len = max_q_len
        # Sized once for the largest call: a captured graph keeps the addresses,
        # so these buffers are never reallocated.
        rows = self.target_chunks + max_num_reqs
        heads = max_virtual_heads if fold_tokens else max_virtual_heads * max_q_len
        bt_rows = rows if fold_tokens else rows * max_q_len
        self._vq = torch.empty(
            rows * heads * head_dim * 2, dtype=torch.uint8, device=device
        )
        self._vout = torch.empty(
            rows * heads * head_dim, dtype=torch.bfloat16, device=device
        )
        self._vlse = torch.empty(rows * heads, dtype=torch.float32, device=device)
        self._vbt = torch.zeros(
            bt_rows * self.max_chunk_pages(max_num_reqs),
            dtype=torch.int32,
            device=device,
        )
        self._vlens = torch.zeros(bt_rows, dtype=torch.int32, device=device)

    def max_chunk_pages(self, num_reqs: int) -> int:
        budget = max(self.target_chunks - num_reqs, 1)
        cap = triton.cdiv(num_reqs * self.max_pages_per_req, budget)
        return max(self.min_chunk_pages, cap)

    def supports(
        self, num_reqs: int, q_len: int, num_heads: int, head_dim: int, page_size: int
    ) -> bool:
        vheads = q_len * num_heads if self.fold_tokens else num_heads
        return (
            self.min_decodes <= num_reqs <= self.max_num_reqs
            and head_dim == self.head_dim
            and page_size == self.page_size
            and q_len <= self.max_q_len
            and vheads <= self.max_virtual_heads
            and vheads & (vheads - 1) == 0
        )

    def run(
        self,
        query: torch.Tensor,  # [num_decodes * q_len, H, D], contiguous
        kv_cache: tuple[torch.Tensor, torch.Tensor],  # HND [blocks, 1, page, D] views
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,  # [num_decodes, max_pages] int32, contiguous
        seq_lens: torch.Tensor,  # [num_decodes] int32
        q_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        out: torch.Tensor,  # [num_decodes * q_len, H, D] bf16/fp16
        trtllm_decode=None,
    ) -> None:
        if trtllm_decode is None:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache

            trtllm_decode = trtllm_batch_decode_with_kv_cache
        num_tokens, num_heads, head_dim = query.shape
        num_reqs = num_tokens // q_len
        k_cache, v_cache = kv_cache
        assert k_cache.shape[1] == 1, "balanced decode folds one KV head per rank"
        page = self.page_size
        fold = self.fold_tokens
        # Query bytes of one request (its q_len token rows), copied per chunk.
        row_bytes = q_len * num_heads * head_dim * query.element_size()
        rows = self.target_chunks + num_reqs
        width = self.max_chunk_pages(num_reqs)
        # trtllm-gen batch: one row per chunk (folded) or per (chunk, token).
        vheads = q_len * num_heads if fold else num_heads
        nrows = rows if fold else rows * q_len
        vq, vout, vlse, vbt, vlens = (
            self._vq,
            self._vout,
            self._vlse,
            self._vbt,
            self._vlens,
        )
        vq_t = vq[: rows * row_bytes].view(query.dtype).view(nrows, vheads, head_dim)
        vout_t = vout[: nrows * vheads * head_dim].view(nrows, vheads, head_dim)
        vlse_t = vlse[: nrows * vheads].view(nrows, vheads)
        vbt_t = vbt[: nrows * width].view(nrows, width)
        vlens_t = vlens[:nrows]
        block_r = max(16, triton.next_power_of_2(num_reqs))
        _plan_gather_kernel[(rows,)](
            seq_lens,
            block_tables,
            block_tables.stride(0),
            query.view(torch.uint8),
            vq,
            vbt,
            vlens,
            num_reqs,
            self.target_chunks,
            self.min_chunk_pages,
            width,
            QLEN=q_len,
            PAGE=page,
            BLOCK_R=block_r,
            Q_ROW_BYTES=row_bytes,
            BLOCK_Q=min(4096, triton.next_power_of_2(row_bytes)),
            BLOCK_P=64,
            FOLD=fold,
            USE_PDL=self.use_pdl,
            **self._pdl_launch,
        )
        trtllm_decode(
            query=vq_t,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=vbt_t,
            seq_lens=vlens_t,
            max_seq_len=width * page,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            out=vout_t,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            lse=vlse_t,
        )
        _merge_kernel[(num_tokens, num_heads)](
            seq_lens,
            block_tables,
            block_tables.stride(0),
            query,
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(2),
            vout_t,
            vlse_t,
            out,
            out.stride(0),
            out.stride(1),
            num_reqs,
            self.target_chunks,
            self.min_chunk_pages,
            width,
            bmm1_scale * LOG2E,
            bmm2_scale,
            QLEN=q_len,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            PAGE=page,
            BLOCK_R=block_r,
            BLOCK_C=64,
            TAIL_P=triton.next_power_of_2(max(q_len - 1, 1)),
            FOLD=fold,
            USE_PDL=self.use_pdl,
            **self._pdl_launch,
        )
