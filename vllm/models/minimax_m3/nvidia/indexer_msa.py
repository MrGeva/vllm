# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

Both sides write block scores into one unified token-major buffer
``[total_q, H, max_k_tiles]``, then a single ``fmha_sm100.sparse_topk_select``
selects the top-k blocks for the whole batch (decode ``[:nd]`` + prefill
``[nd:]``) into the shared ``topk_indices_buffer``. It bounds each row by its
causal page count and force-includes the init/local blocks, so the unwritten
tail of the buffer is pre-filled with ``-inf``.

Prefill scores with ``fmha_sm100``'s score-only (``OnlyScore``) path (much faster
than Triton for the wide prefill score, benchmarked ~3-5x), writing its
``max_score`` straight into the buffer's prefill region (stride-aware, no copy).

Decode scores with CuteDSL when the flattened query tile is supported and fall
back to Triton otherwise, writing into the decode region. Only the top-k is
shared with prefill. With the replicated index-K cache the CuteDSL decode score
keeps per-CTA top-3 records in its epilogue and a one-warp-per-row merge selects
the decode rows' top-k from them (``index_decode_score_select``;
``VLLM_MM3_FUSED_INDEX_SELECT``, on by default when ``max_num_seqs`` <= 8), so
the unified top-k covers the prefill rows only.

With a TP-sharded index-K cache (``idxk_shard``), ``index_query`` carries every
index head; decode scores all of them over this rank's token slice and
max-combines across ranks, and prefill gathers the prefill requests' full pages
for the stock OnlyScore pass over this rank's heads.

``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode_score,
)
from vllm.models.minimax_m3.nvidia import idxk_shard, indexer_kv_split
from vllm.models.minimax_m3.nvidia.ops import minimax_m3_index_decode_score_cutedsl
from vllm.models.minimax_m3.nvidia.ops.index_decode_score_select import (
    DecodeTopKWorkspace,
    minimax_m3_index_decode_score_topk_cutedsl,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu import mixed_attn_timer

logger = init_logger(__name__)

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128

# Fill for unwritten score tiles: -inf so they never win the top-k (score kernels
# only write causally-valid blocks).
_SCORE_SENTINEL = float("-inf")

# Tile (KV-block) dim of the unified score buffer, hardcoded as a cudagraph
# capture-time constant so the decode score kernel's buffer shape is frozen
# across replays. 8192 tiles == 1M tokens of context; -inf padding +
# num_valid_pages bound each row to its causal range, so shorter replays reuse
# the same buffer safely.
MAX_K_TILES = 8192

# Decode top-k from the score kernel's records: "1" on, "0" off, unset on when
# max_num_seqs is at most FUSED_SELECT_AUTO_MAX_SEQS (the c1-c8 decode shapes).
FUSED_SELECT_ENV = "VLLM_MM3_FUSED_INDEX_SELECT"
FUSED_SELECT_AUTO_MAX_SEQS = 8


def _fused_select_enabled(max_num_seqs: int) -> bool:
    value = os.environ.get(FUSED_SELECT_ENV, "").strip().lower()
    if value in ("0", "false", "off"):
        return False
    if value in ("1", "true", "on"):
        return True
    return max_num_seqs <= FUSED_SELECT_AUTO_MAX_SEQS


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSAPrefillMetadata:
    """fmha score plan + Triton top-k inputs for the prefill side (eager)."""

    plan: dict  # fmha_sm100 PlanInfo
    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    prefix_lens: torch.Tensor  # [num_prefills] int32, context tokens
    max_query_len: int
    page_table: torch.Tensor  # flat physical page indices for the prefill side
    # Opt-in KV-split A/B check: the stock (plan, page_table), consumed by the
    # first indexer layer of the step.
    verify: tuple[dict, torch.Tensor] | None = None
    # Sharded index-K: the prefill requests' pages in request order (int64);
    # page_table then indexes this list (the gathered full pages).
    gather_pages: torch.Tensor | None = None


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Decode reuses the inherited base ``decode`` field (the Triton decode
    metadata); ``prefill_msa`` carries the fmha score plan for the prefill side
    (the base ``prefill`` field is unused on this path)."""

    prefill_msa: MiniMaxM3IndexerMSAPrefillMetadata | None = None
    # Per-forward view (``[:num_tokens]``) of the builder's persistent unified
    # score buffer ``[total_q, H, MAX_K_TILES]``, shared by decode and prefill
    # and reused across all layers. Pre-filled with the -inf sentinel in
    # ``build()``; each layer overwrites its valid tiles before its own top-k.
    unified_scores: torch.Tensor | None = None
    # Tile (KV-block) dim of the unified score buffer (== ``MAX_K_TILES``).
    # Forced as the fmha plan's ``max_k_tiles`` so prefill writes its max_score
    # straight into the shared buffer.
    max_k_tiles: int = 0
    # Batch-wide inputs for the single top-k over the unified buffer (decode +
    # prefill in one call). ``cu_seqlens_q`` is the per-request query-start
    # offsets (== query_start_loc, a stable view for cudagraph); ``prefix_lens``
    # is the per-request context length (== context_lens).
    topk_cu_seqlens_q: torch.Tensor | None = None
    topk_prefix_lens: torch.Tensor | None = None
    topk_max_query_len: int = 0
    # Per-token causal page count cdiv(seq_pos+1, PAGE_SIZE), [total_q] int32:
    # drives sparse_topk_select force_end_blocks + -1 out-of-range clamp.
    topk_num_valid_pages: torch.Tensor | None = None
    # Sharded index-K: slots of this rank's index-K rows (-1 elsewhere), the
    # fused insert kernel's index slot mapping (stable cudagraph buffer view).
    index_slot_mapping: torch.Tensor | None = None
    # Records of the decode score kernel's top-k epilogue (None: the unified
    # sparse_topk_select also selects the decode rows).
    fused_select_ws: DecodeTopKWorkspace | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Decode metadata is the cudagraph-safe Triton decode metadata; the prefill
    fmha plan is built eagerly (prefill batches are not captured)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Persistent unified score buffer [T, H, MAX_K_TILES] shared by all
        # indexer layers and reused across forwards. Stable address (required
        # for the captured decode path) + fixed tile dim so the decode score
        # kernel's shape is frozen at capture. Filled with -inf per forward in
        # build(); the valid/padding partition is a per-forward constant, so
        # every layer overwrites the same valid tiles before its own top-k.
        self.unified_scores_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.num_index_heads,
            MAX_K_TILES,
            dtype=torch.float32,
            device=device,
        )
        # Mixed-step timer (opt-in) and prefill KV-split plan (on by default);
        # both are configured through additional_config.
        self.mixed_timer = mixed_attn_timer.maybe_init(vllm_config)
        # TP-sharded index-K cache (encoded in the spec's packed page size).
        self.idxk_tp = idxk_shard.shard_factor_from_spec(kv_cache_spec)
        self.idxk_rank = 0
        self.idxk_slots: torch.Tensor | None = None
        if self.idxk_tp > 1:
            self.idxk_rank = get_tensor_model_parallel_rank()
            max_capture = vllm_config.compilation_config.max_cudagraph_capture_size or 0
            self.idxk_slots = torch.full(
                (
                    max(
                        vllm_config.scheduler_config.max_num_batched_tokens, max_capture
                    ),
                ),
                -1,
                dtype=torch.int64,
                device=device,
            )
        self.kv_split: indexer_kv_split.PrefillKVSplitPlanner | None = None
        self.kv_split_verify_left = 0
        kv_split_cfg = indexer_kv_split.parse_config(vllm_config)
        if kv_split_cfg is not None:
            self.kv_split = indexer_kv_split.PrefillKVSplitPlanner(
                kv_split_cfg, self.num_index_heads, PAGE_SIZE, MAX_K_TILES, device
            )
            # Only the opt-in self-test and verify need the TP rank.
            is_rank0 = (kv_split_cfg.selftest or kv_split_cfg.verify > 0) and (
                get_tensor_model_parallel_rank() == 0
            )
            if kv_split_cfg.selftest:
                text_config = vllm_config.model_config.hf_text_config
                sparse_cfg = text_config.sparse_attention_config
                index_dtype = kv_cache_spec.dtype
                if index_dtype == torch.uint8:
                    index_dtype = torch.float8_e4m3fn
                self.kv_split.selftest(
                    head_dim=kv_cache_spec.head_size,
                    dtype=index_dtype,
                    force_begin_blocks=sparse_cfg.get("sparse_init_block", 0),
                    force_end_blocks=sparse_cfg.get("sparse_local_block", 0),
                    emit=is_rank0,
                )
            if is_rank0 and self.idxk_tp == 1:
                self.kv_split_verify_left = kv_split_cfg.verify
        # Decode top-k from score records: replicated cache + CuteDSL decode
        # score only.
        self.fused_select_ws: DecodeTopKWorkspace | None = None
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        if (
            self.idxk_tp == 1
            and self.num_index_heads * self.max_decode_query_len <= 32
            and _fused_select_enabled(max_num_seqs)
        ):
            self.fused_select_ws = DecodeTopKWorkspace(max_num_seqs, device)
            logger.info_once(
                "MiniMax M3 MSA indexer: decode top-16 selected from index decode "
                "score records (%d heads x %d queries, %d requests)",
                self.num_index_heads,
                self.max_decode_query_len,
                max_num_seqs,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        query_start_loc = common_attn_metadata.query_start_loc

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Sharded index-K: this rank's insert slots, into the stable buffer.
        index_slot_mapping = None
        if self.idxk_slots is not None:
            index_slot_mapping = idxk_shard.shard_slot_mapping(
                common_attn_metadata.slot_mapping,
                self.idxk_rank,
                self.idxk_tp,
                self.idxk_slots,
            )

        # Context (prefix) lengths into the stable cudagraph buffer.
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        # Per-token causal page count for the top-k, into the stable cg buffer.
        positions = common_attn_metadata.positions
        assert positions is not None
        num_valid_pages = self.num_valid_pages_buffer[:num_tokens]
        num_valid_pages.copy_(positions[:num_tokens] // PAGE_SIZE + 1)

        # Unified score buffer: a per-forward view of the persistent buffer,
        # reset to the -inf sentinel once here and shared by every layer (the
        # tile dim is the capture-time constant MAX_K_TILES).
        max_k_tiles = MAX_K_TILES
        unified_scores = self.unified_scores_buffer[:num_tokens]
        unified_scores.fill_(_SCORE_SENTINEL)

        decode: MiniMaxM3IndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            decode = MiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        prefill: MiniMaxM3IndexerMSAPrefillMetadata | None = None
        if num_prefills > 0:
            # Prefill is eager (not captured); the host lengths it needs (and the
            # _fmha_sm100_plan .tolist() inside) make the D->H sync acceptable.
            from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

            lo, hi = num_decodes, num_reqs
            qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
            qo_lens_cpu = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
            kv_lens_cpu = seq_lens[:num_reqs].cpu().to(torch.int32)
            nvp = (kv_lens_cpu + PAGE_SIZE - 1) // PAGE_SIZE
            side_qo = qo_lens_cpu[lo:hi]
            side_kv = kv_lens_cpu[lo:hi]
            # Sharded index-K: gather the prefill requests' pages (request
            # order) once per layer; the page tables index that list.
            gather_pages = row_offsets = None
            if self.idxk_tp > 1:
                side_pages = nvp[lo:hi].to(torch.int64)
                row_offsets = torch.cumsum(side_pages, 0) - side_pages
                rows = torch.repeat_interleave(
                    torch.arange(side_pages.numel()), side_pages
                )
                cols = torch.arange(rows.numel()) - torch.repeat_interleave(
                    row_offsets, side_pages
                )
                table = block_table[lo:hi].contiguous()
                gather_pages = (
                    table.view(-1)
                    .index_select(
                        0,
                        (rows * table.shape[1] + cols).to(
                            table.device, non_blocking=True
                        ),
                    )
                    .to(torch.int64)
                )
            # Default on: <=128-token slices planned with num_kv_splits > 1.
            split = (
                None
                if self.kv_split is None
                else self.kv_split.plan(
                    side_qo, side_kv, block_table[lo:hi], row_offsets=row_offsets
                )
            )
            if split is not None:
                plan, page_table, kv_split_info = split
            else:
                kv_split_info = {"kvs_splits": 0}
                plan = _fmha_sm100_plan(
                    side_qo,
                    side_kv,
                    self.num_index_heads,
                    num_kv_heads=1,
                    qo_offset=side_kv - side_qo,  # bottom-right causal
                    page_size=PAGE_SIZE,
                    output_maxscore=True,
                    causal=True,
                    num_kv_splits=1,
                )
                # Force the plan's tile dim to the unified buffer's so prefill
                # writes its max_score straight into unified[:, :, nd:] (the
                # stride-aware binding shape-matches the tile dim exactly).
                # max_k_tiles >= the plan's natural value, so the extra tiles
                # are simply never written.
                plan["max_k_tiles"] = max_k_tiles
                if gather_pages is not None:
                    page_table = torch.arange(
                        gather_pages.numel(),
                        dtype=torch.int32,
                        device=block_table.device,
                    )
                else:
                    cols = torch.arange(block_table.shape[1], device=block_table.device)
                    valid = cols[None, :] < nvp[lo:hi].to(block_table.device)[:, None]
                    page_table = block_table[lo:hi][valid].to(torch.int32)
            verify = None
            if (
                split is not None
                and self.kv_split_verify_left > 0
                and num_prefill_tokens <= indexer_kv_split.VERIFY_MAX_TOKENS
            ):
                self.kv_split_verify_left -= 1
                verify = indexer_kv_split.stock_prefill_plan(
                    side_qo,
                    side_kv,
                    self.num_index_heads,
                    PAGE_SIZE,
                    max_k_tiles,
                    block_table[lo:hi],
                )
            prefill = MiniMaxM3IndexerMSAPrefillMetadata(
                plan=plan,
                cu_seqlens_q=(query_start_loc[lo : hi + 1] - query_start_loc[lo]).to(
                    torch.int32
                ),
                prefix_lens=context_lens[lo:hi],
                max_query_len=int(side_qo.max()),
                page_table=page_table,
                verify=verify,
                gather_pages=gather_pages,
            )
            if self.mixed_timer is not None:
                prefix = side_kv - side_qo
                self.mixed_timer.arm(
                    nt=num_tokens,
                    nd=num_decode_tokens,
                    np=num_prefills,
                    npt=num_prefill_tokens,
                    ctx_max=int(prefix.max()),
                    ctx_sum=int(prefix.sum()),
                    kvs_verify=int(verify is not None),
                    **kv_split_info,
                )

        return MiniMaxM3IndexerMSAMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            decode=decode,
            prefill_msa=prefill,
            unified_scores=unified_scores,
            max_k_tiles=max_k_tiles,
            topk_cu_seqlens_q=query_start_loc[: num_reqs + 1],
            topk_prefix_lens=context_lens,
            topk_max_query_len=common_attn_metadata.max_query_len,
            topk_num_valid_pages=num_valid_pages,
            index_slot_mapping=index_slot_mapping,
            fused_select_ws=self.fused_select_ws,
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Decode: CuteDSL/Triton score. Prefill: fmha_sm100 OnlyScore + top-k."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.idxk_ordinal = -1
        if self.index_cache.tp_shard > 1:
            comm = idxk_shard.get_comm()
            assert comm is not None
            self.idxk_ordinal = comm.register_layer()
            self.idxk_rank = comm.rank

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)

        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        sharded = self.index_cache.tp_shard > 1
        # Sharded: every index head (this rank's are [rank * H, (rank + 1) * H)).
        index_q = index_query[:num_tokens].view(-1, self.index_head_dim)
        index_q = index_q.view(num_tokens, -1, self.index_head_dim)
        kv = self.index_cache.kv_cache
        # Shared persistent top-k output buffer; the unified top-k below writes
        # the selected block ids into buf[:num_tokens].
        buf = self.topk_indices_buffer
        assert buf is not None

        # Unified token-major score buffer [total_q, H, MAX_K_TILES]: the tile
        # dim is innermost/contiguous, so both fmha writes (native [T,H,K]) and
        # the block-iterating top-k reads hit contiguous tiles. Each side gets a
        # contiguous slice on dim 0: decode [:nd], prefill [nd:]; the kernels
        # read/write by strides. The builder allocates it once (persistent,
        # shared by all layers) and resets it to the sentinel each forward, so
        # the top-k never picks an unwritten tile.
        unified_scores = md.unified_scores
        assert unified_scores is not None

        # Decode scores -> unified[:nd] (transposed [H, nd, MK] view; the kernel
        # writes by strides). Top-k is deferred to the single unified call below
        # unless the score records already selected the decode rows.
        decode_selected = False
        if md.decode is not None:
            timer_token = mixed_attn_timer.span_start()
            d = md.decode
            # max_decode_query_len avoids recompiles across runtime decode sizes.
            # Fall back when the flattened Q tile gets too wide for this kernel.
            if sharded:
                # Every head over this rank's token slice, then a cross-rank
                # max of the per-block partial maxima (exact).
                idxk_shard.decode_score(
                    index_q[:nd],
                    kv,
                    d.block_table,
                    d.seq_lens,
                    d.decode_query_len,
                    d.max_decode_query_len,
                    unified_scores[:nd],
                    self.idxk_ordinal,
                )
            elif (
                md.fused_select_ws is not None
                and self.num_index_heads * d.max_decode_query_len <= 32
                and d.block_table.shape[0] <= md.fused_select_ws.max_batch
            ):
                minimax_m3_index_decode_score_topk_cutedsl(
                    index_q[:nd],
                    kv,
                    d.block_table,
                    d.seq_lens,
                    d.decode_query_len,
                    d.max_decode_query_len,
                    unified_scores[:nd].transpose(0, 1),
                    md.topk_num_valid_pages[:nd],
                    buf[:nd],
                    md.fused_select_ws,
                    self.init_blocks,
                    self.local_blocks,
                )
                decode_selected = True
            else:
                decode_score = (
                    minimax_m3_index_decode_score_cutedsl
                    if self.num_index_heads * d.max_decode_query_len <= 32
                    else minimax_m3_index_decode_score
                )
                decode_score(
                    index_q[:nd],
                    kv,
                    d.block_table,
                    d.seq_lens,
                    d.max_seq_len,
                    self.init_blocks,
                    self.local_blocks,
                    self.num_kv_heads,
                    d.decode_query_len,
                    d.max_decode_query_len,
                    score_out=unified_scores[:nd].transpose(0, 1),
                )
            mixed_attn_timer.span_end("idec", timer_token)

        if md.prefill_msa is not None:
            from vllm.third_party.fmha_sm100.api import _fmha_sm100

            p = md.prefill_msa
            prefill_q = index_q[nd:]
            if p.gather_pages is not None:
                # Sharded index-K: full pages of the prefill requests, scored
                # for this rank's heads only.
                timer_token = mixed_attn_timer.span_start()
                kv = idxk_shard.gather_prefill_pages(kv, p.gather_pages)
                h = self.num_index_heads
                prefill_q = prefill_q[:, self.idxk_rank * h : (self.idxk_rank + 1) * h]
                prefill_q = prefill_q.contiguous()
                mixed_attn_timer.span_end("igat", timer_token)
            # Index-K cache (num_blocks, 128, D) -> paged MQA (num_blocks,1,128,D).
            k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
            # fmha writes its max_score natively as [nnz_p, H, max_k_tiles] into
            # the prefill region (stride-aware; plan max_k_tiles forced to
            # md.max_k_tiles so the shape matches exactly -> no copy).
            timer_token = mixed_attn_timer.span_start()
            _fmha_sm100(
                prefill_q,
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                p.plan,
                kv_indices=p.page_table,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
                max_score=unified_scores[nd:],
            )
            mixed_attn_timer.span_end("isc", timer_token)

        # Single top-k over the unified buffer via fmha_sm100 sparse_topk_select
        # (THK, no transpose) into ``buf``. num_valid_pages drives force_end_blocks
        # (always keep each token's local block; fmha OnlyScore won't) + the -1
        # out-of-range clamp.
        from vllm.third_party.fmha_sm100.api import sparse_topk_select

        # Rows still to select: all, or the prefill rows after a fused decode.
        lo = nd if decode_selected else 0
        if lo < num_tokens:
            timer_token = mixed_attn_timer.span_start()
            sparse_topk_select(
                unified_scores[lo:num_tokens],
                self.topk_blocks,
                num_valid_pages=md.topk_num_valid_pages[lo:num_tokens],
                force_begin_blocks=self.init_blocks,
                force_end_blocks=self.local_blocks,
                output=buf[lo:num_tokens],
                max_score_layout="THK",
            )
            mixed_attn_timer.span_end("itk", timer_token)

        if md.prefill_msa is not None and md.prefill_msa.verify is not None:
            self._verify_kv_split(md, index_q[nd:], kv, unified_scores[nd:], buf)

        # The attend reads ``buf`` directly; this return is vestigial.
        return None, None

    def _verify_kv_split(
        self,
        md: MiniMaxM3IndexerMSAMetadata,
        prefill_q: torch.Tensor,
        kv: torch.Tensor,
        split_scores: torch.Tensor,
        buf: torch.Tensor,
    ) -> None:
        """Re-score the prefill rows with the stock plan and compare the
        scores and top-k with the KV-split plan's (opt-in, first layer only)."""
        p = md.prefill_msa
        assert p is not None and p.verify is not None
        stock_plan, stock_table = p.verify
        p.verify = None
        nd, num_tokens = md.num_decode_tokens, md.num_actual_tokens
        k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
        stock_scores = indexer_kv_split.score_prefill(
            prefill_q, k_pages, stock_plan, stock_table, self.scale, md.max_k_tiles
        )
        num_valid_pages = md.topk_num_valid_pages[nd:num_tokens]
        stock_topk = indexer_kv_split.topk_select(
            stock_scores,
            num_valid_pages,
            self.init_blocks,
            self.local_blocks,
            self.topk_blocks,
        )
        valid = (
            torch.arange(md.max_k_tiles, device=split_scores.device)[None, None, :]
            < num_valid_pages[:, None, None]
        )
        differ = stock_scores != split_scores
        indexer_kv_split.write_record(
            "ARIA_MSA_KVSPLIT_VERIFY",
            {
                "layer": self.index_cache.prefix,
                "npt": num_tokens - nd,
                "nd": nd,
                "score_mismatch_valid": int((differ & valid).sum()),
                "score_mismatch_all": int(differ.sum()),
                "finite_valid": int((torch.isfinite(split_scores) & valid).sum()),
                "valid": int(valid.sum()),
                "topk_mismatch_rows": int(
                    (stock_topk != buf[nd:num_tokens]).any(dim=-1).any(dim=-1).sum()
                ),
            },
        )
