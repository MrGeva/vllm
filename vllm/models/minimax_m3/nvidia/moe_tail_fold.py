# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold the MiniMax-M3 MoE tail into the next TP all-reduce + RMSNorm.

Stock decode chain per MoE layer at TP > 1 (shared expert on the aux stream):

    ... FC2 -> top-k finalize -> wait for the shared expert -> shared + routed
    add -> next layer's one-shot AR + residual add + GemmaRMSNorm

Folded chain, for decode-sized calls:

    ... FC2 (finalize deferred) -> wait for the shared expert -> FlashInfer's
    MNNVL CuTe DSL low-latency MoE-finalize all-reduce

The fused tail is two PDL-chained kernels. The publish kernel sums the top-k
GEMM2 rows with their routing weights and the shared-expert partial in fp32,
rounds the per-rank partial to bf16 once and multicasts it into a Lamport
mailbox. The consumer sums the ranks in rank order, adds the residual and
applies the GemmaRMSNorm. This drops the finalize and add kernels and the
non-PDL boundary around the add.

Reduction orders are fixed (top-k slot order, then rank order), so the result
is run-to-run deterministic and identical on every rank. It is not bitwise
equal to the stock chain, which rounds the routed sum before the shared add.

Knobs, read once at import:
    VLLM_MM3_MOE_TAIL_FOLD             "1" enables the fold (default "0").
    VLLM_MM3_MOE_TAIL_FOLD_MAX_TOKENS  largest folded call (default 16).
    VLLM_MM3_MOE_TAIL_FOLD_TUNING      LL kernel geometry (see TUNINGS).
"""

import os

import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_MM3_MOE_TAIL_FOLD", "0") == "1"
MAX_TOKENS = int(os.environ.get("VLLM_MM3_MOE_TAIL_FOLD_MAX_TOKENS", "16"))
TUNING = os.environ.get("VLLM_MM3_MOE_TAIL_FOLD_TUNING", "c8t96")

# name -> (publish threads, consumer cluster size, consumer threads). The
# publish kernel covers 4 bf16 per thread; the consumer covers 8 per thread
# with one cluster of CTAs per token (hidden 6144 = 768 fragments of 8).
TUNINGS = {
    "c8t96": (128, 8, 96),
    "c8t128": (128, 8, 128),
    "c6t128": (128, 6, 128),
    "c16t64": (128, 16, 64),
    "c4t192": (128, 4, 192),
    "p64c8t96": (64, 8, 96),
}

_SUPPORTED_TP_SIZES = (2, 4, 8, 16)

_workspace = None


def _make_config(tp_size: int, hidden_size: int, top_k: int, tuning: str):
    from flashinfer.comm.mnnvl_cutedsl import (
        KernelTarget,
        MNNVLCuteDSLConfig,
        MRangeDispatch,
        ProtocolKind,
        StaticProfile,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
        LLAllReduceTuning,
        LLCollectiveTuning,
        LLFinalizeTuning,
    )

    publish_threads, cluster_size, threads = TUNINGS[tuning]
    collective = LLCollectiveTuning(
        cluster_size=cluster_size, rank_lanes=1, threads=threads, enable_pdl=True
    )
    finalize = LLFinalizeTuning(
        elements_per_thread=4,
        threads=publish_threads,
        prefetch_group=top_k,
        load_shared_expert_before_pdl=False,
        collective=collective,
    )
    # The profile must also route plain all-reduces; the fold never calls them.
    all_reduce = LLAllReduceTuning(collective=collective)
    profile = StaticProfile(
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=torch.bfloat16,
        finalize_routes=MRangeDispatch(
            upper_bounds=(None,),
            targets=(KernelTarget(ProtocolKind.LL, finalize),),
        ),
        all_reduce_routes=MRangeDispatch(
            upper_bounds=(None,),
            targets=(KernelTarget(ProtocolKind.LL, all_reduce),),
        ),
    )
    return MNNVLCuteDSLConfig(profiles=(profile,))


def defer_finalize(experts: torch.nn.Module) -> bool:
    """Ask ``experts`` (a MiniMax-M3 ``MoERunner``) to leave its top-k
    finalize to the next RMSNorm, when the fold is enabled and applies.

    Requires the monolithic TRT-LLM NVFP4 experts (the ``do_finalize=False``
    path), a TP-only deployment of a supported size whose MoE output is left
    un-reduced, bf16 activations and NVLink multicast. Every rank decides the
    same way, at model construction.
    """
    if not ENABLED:
        return False
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    config = experts.moe_config
    experts_cls = getattr(experts._quant_method, "experts_cls", None)
    if (
        experts_cls is not TrtLlmNvFp4ExpertsMonolithic
        or experts.shared_experts is None
        or experts.routed_scaling_factor != 1.0
        or not config.skip_final_all_reduce
        or config.tp_size not in _SUPPORTED_TP_SIZES
        or config.in_dtype != torch.bfloat16
        or MAX_TOKENS <= 0
        or TUNING not in TUNINGS
    ):
        return False
    from flashinfer.comm.mnnvl import is_multicast_supported

    if not is_multicast_supported(torch.accelerator.current_device_index()):
        logger.warning_once("MiniMax-M3 MoE tail fold off: no NVLink multicast.")
        return False
    # Deferred calls run the GEMM tactics tuned for finalized calls. Tuned on
    # their own, they can flip to a near-tie tactic that starves the
    # aux-stream shared expert of SMs (served TP2 c1 K=5: ~14 us/layer later).
    config.defer_reuses_finalized_tactics = True
    config.defer_moe_finalize(MAX_TOKENS)
    return config.use_deferred_moe_finalize


def keep_finalize(experts: torch.nn.Module) -> None:
    """Make ``experts`` finalize every call again."""
    experts.moe_config.limit_deferred_moe_finalize(0)


def init_workspace(hidden_size: int, top_k: int, rms_eps: float) -> None:
    """Build the fused tail's Lamport mailbox and compile its kernels.

    Collective over the TP group: every rank calls it once, at model
    construction, before any forward.
    """
    global _workspace
    if _workspace is not None:
        return
    from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

    tp_size = get_tensor_model_parallel_world_size()
    _workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        tp_size,
        get_tensor_model_parallel_rank(),
        MAX_TOKENS,
        hidden_size,
        torch.bfloat16,
        group=get_tp_group().device_group,
        top_k=top_k,
        rms_eps=rms_eps,
        routed_scaling_factor=1.0,
        weight_bias=1.0,
        include_shared_expert=True,
        add_residual=True,
        write_residual_output=True,
        config=_make_config(tp_size, hidden_size, top_k, TUNING),
    )
    logger.info_once(
        "MiniMax-M3 MoE tail fold: deferred top-k finalize + shared add folded "
        "into the TP%d all-reduce + RMSNorm for up to %d tokens (LL %s).",
        tp_size,
        MAX_TOKENS,
        TUNING,
    )


def finalize_ar_rmsnorm(
    moe_output: MoEOutput,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k finalize + shared-expert add + all-reduce + residual add +
    GemmaRMSNorm of a folded MoE output. Returns ``(normed, new_residual)``."""
    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

    routed = moe_output.routed
    assert isinstance(routed, UnfinalizedMoEOutput)
    assert moe_output.shared_output is not None
    assert moe_output.routed_scaling_factor == 1.0
    assert _workspace is not None, "init_workspace was not called"
    normed = torch.empty_like(residual)
    new_residual = torch.empty_like(residual)
    allreduce_fusion(
        input=routed.gemm2_permuted,
        workspace=_workspace,
        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        launch_with_pdl=True,
        residual_out=new_residual,
        norm_out=normed,
        residual_in=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.variance_epsilon,
        expanded_idx_to_permuted_idx=routed.expanded_idx_to_permuted_idx,
        expert_scale_factor=routed.expert_weights,
        shared_expert_output=moe_output.shared_output,
        weight_bias=1.0,
    )
    return normed, new_residual
