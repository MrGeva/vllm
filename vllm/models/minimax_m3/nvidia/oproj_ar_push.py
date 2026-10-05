# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push the MiniMax-M3 o_proj output straight into the post-attention
all-reduce + residual add + GemmaRMSNorm.

Stock decode chain per decoder layer at TP > 1:

    ... attention -> o_proj (small-M MXFP8 GEMM; bf16 partial to HBM) ->
    FlashInfer MNNVL one-shot AR + residual + RMSNorm (waits for the whole
    GEMM grid, reloads the partial, multicasts it, polls for the peers)

Pushed chain, for decode-sized calls inside CUDA graphs:

    ... attention -> o_proj GEMM whose epilogue multicasts its bf16 output
    into a Lamport mailbox -> consumer kernel (launched once every GEMM CTA
    has read the mailbox generation) that polls the mailbox, sums the ranks,
    adds the residual and normalizes (_oproj_ar_push_kernel.py)

The transfer overlaps the GEMM tail and the consumer starts on arrival rather
than on grid completion. The new residual is bitwise equal to the stock
chain's; the normed output differs only in the fp32 order of the sum of
squares.

Every cross-rank wait is bounded: a consumer whose peer data has not arrived
after VLLM_MM3_OPROJ_AR_PUSH_TIMEOUT_MS completes with zeros for the missing
words, poisons later calls (they stop waiting) and counts the event in
host-mapped memory; a watchdog thread then logs it and, by default, exits
the process instead of serving wrong results.

Only calls captured into CUDA graphs are pushed: replays run in lockstep on
every rank, while eager calls (profiling, warmup, autotuning) can drift by
seconds. Kernels are compiled and warm-launched collectively at model
construction, so no module loads lazily next to a spinning peer.

Knobs, read once at import:
    VLLM_MM3_OPROJ_AR_PUSH             "1" enables the push (default "0").
    VLLM_MM3_OPROJ_AR_PUSH_MAX_TOKENS  largest pushed call (default 16).
    VLLM_MM3_OPROJ_AR_PUSH_TIMEOUT_MS  bound on every wait (default 10000).
"""

import ctypes
import os
import threading
import time

import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8 import small_m

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_MM3_OPROJ_AR_PUSH", "0") == "1"
MAX_TOKENS = int(os.environ.get("VLLM_MM3_OPROJ_AR_PUSH_MAX_TOKENS", "16"))
TIMEOUT_MS = float(os.environ.get("VLLM_MM3_OPROJ_AR_PUSH_TIMEOUT_MS", "10000"))

# Consumer geometry: clusters of 8 CTAs x 96 threads cover hidden 6144 as 768
# 16-byte fragments in one trip.
CLUSTER_SIZE = 8
THREADS = 96
# Test hooks: push eager calls too; wait for the GEMM grid before polling;
# exit the process when a wait times out.
FORCE = False
WAIT_FOR_GEMM = False
ABORT_ON_TIMEOUT = True

_SUPPORTED_TP_SIZES = (2, 4, 8)

_workspace = None


class _Workspace:
    def __init__(self, hidden: int, in_features: int, eps: float) -> None:
        from flashinfer.comm.mnnvl_cutedsl.symmetric_buffer import SymmetricBuffer

        from vllm.model_executor.kernels.linear.mxfp8._small_m_kernel import (
            PUSH_GENERATIONS,
            PUSH_SENTINEL,
            PushSpec,
        )
        from vllm.models.minimax_m3.nvidia._oproj_ar_push_kernel import STAGE_WORDS

        self.hidden = hidden
        self.in_features = in_features
        self.eps = eps
        self.tp = get_tensor_model_parallel_world_size()
        self.push = PushSpec(self.tp, get_tensor_model_parallel_rank(), MAX_TOKENS)
        device = torch.device("cuda", torch.cuda.current_device())
        self.mailbox = SymmetricBuffer.allocate(
            (PUSH_GENERATIONS, self.tp, MAX_TOKENS, hidden),
            torch.bfloat16,
            device,
            get_tp_group().device_group,
            require_multicast=True,
        )
        self.mailbox.tensor.view(torch.int16).fill_(PUSH_SENTINEL - (1 << 16))
        self.multicast_address = int(self.mailbox.multicast_address)
        self.stage = torch.zeros((STAGE_WORDS,), dtype=torch.int32, device=device)
        self.errors_host, self.errors_device = _host_mapped_counter()
        self.timeout_ns = int(TIMEOUT_MS * 1e6)
        self._consumers: dict[bool, object] = {}
        torch.cuda.synchronize()

    def consumer(self):
        fn = self._consumers.get(WAIT_FOR_GEMM)
        if fn is None:
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import make_fake_stream

            from vllm.models.minimax_m3.nvidia._oproj_ar_push_kernel import (
                OprojArPushConsumer,
            )

            kernel = OprojArPushConsumer(
                hidden=self.hidden,
                tp=self.tp,
                capacity_m=MAX_TOKENS,
                cluster_size=CLUSTER_SIZE,
                threads=THREADS,
                eps=self.eps,
                weight_bias=1.0,
                wait_for_gemm=WAIT_FOR_GEMM,
            )
            fn = cute.compile(
                kernel,
                *([cutlass.Int64(0)] * 8),
                cutlass.Int32(1),
                make_fake_stream(),
            )
            self._consumers[WAIT_FOR_GEMM] = fn
        return fn

    def errors(self) -> int:
        return ctypes.c_uint32.from_address(self.errors_host).value

    def run(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        normed: torch.Tensor,
        new_residual: torch.Tensor,
    ) -> None:
        import cutlass
        from cuda.bindings.driver import CUstream

        from vllm.utils.torch_utils import current_stream

        small_m.small_m_gemm_push(
            x,
            weight,
            weight_scale,
            self.push,
            self.multicast_address,
            self.stage.data_ptr(),
        )
        self.consumer()(
            cutlass.Int64(self.mailbox.tensor.data_ptr()),
            cutlass.Int64(residual.data_ptr()),
            cutlass.Int64(gamma.data_ptr()),
            cutlass.Int64(new_residual.data_ptr()),
            cutlass.Int64(normed.data_ptr()),
            cutlass.Int64(self.stage.data_ptr()),
            cutlass.Int64(self.errors_device),
            cutlass.Int64(self.timeout_ns),
            cutlass.Int32(x.shape[0]),
            CUstream(current_stream().cuda_stream),
        )

    def warm_up(self) -> None:
        """Compile and run every GEMM tile config of 1..MAX_TOKENS tokens and
        the consumer once, on every rank together."""
        device = self.stage.device
        n, k = self.hidden, self.in_features
        weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device).t()
        weight_scale = torch.full((n * k // 32,), 127, dtype=torch.uint8, device=device)
        gamma = torch.zeros((n,), dtype=torch.bfloat16, device=device)
        sizes: dict = {}
        for m in range(1, MAX_TOKENS + 1):
            sizes.setdefault(small_m.select_config(m, n, k), m)
        for cfg in sizes:
            small_m._compile(cfg, True, self.push)
        self.consumer()
        torch.distributed.barrier(group=get_tp_group().cpu_group)
        for m in sizes.values():
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
            residual = torch.zeros((m, n), dtype=torch.bfloat16, device=device)
            normed, new_residual = (
                torch.empty_like(residual),
                torch.empty_like(residual),
            )
            self.run(x, weight, weight_scale, residual, gamma, normed, new_residual)
        torch.cuda.synchronize()
        if self.errors():
            raise RuntimeError("MiniMax-M3 o_proj AR push: warm-up timed out")


def _host_mapped_counter() -> tuple[int, int]:
    """A zeroed pinned host word the device can add to; (host, device) address."""
    from cuda.bindings import runtime as cudart

    err, host = cudart.cudaHostAlloc(64, cudart.cudaHostAllocMapped)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostAlloc failed: {err}")
    ctypes.memset(host, 0, 64)
    err, device = cudart.cudaHostGetDevicePointer(host, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
    return int(host), int(device)


def _watch(workspace: _Workspace) -> None:
    reported = 0
    while True:
        time.sleep(1.0)
        errors = workspace.errors()
        if errors == reported:
            continue
        reported = errors
        logger.critical(
            "MiniMax-M3 o_proj AR push: %d consumer CTA(s) timed out after %.0f ms "
            "waiting for a peer rank; results are invalid.",
            errors,
            TIMEOUT_MS,
        )
        if ABORT_ON_TIMEOUT:
            os._exit(70)


def init_workspace(hidden: int, in_features: int, eps: float) -> bool:
    """Build the mailbox, compile and warm-launch the kernels. Collective over
    the TP group (every rank calls it once, at model construction). Returns
    whether the push is on."""
    global _workspace
    if _workspace is not None:
        return True
    tp = get_tensor_model_parallel_world_size()
    if (
        not ENABLED
        or tp not in _SUPPORTED_TP_SIZES
        or not 1 <= MAX_TOKENS <= small_m.SMALL_M_MAX
        or not small_m.is_available()
        or not small_m.supports_shape(hidden, in_features)
    ):
        return False
    from flashinfer.comm.mnnvl import is_multicast_supported

    if not is_multicast_supported(torch.accelerator.current_device_index()):
        logger.warning_once("MiniMax-M3 o_proj AR push off: no NVLink multicast.")
        return False
    workspace = _Workspace(hidden, in_features, eps)
    workspace.warm_up()
    threading.Thread(target=_watch, args=(workspace,), daemon=True).start()
    _workspace = workspace
    logger.info_once(
        "MiniMax-M3 o_proj AR push: o_proj epilogue multicasts into the TP%d "
        "AR + RMSNorm for captured calls up to %d tokens (timeout %.0f ms).",
        tp,
        MAX_TOKENS,
        TIMEOUT_MS,
    )
    return True


def applies(o_proj: torch.nn.Module, num_tokens: int) -> bool:
    """Whether this call is pushed. Depends only on facts every rank shares."""
    return (
        _workspace is not None
        and 1 <= num_tokens <= MAX_TOKENS
        and getattr(o_proj, "mxfp8_small_m_fused", False)
        and (FORCE or torch.cuda.is_current_stream_capturing())
    )


def o_proj_ar_rmsnorm(
    o_proj: torch.nn.Module,
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(o_proj(attn_output)), residual)`` with the o_proj
    output pushed into the all-reduce. Returns ``(normed, new_residual)``."""
    ws = _workspace
    assert ws is not None, "init_workspace was not called"
    x = attn_output
    m, hidden = x.shape[0], ws.hidden
    # Asserted, not a fallback: a per-rank fallback would desynchronize ranks.
    assert small_m._small_m_ok(x, o_proj.weight) and x.shape[1] == ws.in_features
    assert residual.is_contiguous() and residual.shape == (m, hidden)
    assert residual.dtype == torch.bfloat16 and norm.variance_epsilon == ws.eps
    assert norm.weight.dtype == torch.bfloat16 and norm.weight.is_contiguous()
    if torch.cuda.is_current_stream_capturing():
        logger.info_once("MiniMax-M3 o_proj AR push: captured at %d tokens.", m)
    normed = torch.empty_like(residual)
    new_residual = torch.empty_like(residual)
    ws.run(
        x,
        o_proj.weight,
        o_proj.weight_scale,
        residual,
        norm.weight,
        normed,
        new_residual,
    )
    return normed, new_residual
