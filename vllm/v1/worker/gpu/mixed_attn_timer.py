# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU attribution of mixed (prefill) steps' eager attention regions.

Enable with ``--additional-config '{"aria_mixed_attn_timer": 1}'``; an integer
N > 1 times every Nth mixed step. A model's attention metadata builder arms the
step when the batch has prefill rows (MiniMax-M3's MSA indexer builder does).
The model runner then brackets the target model forward with CUDA events, and
the eager attention regions inside it (which run in Python even when the rest
of the forward replays a breakable piecewise CUDA graph) record spans: for
MiniMax-M3 the indexer (decode score, prefill score, unified top-k), the sparse
attend (decode, k2q CSR build, prefill sparse attention) and the dense layers'
``unified_attention_with_output``. Events are read only after ``query()``
reports the forward's last event complete, so the timer never blocks on the
GPU. TP rank 0 writes one ``ARIA_MIXED_ATTN {json}`` line per timed step
(fields are documented in ``_emit``). Disabled, every hook is a ``None`` check.
"""

import json
import os
import sys
import time
from collections import deque
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

CONFIG_KEY = "aria_mixed_attn_timer"

_MARKER = "ARIA_MIXED_ATTN"
_READY_MARKER = "ARIA_MIXED_ATTN_READY"
# Span categories, in emission order.
_CATEGORIES = (
    "srun",  # whole sparse-layer eager region (indexer + attend)
    "idec",  # indexer decode-row score
    "isc",  # indexer prefill score (fmha_sm100 OnlyScore)
    "itk",  # unified top-k select
    "adec",  # sparse attend, decode rows
    "k2q",  # sparse attend, k2q CSR + schedule build
    "spa",  # sparse attend, prefill sparse_atten_func
    "dat",  # dense-layer attention (unified_attention_with_output)
)
_NUM_SLOTS = 4

_TIMER: "MixedAttnTimer | None" = None


def additional_config_value(vllm_config: VllmConfig, key: str, default: Any) -> Any:
    additional_config = vllm_config.additional_config
    if isinstance(additional_config, dict):
        return additional_config.get(key, default)
    return default


class _StepRecord:
    __slots__ = ("start", "end", "events", "spans", "info", "busy")

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # Grown on demand and reused across steps.
        self.events: list[torch.cuda.Event] = []
        self.spans: list[tuple[str, int, int]] = []
        self.info: dict[str, Any] = {}
        self.busy = False


class MixedAttnTimer:
    def __init__(self, every: int) -> None:
        self.every = max(1, every)
        self._slots: list[_StepRecord] | None = None
        self._pending: deque[_StepRecord] = deque()
        # Batch facts from the builder for the step about to run.
        self._armed: dict[str, Any] | None = None
        # Recording between begin_fwd() and end_fwd().
        self._cur: _StepRecord | None = None
        self._num_events = 0
        self._num_mixed = 0
        self._num_dropped = 0
        self._ready_emitted = False

    # Step lifecycle (host side).
    def disarm(self) -> None:
        """Called at the top of every execute_model, before metadata builds."""
        self._armed = None

    def arm(self, **info: Any) -> None:
        """Called by a metadata builder when the batch has prefill rows."""
        if torch.cuda.is_current_stream_capturing():
            return
        self._armed = {"wall": round(time.time(), 6), **info}

    def begin_fwd(self) -> None:
        """Called by the runner right before a real (non-dummy) target forward."""
        info, self._armed = self._armed, None
        if info is None or torch.cuda.is_current_stream_capturing():
            return
        self._num_mixed += 1
        if (self._num_mixed - 1) % self.every:
            return
        if self._slots is None:
            self._slots = [_StepRecord() for _ in range(_NUM_SLOTS)]
        self._poll()
        rec = next((slot for slot in self._slots if not slot.busy), None)
        if rec is None:
            # The GPU is >= _NUM_SLOTS timed mixed steps behind; skip this one.
            self._num_dropped += 1
            return
        rec.busy = True
        rec.spans.clear()
        rec.info = {"seq": self._num_mixed - 1, **info}
        self._num_events = 0
        self._cur = rec
        rec.start.record()

    def end_fwd(self) -> None:
        rec = self._cur
        if rec is None:
            return
        rec.end.record()
        self._cur = None
        self._pending.append(rec)
        self._poll()

    def span_start(self) -> int | None:
        rec = self._cur
        if rec is None:
            return None
        return self._record(rec)

    def span_end(self, category: str, token: int | None) -> None:
        rec = self._cur
        if rec is None or token is None:
            return
        rec.spans.append((category, token, self._record(rec)))

    def _record(self, rec: _StepRecord) -> int:
        idx = self._num_events
        if idx == len(rec.events):
            rec.events.append(torch.cuda.Event(enable_timing=True))
        rec.events[idx].record()
        self._num_events = idx + 1
        return idx

    # Non-blocking readout.
    def _poll(self) -> None:
        while self._pending and self._pending[0].end.query():
            rec = self._pending.popleft()
            self._emit(rec)
            rec.busy = False

    def _emit(self, rec: _StepRecord) -> None:
        """One line per timed mixed step.

        seq: timed-forward index among mixed steps on this rank; wall: host
        epoch seconds when the builder armed the step (join to
        ARIA_VLLM_ITERATION.wall_epoch_s). Builder facts: nt (actual tokens),
        nd (decode tokens), np/npt (prefill requests/tokens), ctx_max/ctx_sum
        (prefill context tokens), kvs_* (prefill KV-split plan; kvs_splits 0 is
        the stock plan), kvs_verify (step also ran the stock-plan A/B check).
        GPU ms: fwd (target model forward as launched by the runner), then per
        category the summed span time (``<cat>``) and span count (``n_<cat>``);
        eager = srun + dat.
        """
        if not self._ready_emitted:
            self._ready_emitted = True
            _write(_READY_MARKER, {"every": self.every, "categories": _CATEGORIES})
        payload: dict[str, Any] = dict(rec.info)
        payload["fwd"] = _round(rec.start.elapsed_time(rec.end))
        sums = dict.fromkeys(_CATEGORIES, 0.0)
        counts = dict.fromkeys(_CATEGORIES, 0)
        events = rec.events
        for category, start, end in rec.spans:
            sums[category] += events[start].elapsed_time(events[end])
            counts[category] += 1
        for category in _CATEGORIES:
            payload[category] = _round(sums[category])
            payload["n_" + category] = counts[category]
        payload["eager"] = _round(sums["srun"] + sums["dat"])
        if self._num_dropped:
            payload["dropped"] = self._num_dropped
        _write(_MARKER, payload)


def maybe_init(vllm_config: VllmConfig) -> "MixedAttnTimer | None":
    """Create the process-wide timer once (TP rank 0, last PP rank only)."""
    global _TIMER
    if _TIMER is not None:
        return _TIMER
    value = additional_config_value(vllm_config, CONFIG_KEY, False)
    if value is True:
        every = 1
    elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
        every = value
    else:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.distributed.parallel_state import get_pp_group

    if get_tensor_model_parallel_rank() != 0 or not get_pp_group().is_last_rank:
        return None
    _TIMER = MixedAttnTimer(every)
    logger.info("Mixed-step attention timer on (every %d)", every)
    return _TIMER


def get() -> "MixedAttnTimer | None":
    return _TIMER


def span_start() -> int | None:
    timer = _TIMER
    return None if timer is None else timer.span_start()


def span_end(category: str, token: int | None) -> None:
    if token is not None:
        _TIMER.span_end(category, token)  # type: ignore[union-attr]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _write(marker: str, payload: dict[str, Any]) -> None:
    record = (
        marker
        + " "
        + json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(record) > 4096:
        logger.warning_once("Dropping oversized %s record", marker)
        return
    # One write() per record keeps lines whole next to ARIA's iteration lines.
    sys.stdout.flush()
    os.write(sys.stdout.fileno(), record)
