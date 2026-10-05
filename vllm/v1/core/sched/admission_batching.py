# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouping of WAITING admissions under the active-request cap.

With a saturated ``max_num_active_seqs`` cap, each completion frees one slot
and the next step admits one waiting request. That request's prefill turns a
decode step into a mixed step that stalls every running decode. Admitting
waiting requests in groups lets one mixed step carry several prefills.

On by default with the defaults below when ``max_num_active_seqs`` is set and
at least ``DEFAULT_MIN_ACTIVE_CAP`` (4); otherwise off, since pairing would
double a cap of 1. Configure with
``--additional-config '{"admission_batching": {...}}'``: ``false`` disables,
``true`` selects the defaults under any cap, and a dict overrides any of:

- ``min_batch`` (default 2): admissions to group into one step.
- ``max_wait_steps`` (default 32): longest time, in scheduler steps, a lone
  free slot is held empty waiting for ``min_batch`` slots to free.
- ``overshoot`` (default true): when the hold expires, admit ``min_batch``
  requests anyway, exceeding the cap by up to ``min_batch - 1`` until enough
  requests finish. Otherwise admit what fits under the cap.

Grouping engages only when 0 < free slots < ``min_batch``, at least
``min_batch`` fresh requests are waiting, and the step would otherwise be a
pure decode step. In every other state admission is unchanged, and
``max_wait_steps=0`` without ``overshoot`` is stock admission.
"""

import time
from collections import Counter

from vllm.logger import init_logger

logger = init_logger(__name__)

_OPTIONS = frozenset({"min_batch", "max_wait_steps", "overshoot"})
_LOG_INTERVAL_S = 60.0
# Smallest max_num_active_seqs for which batching is on without configuration.
DEFAULT_MIN_ACTIVE_CAP = 4


class AdmissionBatcher:
    def __init__(
        self, min_batch: int = 2, max_wait_steps: int = 32, overshoot: bool = True
    ) -> None:
        if not isinstance(min_batch, int) or min_batch < 2:
            raise ValueError(f"admission_batching.min_batch must be >= 2: {min_batch}")
        if not isinstance(max_wait_steps, int) or max_wait_steps < 0:
            raise ValueError(
                f"admission_batching.max_wait_steps must be >= 0: {max_wait_steps}"
            )
        if not isinstance(overshoot, bool):
            raise ValueError(
                f"admission_batching.overshoot must be a bool: {overshoot}"
            )
        self.min_batch = min_batch
        self.max_wait_steps = max_wait_steps
        self.overshoot = overshoot
        # Scheduler step at which the pending hold began; None when none pends.
        self._hold_start_step: int | None = None
        self._hold_expired = False

        # Counters for the periodic activation log line.
        self.admissions_per_step: Counter[int] = Counter()
        self.num_hold_episodes = 0
        self.num_held_steps = 0
        self.num_expired_holds = 0
        self.num_overshoot_steps = 0
        self._last_log_time = time.monotonic()

    @classmethod
    def from_additional_config(
        cls, additional_config: object, max_num_active_seqs: int | None = None
    ) -> "AdmissionBatcher | None":
        options = None
        if isinstance(additional_config, dict):
            options = additional_config.get("admission_batching")
        if options is None:
            # Unconfigured: on with the defaults only under a cap of >= 4.
            if (
                max_num_active_seqs is None
                or max_num_active_seqs < DEFAULT_MIN_ACTIVE_CAP
            ):
                return None
            options = {}
        if options is False:
            return None
        if options is True:
            options = {}
        if not isinstance(options, dict):
            raise ValueError(f"admission_batching must be a dict: {options!r}")
        unknown = set(options) - _OPTIONS
        if unknown:
            raise ValueError(f"Unknown admission_batching options: {sorted(unknown)}")
        return cls(**options)

    def __repr__(self) -> str:
        return (
            f"AdmissionBatcher(min_batch={self.min_batch}, "
            f"max_wait_steps={self.max_wait_steps}, overshoot={self.overshoot})"
        )

    def admission_limit(
        self,
        *,
        step: int,
        num_active: int,
        max_active: int,
        max_running: int,
        num_waiting: int,
        can_group: bool,
        stalls_decodes: bool,
    ) -> int:
        """Active-request count up to which this step may admit waiting requests.

        `stalls_decodes` is whether this step schedules decodes and no prefill
        chunk, i.e. whether an admission would turn it into a mixed step.
        """
        num_free = max_active - num_active
        if (
            not can_group
            or num_free <= 0
            or num_free >= self.min_batch
            or num_waiting < self.min_batch
        ):
            self._hold_start_step = None
            return max_active
        if not stalls_decodes:
            # Nothing to stall, or a prefill already rides this step.
            return max_active
        if self._hold_start_step is None:
            self._hold_start_step = step
            self._hold_expired = False
            self.num_hold_episodes += 1
        if step - self._hold_start_step < self.max_wait_steps:
            self.num_held_steps += 1
            return num_active
        # Expired: stays released until the slot or queue state changes.
        if not self._hold_expired:
            self._hold_expired = True
            self.num_expired_holds += 1
        if self.overshoot:
            self.num_overshoot_steps += 1
            return min(num_active + self.min_batch, max_running)
        return max_active

    def record_admissions(self, num_admitted: int) -> None:
        if num_admitted:
            self.admissions_per_step[num_admitted] += 1
        now = time.monotonic()
        if now - self._last_log_time < _LOG_INTERVAL_S:
            return
        self._last_log_time = now
        logger.info(
            "Admission batching: admissions/step=%s hold_episodes=%d "
            "held_steps=%d expired_holds=%d overshoot_steps=%d",
            dict(sorted(self.admissions_per_step.items())),
            self.num_hold_episodes,
            self.num_held_steps,
            self.num_expired_holds,
            self.num_overshoot_steps,
        )
