# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedulers for evaluating TTFT-focused token admission policies.

Select one with ``--scheduler-cls``. Both schedulers retain vLLM's async
scheduler behavior and only change how much of an iteration RUNNING requests
may consume before WAITING requests are considered.
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def _read_positive_float(name: str, default: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be in (0, {maximum}], got {value}")
    return value


def _read_positive_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _read_nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _read_debug_enabled() -> bool:
    value = os.getenv("VLLM_TTFT_DEBUG", "0").lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise ValueError("VLLM_TTFT_DEBUG must be a boolean value")


@dataclass
class TTFTDebugStats:
    scheduler_iterations: int = 0
    guard_triggers: int = 0
    guard_blocked_by_seq_cap: int = 0
    waiting_admission_blocked_by_token_budget: int = 0
    waiting_admission_blocked_by_kv_cache: int = 0
    new_prefills_admitted: int = 0
    prefill_tokens_scheduled: int = 0
    prefill_tokens_executed: int = 0
    preemptions: int = 0
    first_token_completions: int = 0
    last_oldest_pre_first_token_age_s: float = 0.0
    max_observed_pre_first_token_age_s: float = 0.0
    last_first_token_age_s: float = 0.0
    max_first_token_age_s: float = 0.0
    last_waiting_requests: int = 0
    last_untouched_waiting_prefills: int = 0
    last_partial_prefills: int = 0
    last_running_requests: int = 0
    last_running_scheduled_requests: int = 0
    last_running_scheduled_tokens: int = 0
    last_running_token_budget_before_policy: int = 0
    last_running_token_budget_after_policy: int = 0
    last_token_budget_before_waiting: int = 0


class _TTFTScheduler(AsyncScheduler):
    """Base class with TTFT lifecycle accounting for experimental policies."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ttft_debug_enabled = _read_debug_enabled()
        self.ttft_debug_interval = _read_nonnegative_int(
            "VLLM_TTFT_DEBUG_INTERVAL", 1000
        )
        if self.ttft_debug_interval == 0:
            raise ValueError("VLLM_TTFT_DEBUG_INTERVAL must be positive")
        self.ttft_debug_stats = TTFTDebugStats()
        self._pre_first_token_ids: set[str] = set()
        self._pre_first_token_ids_by_output: dict[int, set[str]] = {}
        self._running_ids_at_schedule_start: set[str] = set()
        self._rescue_request_id: str | None = None
        self._guard_active = False

    def _log_config(self, **config: int | float | str | bool | None) -> None:
        logger.info("TTFT scheduler config: %s", json.dumps(config, sort_keys=True))

    @staticmethod
    def _is_pre_first_token(request: Request) -> bool:
        return (
            request.sampling_params is not None
            and request.num_output_tokens == 0
            and not request.is_finished()
        )

    @classmethod
    def _is_untouched_waiting_prefill(cls, request: Request) -> bool:
        return (
            request.status == RequestStatus.WAITING
            and request.num_computed_tokens == 0
            and cls._is_pre_first_token(request)
        )

    @classmethod
    def _is_rescuable_waiting_prefill(cls, request: Request) -> bool:
        return (
            request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            and cls._is_pre_first_token(request)
        )

    def _iter_rescuable_waiting_prefills(self):
        for queue in (self.waiting, self.skipped_waiting):
            yield from (
                request
                for request in queue
                if self._is_rescuable_waiting_prefill(request)
            )

    def _oldest_pre_first_token_request(self) -> Request | None:
        candidates = [
            request
            for request in self.requests.values()
            if self._is_pre_first_token(request)
        ]
        return min(candidates, key=lambda request: request.arrival_time, default=None)

    def _oldest_rescuable_waiting_prefill(self) -> Request | None:
        return min(
            self._iter_rescuable_waiting_prefills(),
            key=lambda request: request.arrival_time,
            default=None,
        )

    def _begin_debug_iteration(self) -> None:
        stats = self.ttft_debug_stats
        stats.scheduler_iterations += 1
        self._pre_first_token_ids = {
            request.request_id
            for request in self.requests.values()
            if self._is_pre_first_token(request)
        }
        self._running_ids_at_schedule_start = {
            request.request_id for request in self.running
        }
        stats.last_waiting_requests = len(self.waiting) + len(self.skipped_waiting)
        stats.last_untouched_waiting_prefills = sum(
            self._is_untouched_waiting_prefill(request)
            for queue in (self.waiting, self.skipped_waiting)
            for request in queue
        )
        stats.last_partial_prefills = sum(
            self._is_pre_first_token(request)
            and request.num_computed_tokens > 0
            and request.num_computed_tokens < request.num_tokens
            for request in self.running
        )
        stats.last_running_requests = len(self.running)
        oldest = self._oldest_pre_first_token_request()
        if oldest is not None:
            age = max(0.0, time.time() - oldest.arrival_time)
            stats.last_oldest_pre_first_token_age_s = age
            stats.max_observed_pre_first_token_age_s = max(
                stats.max_observed_pre_first_token_age_s, age
            )
        else:
            stats.last_oldest_pre_first_token_age_s = 0.0
        self._rescue_request_id = None
        self._guard_active = False

    def _finish_debug_iteration(self, output: SchedulerOutput) -> None:
        stats = self.ttft_debug_stats
        scheduled_tokens = output.num_scheduled_tokens
        running_ids = self._running_ids_at_schedule_start
        stats.last_running_scheduled_requests = sum(
            request_id in running_ids for request_id in scheduled_tokens
        )
        stats.last_running_scheduled_tokens = sum(
            tokens
            for request_id, tokens in scheduled_tokens.items()
            if request_id in running_ids
        )
        stats.prefill_tokens_scheduled += sum(
            tokens
            for request_id, tokens in scheduled_tokens.items()
            if request_id in self._pre_first_token_ids
        )
        stats.new_prefills_admitted += sum(
            request.req_id in self._pre_first_token_ids
            for request in output.scheduled_new_reqs
        )
        stats.preemptions += len(output.preempted_req_ids)
        if (
            self.ttft_debug_enabled
            and stats.scheduler_iterations % self.ttft_debug_interval == 0
        ):
            logger.info("TTFT scheduler stats: %s", json.dumps(asdict(stats)))

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._begin_debug_iteration()
        output = super().schedule(throttle_prefills)
        self._finish_debug_iteration(output)
        self._pre_first_token_ids_by_output[id(output)] = self._pre_first_token_ids
        return output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        pre_first_token_ids = self._pre_first_token_ids_by_output.pop(
            id(scheduler_output), set()
        )
        self.ttft_debug_stats.prefill_tokens_executed += sum(
            tokens
            for request_id, tokens in scheduler_output.num_scheduled_tokens.items()
            if request_id in pre_first_token_ids
        )
        return super().update_from_output(scheduler_output, model_runner_output)

    def _on_waiting_schedule_start(self, token_budget: int) -> None:
        self.ttft_debug_stats.last_token_budget_before_waiting = token_budget
        if self._rescue_request_id is not None and token_budget == 0:
            self.ttft_debug_stats.waiting_admission_blocked_by_token_budget += 1

    def _on_waiting_admission_blocked_by_seq_cap(self, token_budget: int) -> None:
        if self._guard_active and self._rescue_request_id is not None:
            self.ttft_debug_stats.guard_blocked_by_seq_cap += 1

    def _on_waiting_admission_blocked_by_kv_cache(self, request: Request) -> None:
        if request.request_id == self._rescue_request_id:
            self.ttft_debug_stats.waiting_admission_blocked_by_kv_cache += 1

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        was_pre_first_token = self._is_pre_first_token(request)
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids, is_stale
        )
        if was_pre_first_token and request.num_output_tokens > 0:
            age = max(0.0, time.time() - request.arrival_time)
            stats = self.ttft_debug_stats
            stats.first_token_completions += 1
            stats.last_first_token_age_s = age
            stats.max_first_token_age_s = max(stats.max_first_token_age_s, age)
        return new_token_ids, stopped


class NaivePrefillReserveScheduler(_TTFTScheduler):
    """Reserve a fixed share of each mixed batch for new-request prefills.

    ``VLLM_PREFILL_RESERVE_TOKENS`` takes precedence when set. Otherwise the
    reserve is ``ceil(VLLM_PREFILL_RESERVE_RATIO * token_budget)``; the default
    ratio is 0.25, which reserves 512 tokens with a 2048-token batch.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prefill_reserve_tokens = _read_positive_int(
            "VLLM_PREFILL_RESERVE_TOKENS"
        )
        self.prefill_reserve_ratio = _read_positive_float(
            "VLLM_PREFILL_RESERVE_RATIO", 0.25, 1.0
        )
        self._log_config(
            scheduler="naive_prefill_reserve",
            prefill_reserve_tokens=self.prefill_reserve_tokens,
            prefill_reserve_ratio=self.prefill_reserve_ratio,
            ttft_debug=self.ttft_debug_enabled,
            ttft_debug_interval=self.ttft_debug_interval,
        )

    def _get_running_token_budget(self, token_budget: int) -> int:
        self.ttft_debug_stats.last_running_token_budget_before_policy = token_budget
        if not any(self._iter_rescuable_waiting_prefills()):
            self.ttft_debug_stats.last_running_token_budget_after_policy = token_budget
            return token_budget
        reserve = self.prefill_reserve_tokens
        if reserve is None:
            reserve = math.ceil(token_budget * self.prefill_reserve_ratio)
        running_token_budget = max(0, token_budget - min(token_budget, reserve))
        self.ttft_debug_stats.last_running_token_budget_after_policy = (
            running_token_budget
        )
        return running_token_budget


class TokenTimeAwareScheduler(_TTFTScheduler):
    """Reserve targeted prefill capacity after a TTFT wait threshold.

    A request is eligible once ``time.time() - arrival_time`` reaches
    ``VLLM_TTFT_WAIT_THRESHOLD_S`` (0.75 seconds by default). The policy keeps
    a decode floor and reserves only enough capacity for the oldest queued
    request that has not yet produced an output token.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ttft_wait_threshold_s = _read_positive_float(
            "VLLM_TTFT_WAIT_THRESHOLD_S", 0.75, float("inf")
        )
        self.ttft_rescue_max_tokens = _read_nonnegative_int(
            "VLLM_TTFT_RESCUE_MAX_TOKENS", 512
        )
        self.ttft_decode_floor_tokens = _read_nonnegative_int(
            "VLLM_TTFT_DECODE_FLOOR_TOKENS", 512
        )
        self._log_config(
            scheduler="token_time_aware",
            ttft_wait_threshold_s=self.ttft_wait_threshold_s,
            ttft_rescue_max_tokens=self.ttft_rescue_max_tokens,
            ttft_decode_floor_tokens=self.ttft_decode_floor_tokens,
            ttft_debug=self.ttft_debug_enabled,
            ttft_debug_interval=self.ttft_debug_interval,
        )

    def _get_running_token_budget(self, token_budget: int) -> int:
        stats = self.ttft_debug_stats
        stats.last_running_token_budget_before_policy = token_budget
        oldest = self._oldest_pre_first_token_request()
        if oldest is None:
            stats.last_running_token_budget_after_policy = token_budget
            return token_budget

        age = max(0.0, time.time() - oldest.arrival_time)
        if age < self.ttft_wait_threshold_s:
            stats.last_running_token_budget_after_policy = token_budget
            return token_budget

        self._guard_active = True
        stats.guard_triggers += 1
        rescue_request = self._oldest_rescuable_waiting_prefill()
        if rescue_request is None:
            stats.last_running_token_budget_after_policy = token_budget
            return token_budget

        num_running = len(self.running) + self.num_waiting_for_streaming_input
        if num_running >= self.max_num_running_reqs:
            stats.guard_blocked_by_seq_cap += 1
            stats.last_running_token_budget_after_policy = token_budget
            return token_budget

        self._rescue_request_id = rescue_request.request_id
        remaining_prefill_tokens = max(
            1, rescue_request.num_tokens - rescue_request.num_computed_tokens
        )
        rescue_tokens = min(remaining_prefill_tokens, self.ttft_rescue_max_tokens)
        reserve = min(
            rescue_tokens,
            max(0, token_budget - self.ttft_decode_floor_tokens),
        )
        if reserve == 0:
            stats.last_running_token_budget_after_policy = token_budget
            return token_budget

        running_token_budget = token_budget - reserve
        stats.last_running_token_budget_after_policy = running_token_budget
        return running_token_budget
