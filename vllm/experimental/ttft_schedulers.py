# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedulers for evaluating TTFT-focused token admission policies.

Select one with ``--scheduler-cls``. Both schedulers retain vLLM's async
scheduler behavior and only change how much of an iteration RUNNING requests
may consume before WAITING requests are considered.
"""

import math
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus


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


class _TTFTScheduler(AsyncScheduler):
    """Base class with helpers for new requests that have not seen TTFT."""

    @staticmethod
    def _is_new_prefill(request: Request) -> bool:
        return (
            request.status == RequestStatus.WAITING
            and request.num_computed_tokens == 0
        )

    def _iter_new_prefills(self):
        for queue in (self.waiting, self.skipped_waiting):
            yield from (
                request for request in queue if self._is_new_prefill(request)
            )


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

    def _get_running_token_budget(self, token_budget: int) -> int:
        if not any(self._iter_new_prefills()):
            return token_budget
        reserve = self.prefill_reserve_tokens
        if reserve is None:
            reserve = math.ceil(token_budget * self.prefill_reserve_ratio)
        return max(0, token_budget - min(token_budget, reserve))


class TokenTimeAwareScheduler(_TTFTScheduler):
    """Give new prefills an entire iteration after their TTFT wait threshold.

    A request is eligible once ``time.time() - arrival_time`` reaches
    ``VLLM_TTFT_WAIT_THRESHOLD_S`` (0.75 seconds by default). The policy uses
    the original arrival time and excludes preempted requests, whose TTFT was
    already delivered.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ttft_wait_threshold_s = _read_positive_float(
            "VLLM_TTFT_WAIT_THRESHOLD_S", 0.75, float("inf")
        )

    def _get_running_token_budget(self, token_budget: int) -> int:
        now = time.time()
        if any(
            now - request.arrival_time >= self.ttft_wait_threshold_s
            for request in self._iter_new_prefills()
        ):
            return 0
        return token_budget
