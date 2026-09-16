# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm.experimental.ttft_schedulers import (
    NaivePrefillReserveScheduler,
    TokenTimeAwareScheduler,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.skip_global_cleanup


def _add_running_request(scheduler, request_id: str = "running") -> None:
    request = create_requests(1, num_tokens=1, req_ids=[request_id])[0]
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.append_output_token_ids(1)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)


def _add_old_waiting_request(scheduler, request_id: str = "waiting") -> None:
    request = create_requests(1, num_tokens=8, req_ids=[request_id])[0]
    request.arrival_time = time.time() - 0.75
    scheduler.add_request(request)


def _add_running_prefill(scheduler, request_id: str = "running-prefill") -> None:
    request = create_requests(1, num_tokens=8, req_ids=[request_id])[0]
    request.status = RequestStatus.RUNNING
    request.is_prefill_chunk = True
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)


def test_default_scheduler_keeps_decode_first_behavior():
    scheduler = create_scheduler(max_num_batched_tokens=8, max_num_seqs=8)
    for index in range(8):
        _add_running_request(scheduler, f"running-{index}")
    _add_old_waiting_request(scheduler)

    output = scheduler.schedule()

    assert set(output.num_scheduled_tokens) == {
        f"running-{index}" for index in range(8)
    }


def test_custom_schedulers_inherit_async_scheduler():
    assert issubclass(NaivePrefillReserveScheduler, AsyncScheduler)
    assert issubclass(TokenTimeAwareScheduler, AsyncScheduler)


def test_naive_prefill_reserve_leaves_tokens_for_new_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_PREFILL_RESERVE_TOKENS", "4")
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=5,
        scheduler_cls=NaivePrefillReserveScheduler,
    )
    for index in range(4):
        _add_running_request(scheduler, f"running-{index}")
    waiting = create_requests(1, num_tokens=8, req_ids=["waiting"])[0]
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    assert sum(
        tokens
        for request_id, tokens in output.num_scheduled_tokens.items()
        if request_id.startswith("running-")
    ) == 4
    assert output.num_scheduled_tokens["waiting"] == 4


def test_time_aware_scheduler_prefills_before_decodes(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.75")
    monkeypatch.setenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", "0")
    monkeypatch.setenv("VLLM_TTFT_RESCUE_MAX_TOKENS", "8")
    monkeypatch.setenv("VLLM_TTFT_DEBUG", "1")
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    _add_running_request(scheduler)
    _add_old_waiting_request(scheduler)

    output = scheduler.schedule()

    assert "running" not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens["waiting"] == 8
    assert scheduler.ttft_debug_stats.guard_triggers == 1
    assert scheduler.ttft_debug_stats.new_prefills_admitted == 1
    assert scheduler.ttft_debug_stats.prefill_tokens_scheduled == 8


def test_time_aware_guard_does_not_throttle_at_sequence_cap(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.75")
    monkeypatch.setenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", "0")
    monkeypatch.setenv("VLLM_TTFT_RESCUE_MAX_TOKENS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    _add_running_request(scheduler)
    _add_old_waiting_request(scheduler)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["running"] == 1
    assert "waiting" not in output.num_scheduled_tokens
    assert scheduler.ttft_debug_stats.guard_blocked_by_seq_cap == 1
    assert scheduler.ttft_debug_stats.last_running_token_budget_after_policy == 8


def test_partial_prefill_remains_ttft_sensitive(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.75")
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    request = create_requests(1, num_tokens=8, req_ids=["partial"])[0]
    request.arrival_time = time.time() - 0.75
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 4
    request.is_prefill_chunk = True
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["partial"] == 4
    assert scheduler.ttft_debug_stats.last_partial_prefills == 1
    assert scheduler.ttft_debug_stats.guard_triggers == 1
    assert scheduler.ttft_debug_stats.prefill_tokens_scheduled == 4


def test_debug_counters_classify_token_budget_blocking(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.75")
    monkeypatch.setenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", "8")
    monkeypatch.setenv("VLLM_TTFT_RESCUE_MAX_TOKENS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=8,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    _add_running_prefill(scheduler)
    _add_old_waiting_request(scheduler)

    scheduler.schedule()

    assert scheduler.ttft_debug_stats.waiting_admission_blocked_by_token_budget == 1
    assert scheduler.ttft_debug_stats.guard_blocked_by_seq_cap == 0


def test_environment_defaults_are_deterministic(monkeypatch):
    monkeypatch.delenv("VLLM_TTFT_WAIT_THRESHOLD_S", raising=False)
    monkeypatch.delenv("VLLM_TTFT_RESCUE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", raising=False)
    monkeypatch.delenv("VLLM_TTFT_DEBUG", raising=False)
    scheduler = create_scheduler(scheduler_cls=TokenTimeAwareScheduler)

    assert scheduler.ttft_wait_threshold_s == 0.75
    assert scheduler.ttft_rescue_max_tokens == 512
    assert scheduler.ttft_decode_floor_tokens == 512
    assert not scheduler.ttft_debug_enabled
