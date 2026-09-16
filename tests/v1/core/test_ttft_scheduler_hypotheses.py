# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused C128 mechanics tests for the TTFT scheduler experiments."""

import time

import pytest

from vllm.experimental.ttft_schedulers import (
    NaivePrefillReserveScheduler,
    TokenTimeAwareScheduler,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.skip_global_cleanup


def _add_decode(scheduler, request_id: str) -> None:
    request = create_requests(1, num_tokens=1, req_ids=[request_id])[0]
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.append_output_token_ids(1)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)


def _add_c128_decodes(scheduler) -> None:
    for index in range(128):
        _add_decode(scheduler, f"decode-{index}")


def _add_old_prefill(scheduler, request_id: str = "prefill"):
    request = create_requests(1, num_tokens=256, req_ids=[request_id])[0]
    request.arrival_time = time.time() - 1.0
    scheduler.add_request(request)
    return request


def _scheduled_counts(output) -> tuple[int, int, int]:
    decode_tokens = sum(
        tokens
        for request_id, tokens in output.num_scheduled_tokens.items()
        if request_id.startswith("decode-")
    )
    prefill_tokens = output.num_scheduled_tokens.get("prefill", 0)
    return decode_tokens, prefill_tokens, len(output.scheduled_new_reqs)


def test_candidate_1_token_budget_has_capacity_when_a_sequence_slot_exists():
    saturated = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        async_scheduling=True,
    )
    _add_c128_decodes(saturated)
    _add_old_prefill(saturated)

    saturated_output = saturated.schedule()

    assert _scheduled_counts(saturated_output) == (128, 0, 0)
    assert 2048 - sum(saturated_output.num_scheduled_tokens.values()) == 1920
    assert saturated.get_request_counts() == (128, 1)

    one_slot_open = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=129,
        async_scheduling=True,
    )
    _add_c128_decodes(one_slot_open)
    _add_old_prefill(one_slot_open)

    open_output = one_slot_open.schedule()

    assert _scheduled_counts(open_output) == (128, 256, 1)
    assert 2048 - sum(open_output.num_scheduled_tokens.values()) == 1664


def test_candidate_2_reserves_do_not_add_prefill_progress_when_capacity_exists(
    monkeypatch,
):
    def run(policy, **environment):
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        scheduler = create_scheduler(
            max_num_batched_tokens=2048,
            max_num_seqs=129,
            scheduler_cls=policy,
        )
        _add_c128_decodes(scheduler)
        _add_old_prefill(scheduler)
        return _scheduled_counts(scheduler.schedule())

    default = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=129,
        async_scheduling=True,
    )
    _add_c128_decodes(default)
    _add_old_prefill(default)

    default_counts = _scheduled_counts(default.schedule())
    reserve_512 = run(
        NaivePrefillReserveScheduler,
        VLLM_PREFILL_RESERVE_TOKENS="512",
    )
    reserve_2048 = run(
        NaivePrefillReserveScheduler,
        VLLM_PREFILL_RESERVE_TOKENS="2048",
    )
    time_aware = run(
        TokenTimeAwareScheduler,
        VLLM_TTFT_WAIT_THRESHOLD_S="0.75",
        VLLM_TTFT_RESCUE_MAX_TOKENS="512",
        VLLM_TTFT_DECODE_FLOOR_TOKENS="512",
    )

    assert default_counts == (128, 256, 1)
    assert reserve_512 == (128, 256, 1)
    assert reserve_2048 == (0, 256, 1)
    assert time_aware == (128, 256, 1)


def test_candidate_3_256_token_prompt_completes_prefill_in_one_step():
    scheduler = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=129,
        async_scheduling=True,
    )
    _add_c128_decodes(scheduler)
    prefill = _add_old_prefill(scheduler)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[prefill.request_id] == 256
    assert prefill.num_computed_tokens == 256
    assert not prefill.is_prefill_chunk
    assert prefill.num_output_tokens == 0


def test_candidate_4_async_state_is_pre_first_token_after_prompt_scheduling(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.001")
    monkeypatch.setenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", "0")
    scheduler = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    request = _add_old_prefill(scheduler, request_id="request")

    first_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 256
    assert request.num_in_flight_tokens == 256
    assert request.num_output_tokens == 0
    assert scheduler._is_pre_first_token(request)

    second_output = scheduler.schedule()

    assert second_output.num_scheduled_tokens[request.request_id] == 1
    assert request.num_computed_tokens == 257
    assert request.num_in_flight_tokens == 257
    assert request.num_output_tokens == 0
    assert scheduler._is_pre_first_token(request)
    assert scheduler.ttft_debug_stats.guard_triggers == 2
    assert scheduler.ttft_debug_stats.last_running_token_budget_after_policy == 2048

    scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1]],
        ),
    )

    assert request.num_in_flight_tokens == 1
    assert request.num_output_tokens == 1
    assert not scheduler._is_pre_first_token(request)


def test_candidate_5_kv_failure_blocks_admission_despite_token_capacity(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_WAIT_THRESHOLD_S", "0.75")
    monkeypatch.setenv("VLLM_TTFT_DECODE_FLOOR_TOKENS", "512")
    scheduler = create_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=129,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    _add_c128_decodes(scheduler)
    prefill = _add_old_prefill(scheduler)
    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def allocate_slots(request, *args, **kwargs):
        if request.request_id == prefill.request_id:
            return None
        return original_allocate_slots(request, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate_slots)

    output = scheduler.schedule()

    assert _scheduled_counts(output) == (128, 0, 0)
    assert 2048 - sum(output.num_scheduled_tokens.values()) == 1920
    assert scheduler.ttft_debug_stats.waiting_admission_blocked_by_kv_cache == 1
