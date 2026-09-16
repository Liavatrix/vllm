# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm.experimental.ttft_schedulers import (
    NaivePrefillReserveScheduler,
    TokenTimeAwareScheduler,
)
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def _add_running_request(scheduler, request_id: str = "running") -> None:
    request = create_requests(1, num_tokens=1, req_ids=[request_id])[0]
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.append_output_token_ids(1)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)


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
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        scheduler_cls=TokenTimeAwareScheduler,
    )
    _add_running_request(scheduler)
    waiting = create_requests(1, num_tokens=8, req_ids=["waiting"])[0]
    waiting.arrival_time = time.time() - 0.75
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    assert "running" not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens["waiting"] == 8
