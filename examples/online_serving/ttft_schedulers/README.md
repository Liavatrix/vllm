# TTFT-aware scheduler experiments

This directory contains two V1 scheduler experiments for decode-heavy workloads
where a new request's prefill is starved by running decodes. They are selected
with `--scheduler-cls` and preserve async scheduling by subclassing
`AsyncScheduler`.

| Policy | Scheduler class | Behavior |
| --- | --- | --- |
| Naive reserve | `NaivePrefillReserveScheduler` | Caps RUNNING work so a fixed share of the iteration is left for new prefills. |
| Token-time aware | `TokenTimeAwareScheduler` | When a new request has waited at least the threshold, schedules no RUNNING work that iteration, allowing WAITING prefills to use the budget. |

The policies intentionally affect only requests in `WAITING` with zero computed
tokens. A preempted request is not considered a TTFT candidate because it has
already received its first token.

## RunPod pod workflow

Start a CUDA-compatible GPU pod with HTTP port `8000` exposed. On the pod,
install the checkout in editable mode once. Keeping the checkout installed is
important: the stock vLLM image does not include these experimental classes.

Use a checkout that matches the vLLM version in the pod image, then apply this
change (for example, by checking out the branch containing it). Check the image
version first with `vllm --version`. This keeps the source scheduler compatible
with the image's compiled CUDA extensions; the model itself stays in the image's
existing Hugging Face cache and does not need to be downloaded again.

```bash
git clone <your-fork-or-repository-url> vllm
cd vllm
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Use the same model, quantization, GPU allocation, and
`--max-num-batched-tokens` as the baseline so the comparison isolates the
scheduler. The helper binds the server to `0.0.0.0:8000`, which RunPod's proxy
can reach.

```bash
# Default is a 25% reserve: 512 tokens when the batch budget is 2048.
bash examples/online_serving/ttft_schedulers/serve.sh naive <QWEN_MODEL> \
  --max-num-batched-tokens 2048 <your-existing-vllm-options>

# Fixed reserve overrides the ratio.
VLLM_PREFILL_RESERVE_TOKENS=512 \
  bash examples/online_serving/ttft_schedulers/serve.sh naive <QWEN_MODEL> \
  --max-num-batched-tokens 2048 <your-existing-vllm-options>

# At 0.75 seconds, an overdue new prefill gets the next iteration's token budget.
VLLM_TTFT_WAIT_THRESHOLD_S=0.75 \
  bash examples/online_serving/ttft_schedulers/serve.sh time-aware <QWEN_MODEL> \
  --max-num-batched-tokens 2048 <your-existing-vllm-options>
```

`vllm bench serve` is the benchmark client, not the inference server. Leave its
workload options unchanged and run it after the selected server is healthy:

```bash
vllm bench serve --base-url http://127.0.0.1:8000 \
  <your-existing-benchmark-options>
```

For the time-aware policy, every iteration containing an overdue new prefill is
prefill-first; it is deliberately a hard latency guard rather than a soft
priority hint. Under sustained overload this can increase decode latency, so
evaluate p99 TTFT and p99 TPOT together against the same C128 workload.

Verify the pod after model loading:

```bash
curl "https://<pod-id>-8000.proxy.runpod.net/health"
```

To return to the baseline, stop the process and launch the normal `vllm serve`
command without `--scheduler-cls`; no code rollback or rebuild is required.
