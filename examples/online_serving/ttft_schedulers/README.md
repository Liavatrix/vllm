# TTFT-aware scheduler experiments

This directory contains two V1 scheduler experiments for decode-heavy workloads
where a new request's prefill is starved by running decodes. They are selected
with `--scheduler-cls` and preserve async scheduling by subclassing
`AsyncScheduler`.

| Policy | Scheduler class | Behavior |
| --- | --- | --- |
| Naive reserve | `NaivePrefillReserveScheduler` | Caps RUNNING work so a fixed share of the iteration is left for new prefills. |
| Token-time aware | `TokenTimeAwareScheduler` | When a pre-first-token request has waited at least the threshold, reserves targeted capacity for the oldest queued prefill while retaining a decode floor. |

Both policies consider queued requests that have no generated output token.
The time-aware policy separately observes partially computed prefills that have
not yet produced an output token; it does not bypass vLLM's sequence limit.

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

# At 0.75 seconds, reserve up to 512 tokens for the oldest overdue queued
# prefill, while preserving a 512-token decode floor. For 256-token prompts,
# the targeted reserve is 256 tokens.
VLLM_TTFT_WAIT_THRESHOLD_S=0.75 \
VLLM_TTFT_RESCUE_MAX_TOKENS=512 \
VLLM_TTFT_DECODE_FLOOR_TOKENS=512 \
  bash examples/online_serving/ttft_schedulers/serve.sh time-aware <QWEN_MODEL> \
  --max-num-batched-tokens 2048 <your-existing-vllm-options>
```

`vllm bench serve` is the benchmark client, not the inference server. Leave its
workload options unchanged and run it after the selected server is healthy:

```bash
vllm bench serve --base-url http://127.0.0.1:8000 \
  <your-existing-benchmark-options>
```

Set `VLLM_TTFT_DEBUG=1` to log the configuration once and a structured counter
snapshot every `VLLM_TTFT_DEBUG_INTERVAL` scheduler iterations (default 1000).
The snapshot separates guard activation, sequence/KV/token admission blocks,
prefill tokens scheduled versus returned from execution, and first-output
completion. Under sustained overload, evaluate p99 TTFT and p99 TPOT together
against the same C128 workload.

Verify the pod after model loading:

```bash
curl "https://<pod-id>-8000.proxy.runpod.net/health"
```

To return to the baseline, stop the process and launch the normal `vllm serve`
command without `--scheduler-cls`; no code rollback or rebuild is required.
