#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash serve.sh {naive|time-aware} <model> [vllm serve options...]

Examples:
  VLLM_PREFILL_RESERVE_TOKENS=512 bash serve.sh naive Qwen/Qwen2.5-7B-Instruct \
    --max-num-batched-tokens 2048
  VLLM_TTFT_WAIT_THRESHOLD_S=0.75 bash serve.sh time-aware Qwen/Qwen2.5-7B-Instruct \
    --max-num-batched-tokens 2048
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

policy="$1"
model="$2"
shift 2

case "$policy" in
  naive)
    scheduler_cls="vllm.experimental.ttft_schedulers.NaivePrefillReserveScheduler"
    ;;
  time-aware)
    scheduler_cls="vllm.experimental.ttft_schedulers.TokenTimeAwareScheduler"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

exec vllm serve "$model" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT:-8000}" \
  --scheduler-cls "$scheduler_cls" \
  "$@"
