#!/usr/bin/env bash
#
# Run before/after eval for Prompt Golf as two HF Jobs.
# Reports one JSONL per label; diff the summaries for the demo.
#
# Usage:
#   bash training/hf_job_eval.sh base           # base model only
#   bash training/hf_job_eval.sh trained        # base + pushed adapter
#   bash training/hf_job_eval.sh both           # submits both jobs

set -euo pipefail

MODE="${1:-both}"

REPO_URL="${REPO_URL:-https://huggingface.co/spaces/rishabh16196/prompt_golf_env}"
REPO_REF="${REPO_REF:-main}"
ADAPTER_REPO="${ADAPTER_REPO:-rishabh16196/prompt-golf-grpo-1.5b}"

AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-3}"

FLAVOR="${FLAVOR:-a10g-large}"
TIMEOUT="${TIMEOUT:-1h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

run_eval() {
  local LABEL=$1
  local EXTRA_FLAGS=$2

  local CMD
  read -r -d '' CMD <<EOF || true
set -euo pipefail
apt-get update -qq && apt-get install -y --no-install-recommends git
pip install -q --upgrade pip
git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app
cd /app
pip install -q -e .
pip install -q -r training/requirements.txt
python -u training/eval_before_after.py \
  --agent-model ${AGENT_MODEL} \
  --target-model ${TARGET_MODEL} \
  --seeds-per-task ${SEEDS_PER_TASK} \
  --label ${LABEL} \
  --output-json /app/outputs/eval_${LABEL}.jsonl \
  ${EXTRA_FLAGS}
EOF

  hf jobs run \
    --flavor "${FLAVOR}" \
    --timeout "${TIMEOUT}" \
    --secret HF_TOKEN \
    "${IMAGE}" \
    bash -lc "${CMD}"
}

case "${MODE}" in
  base)
    run_eval base ""
    ;;
  trained)
    run_eval trained "--adapter ${ADAPTER_REPO}"
    ;;
  both)
    echo "[hf-jobs] submitting BASE eval..."
    run_eval base ""
    echo "[hf-jobs] submitting TRAINED eval..."
    run_eval trained "--adapter ${ADAPTER_REPO}"
    ;;
  *)
    echo "usage: $0 {base|trained|both}" >&2
    exit 1
    ;;
esac
