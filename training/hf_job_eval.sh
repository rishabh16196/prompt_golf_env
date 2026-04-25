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

AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-1.7B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-1.7B}"
# Eval is deterministic at temperature=0; seeds>1 produces bit-identical
# duplicate rows. Override only when running with temperature>0.
SEEDS_PER_TASK="${SEEDS_PER_TASK:-1}"
# Match the chat template the adapter was TRAINED against. The
# in-flight v2 adapter trained with thinking=OFF; v3 cross-family runs
# will train with thinking=ON. Override accordingly.
ENABLE_THINKING="${ENABLE_THINKING:-false}"

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-1h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"

# Build conditional thinking flag
THINKING_FLAG="--no-enable-thinking"
if [[ "${ENABLE_THINKING}" == "true" || "${ENABLE_THINKING}" == "True" ]]; then
  THINKING_FLAG="--enable-thinking"
fi

run_eval() {
  local LABEL=$1
  local EXTRA_FLAGS=$2

  local CMD
  read -r -d '' CMD <<EOF || true
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git curl build-essential
pip install --upgrade -q uv
uv pip install --system -q \\
    "torch>=2.8.0" "torchvision>=0.25.0" "triton>=3.4.0" bitsandbytes \\
    "transformers==4.56.2" \\
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --system --upgrade --no-deps -q \\
    "transformers==4.56.2" tokenizers "trl==0.22.2" unsloth unsloth_zoo
git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app
cd /app
pip install -q --no-deps -e .
pip install -q 'openenv-core[core]>=0.2.2' \\
               'peft>=0.13.0' 'datasets>=3.0.0' 'accelerate>=0.34.0' \\
               'huggingface_hub>=0.26.0' 'safetensors>=0.4.0' matplotlib
python -u training/eval_before_after.py \
  --agent-model ${AGENT_MODEL} \
  --target-model ${TARGET_MODEL} \
  --seeds-per-task ${SEEDS_PER_TASK} \
  ${THINKING_FLAG} \
  --label ${LABEL} \
  --output-json /app/outputs/eval_${LABEL}.jsonl \
  --push-to-hub ${ADAPTER_REPO} \
  ${EXTRA_FLAGS}
EOF

  hf jobs run \
    --flavor "${FLAVOR}" \
    --timeout "${TIMEOUT}" \
    --detach \
    --secrets HF_TOKEN \
    "${IMAGE}" \
    -- bash -c "${CMD}"
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
