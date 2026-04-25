#!/usr/bin/env bash
#
# Launch the per-task baseline-capability profile on HuggingFace Jobs.
# Runs the target model on every task using the verbose hand-written
# description as the prompt, and records description_baseline per task.
#
# Use this to decide whether to bump the target from 1.7B to a larger
# model BEFORE spending GPU hours on a long training run.

set -euo pipefail

# -------- Configuration --------
REPO_URL="${REPO_URL:-https://huggingface.co/spaces/rishabh16196/prompt_golf_env}"
REPO_REF="${REPO_REF:-main}"
PUSH_TO_HUB="${PUSH_TO_HUB:-rishabh16196/prompt-golf-grpo-1.5b}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-1.7B}"
TASKS="${TASKS:-all}"

FLAVOR="${FLAVOR:-l4x1}"           # smaller flavor — no agent, no judge, no GRPO
TIMEOUT="${TIMEOUT:-30m}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"

echo "[hf-jobs] repo=$REPO_URL@$REPO_REF"
echo "[hf-jobs] target=$TARGET_MODEL  tasks=$TASKS"
echo "[hf-jobs] flavor=$FLAVOR  push_to_hub=$PUSH_TO_HUB"

read -r -d '' JOB_CMD <<EOF || true
set -euo pipefail

apt-get update -qq
apt-get install -y -qq git curl build-essential

pip install --upgrade -q uv

uv pip install --system -q \\
    "torch>=2.8.0" "torchvision>=0.25.0" "triton>=3.4.0" bitsandbytes \\
    "transformers==4.56.2"

uv pip install --system --upgrade --no-deps -q \\
    "transformers==4.56.2" tokenizers

git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app
cd /app
pip install -q --no-deps -e .

pip install -q 'openenv-core[core]>=0.2.2' \\
               'datasets>=3.0.0' 'accelerate>=0.34.0' \\
               'huggingface_hub>=0.26.0' 'safetensors>=0.4.0'

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

python -u training/profile_baseline.py \\
  --target-model ${TARGET_MODEL} \\
  --tasks ${TASKS} \\
  --output-csv /app/outputs/baseline_profile.csv \\
  ${PUSH_TO_HUB:+--push-to-hub ${PUSH_TO_HUB}}
echo "[hf-jobs] done."
EOF

hf jobs run \
  --flavor "${FLAVOR}" \
  --timeout "${TIMEOUT}" \
  --detach \
  --secrets HF_TOKEN \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --env TRANSFORMERS_VERBOSITY=warning \
  "${IMAGE}" \
  -- bash -c "${JOB_CMD}"
