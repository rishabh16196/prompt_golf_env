#!/usr/bin/env bash
#
# Launch the Prompt Golf GRPO training run on HuggingFace Jobs.
#
# Prereqs:
#   - `hf` CLI installed:  pip install -U huggingface_hub
#   - Logged in with a WRITE token:  hf auth login
#   - The repo containing this code is pushed to either GitHub or HF Hub so
#     the job can `git clone` it. Set REPO_URL below.
#
# Monitor:  hf jobs ls
# Logs:     hf jobs logs <job-id> --follow
# Cancel:   hf jobs cancel <job-id>

set -euo pipefail

# -------- Configuration --------
REPO_URL="${REPO_URL:-https://huggingface.co/spaces/rishabh16196/prompt_golf_env}"
REPO_REF="${REPO_REF:-main}"
PUSH_TO_HUB="${PUSH_TO_HUB:-rishabh16196/prompt-golf-grpo-1.5b}"

AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

MAX_STEPS="${MAX_STEPS:-500}"
NUM_GENS="${NUM_GENS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.04}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-4}"

FLAVOR="${FLAVOR:-l40sx1}"            # t4-medium | l4x1 | a10g-large | l40sx1 | a100-large
TIMEOUT="${TIMEOUT:-3h}"
IMAGE="${IMAGE:-python:3.12-slim}"

echo "[hf-jobs] repo=$REPO_URL@$REPO_REF"
echo "[hf-jobs] agent=$AGENT_MODEL target=$TARGET_MODEL"
echo "[hf-jobs] steps=$MAX_STEPS  gens=$NUM_GENS  flavor=$FLAVOR"
echo "[hf-jobs] push_to_hub=$PUSH_TO_HUB"

# Single-line bash command the job will execute.
read -r -d '' JOB_CMD <<EOF || true
set -euo pipefail
apt-get update -qq && apt-get install -y --no-install-recommends git curl build-essential
pip install -q --upgrade pip
# --- Stack: torch 2.7 (has FSDPModule) + TRL + HF ecosystem from cu124 wheels ---
# Torch 2.7 cu124 bundles CUDA/cuDNN runtime libs, so python:3.12-slim base is fine
# as long as the HF Jobs hardware has an NVIDIA driver (which l40sx1 does).
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124
# ---------------------------------------------------------------------------------
git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app
cd /app
pip install -q --no-deps -e .
pip install -q -r training/requirements.txt
python -u training/train_grpo.py \
  --agent-model ${AGENT_MODEL} \
  --target-model ${TARGET_MODEL} \
  --max-steps ${MAX_STEPS} \
  --num-generations ${NUM_GENS} \
  --per-device-batch-size ${BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --learning-rate ${LR} \
  --beta ${BETA} \
  --seeds-per-task ${SEEDS_PER_TASK} \
  --output-dir /app/outputs/grpo \
  ${PUSH_TO_HUB:+--push-to-hub ${PUSH_TO_HUB}}
echo "[hf-jobs] training complete. Rendering plots..."
python -u training/make_plots.py --metrics /app/outputs/grpo/train_metrics.jsonl --out-dir /app/outputs/grpo/plots
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
