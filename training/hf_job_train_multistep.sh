#!/usr/bin/env bash
#
# Launch multi-step GRPO training on HuggingFace Jobs. Hand-rolled
# trajectory-level GRPO loop (custom rollout + REINFORCE + KL); used
# when turn_limit > 1 and TRL's single-step GRPOTrainer cannot do
# the job.
#
# Mirrors hf_job_train.sh's install pattern verbatim — same OpenEnv-
# official torch/transformers/trl pin so the env loads identically.

set -euo pipefail

# -------- Configuration --------
REPO_URL="${REPO_URL:-https://huggingface.co/spaces/rishabh16196/prompt_golf_env}"
REPO_REF="${REPO_REF:-main}"
PUSH_TO_HUB="${PUSH_TO_HUB:-rishabh16196/prompt-golf-grpo-multistep}"

AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-1.7B}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
SFT_ADAPTER="${SFT_ADAPTER:-}"   # optional warmstart from a single-step adapter

# Multi-step GRPO knobs (smaller defaults than train.sh because
# trajectories cost ~turn_limit× more per step).
MAX_STEPS="${MAX_STEPS:-200}"
NUM_GENS="${NUM_GENS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-3e-6}"
BETA="${BETA:-0.04}"
TURN_LIMIT="${TURN_LIMIT:-3}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-5h}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"

echo "[hf-jobs] repo=$REPO_URL@$REPO_REF"
echo "[hf-jobs] agent=$AGENT_MODEL target=$TARGET_MODEL judge=$JUDGE_MODEL"
echo "[hf-jobs] sft_adapter=${SFT_ADAPTER:-(none)}"
echo "[hf-jobs] turn_limit=$TURN_LIMIT enable_thinking=$ENABLE_THINKING"
echo "[hf-jobs] steps=$MAX_STEPS gens=$NUM_GENS B=$BATCH_SIZE lr=$LR beta=$BETA"
echo "[hf-jobs] flavor=$FLAVOR timeout=$TIMEOUT push_to_hub=$PUSH_TO_HUB"

# Build CLI tail conditionally (--no-enable-thinking when ENABLE_THINKING=false,
# --sft-adapter only when set).
THINKING_FLAG="--enable-thinking"
if [[ "${ENABLE_THINKING}" == "false" || "${ENABLE_THINKING}" == "False" ]]; then
  THINKING_FLAG="--no-enable-thinking"
fi
SFT_FLAG=""
if [[ -n "${SFT_ADAPTER}" ]]; then
  SFT_FLAG="--sft-adapter ${SFT_ADAPTER}"
fi

read -r -d '' JOB_CMD <<EOF || true
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

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

python -u training/train_grpo_multistep.py \\
  --agent-model ${AGENT_MODEL} \\
  --target-model ${TARGET_MODEL} \\
  --judge-model ${JUDGE_MODEL} \\
  --turn-limit ${TURN_LIMIT} \\
  ${THINKING_FLAG} \\
  --max-steps ${MAX_STEPS} \\
  --num-gens ${NUM_GENS} \\
  --batch-size ${BATCH_SIZE} \\
  --lr ${LR} \\
  --beta ${BETA} \\
  --output-dir /app/outputs/grpo_multistep \\
  ${SFT_FLAG} \\
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
