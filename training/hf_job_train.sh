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

# Qwen3 family (supported by transformers==4.56.2 from OpenEnv recipe).
# Qwen3.5 / Qwen3.5-9B need transformers>=4.60 which pulls vllm as a
# hard dep via TRL's newer import path; installing vllm on top of the
# current image is flaky. Revisit for v3.
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-1.7B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-1.7B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"

MAX_STEPS="${MAX_STEPS:-500}"
NUM_GENS="${NUM_GENS:-8}"   # must divide BATCH_SIZE*GRAD_ACCUM (2*4=8)
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.04}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-4}"

FLAVOR="${FLAVOR:-l40sx1}"            # t4-medium | l4x1 | a10g-large | l40sx1 | a100-large
TIMEOUT="${TIMEOUT:-4h}"              # bumped 3h -> 4h to cover judge-inference overhead
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"

echo "[hf-jobs] repo=$REPO_URL@$REPO_REF"
echo "[hf-jobs] agent=$AGENT_MODEL target=$TARGET_MODEL"
echo "[hf-jobs] steps=$MAX_STEPS  gens=$NUM_GENS  flavor=$FLAVOR"
echo "[hf-jobs] push_to_hub=$PUSH_TO_HUB"

# Single-line bash command the job will execute.
read -r -d '' JOB_CMD <<EOF || true
set -euo pipefail

# --- OpenEnv-official install pattern (verbatim from unsloth_2048.ipynb) ---
# Triton JIT needs gcc
apt-get update -qq
apt-get install -y -qq git curl build-essential

# uv for deterministic resolution
pip install --upgrade -q uv

# Upgrade torch 2.4 -> 2.8+, install Unsloth from git with [base] extras.
uv pip install --system -q \\
    "torch>=2.8.0" "torchvision>=0.25.0" "triton>=3.4.0" bitsandbytes \\
    "transformers==4.56.2" \\
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"

# --no-deps second pass to pin specific versions (including trl==0.22.2)
uv pip install --system --upgrade --no-deps -q \\
    "transformers==4.56.2" tokenizers "trl==0.22.2" unsloth unsloth_zoo

# Clone our env and install (no-deps; we've pinned the heavy stuff above)
git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app
cd /app
pip install -q --no-deps -e .

# Remaining light deps (peft/datasets/accelerate/etc are not in the
# OpenEnv-official list but are needed by train_grpo.py).
# openenv-core is our env's runtime — pin >=0.2.2 to match pyproject.
pip install -q 'openenv-core[core]>=0.2.2' \\
               'peft>=0.13.0' 'datasets>=3.0.0' 'accelerate>=0.34.0' \\
               'huggingface_hub>=0.26.0' 'safetensors>=0.4.0' matplotlib

# Verify (prints to logs for debugging)
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import trl; print('trl:', trl.__version__)"
# -------------------------------------------------------------------------------

python -u training/train_grpo.py \\
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
