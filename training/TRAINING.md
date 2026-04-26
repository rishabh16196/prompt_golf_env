# Training & evaluation pipeline

End-to-end recipe for reproducing the Prompt Golf adapters and demo CSVs from scratch — single-step hero, multi-step variant, control runs, plots, and the Trackio dashboard. Every step runs on HuggingFace Jobs (single L40S, 48 GB) so you don't need a local GPU.

> **TL;DR:** profile the target → train → eval base + trained → build demo CSV → render plots → replay metrics to Trackio. Each step is a separate script under `training/` with a `hf_job_*.sh` wrapper.

---

## What the `hf_job_*.sh` launchers actually do

Each `.sh` is a thin wrapper around the [`hf jobs`](https://huggingface.co/docs/huggingface_hub/guides/jobs) CLI — HuggingFace's managed-GPU runner. The pattern is identical across all four:

1. **Read config from env vars** (with sensible defaults) — model names, destination repo, hyperparameters, GPU flavor (`l40sx1` / `l4x1` / `t4-medium`), timeout, etc.
2. **Compose a long bash command** to run *inside* the remote container:
   - `apt-get install` system deps (git, curl, build tools).
   - `pip install` the **OpenEnv-official torch/transformers/trl pin** — this is finicky (torch ≥2.8, transformers==4.56.2, trl==0.22.2). That's why the install lives in the `.sh`, not in `requirements.txt`.
   - `git clone` this repo at `${REPO_REF}` and `pip install -e .` it.
   - Run the actual Python entry point (`train_grpo.py` / `eval_before_after.py` / `profile_baseline.py`).
3. **Submit it via `hf jobs run`** with `--flavor`, `--timeout`, `--secrets HF_TOKEN`, `--detach`. Returns a job ID and runs in the background on HF's GPUs.

| Script | Wraps | Time | Purpose |
|---|---|---|---|
| [`hf_job_profile.sh`](./hf_job_profile.sh) | `profile_baseline.py` | ≈30m on L4 | Verbose-prompt accuracy per task on a given target. No agent, no judge — cheap. |
| [`hf_job_train.sh`](./hf_job_train.sh) | `train_grpo.py` | ≈3h on L40S | Hero recipe — TRL GRPO single-step, 500 steps × 8 generations. Pushes adapter + plots + metrics. |
| [`hf_job_train_multistep.sh`](./hf_job_train_multistep.sh) | `train_grpo_multistep.py` | ≈3.5h on L40S | 3-turn variant — hand-rolled trajectory-level GRPO. Reads `SFT_ADAPTER` to warm-start from a hero adapter. |
| [`hf_job_eval.sh`](./hf_job_eval.sh) | `eval_before_after.py` | 2 × ≈15m on L40S | Takes `base \| trained \| both` as `$1`. `both` submits two jobs (with and without `--adapter`). |

**Why this layer exists at all:**
- The "compose the command that runs inside the container" step is a 30-line bash heredoc with very particular pip-install ordering. You don't want to retype that from memory each run.
- Defaults make `bash training/hf_job_train.sh` a one-liner. Customize via env-var overrides (`PUSH_TO_HUB=... TARGET_MODEL=... bash ...`).
- Same `.sh` works locally and on CI — they don't run anything on your machine, they only **dispatch** to HF's cluster.

**What the `.sh` files don't do:**
- Don't wait for the job to finish — `--detach` returns immediately. Monitor with `hf jobs ps -a` and `hf jobs logs <id> --follow`.
- Don't run on your laptop. No local GPU required.
- The CSV-builder (`build_before_after_csv.py`), plot-renderer (`make_plots.py`), and Trackio-replayer (`replay_to_trackio.py`) **don't** have `.sh` wrappers — they're cheap CPU-only scripts you run locally after the GPU jobs finish.

### Quick start — just run the .sh

For most users, two setup commands and you're dispatching real training jobs:

```bash
# 1. HF write token (free account; takes 30 seconds)
hf auth login                                          # paste a write token

# 2. Override the default PUSH_TO_HUB so artifacts go to your namespace
export PUSH_TO_HUB=your-username/your-adapter-repo
```

Then any of:

```bash
bash training/hf_job_profile.sh                        # ≈30m on L4
bash training/hf_job_train.sh                          # ≈3h on L40S
bash training/hf_job_eval.sh both                      # 2 × ≈15m on L40S
SFT_ADAPTER=$PUSH_TO_HUB \
  bash training/hf_job_train_multistep.sh              # ≈3.5h on L40S
```

Each call returns a job ID immediately (`hf jobs run --detach`). Monitor with `hf jobs ps -a` and `hf jobs logs <job-id> --follow`.

**Common gotchas before you click run:**
- **Llama-3.2 is gated.** Accept the license at <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> first; the job dies on model-download otherwise. (Approval is usually instant.)
- **HF Jobs quota.** Free accounts get a small monthly GPU budget; a hero training run burns several hours of it. The script returns a quota error within 30 seconds if you're out — nothing has actually started.
- **`whoami` rate-limit.** Dispatching 4–5 jobs in rapid succession will lock you out for 5–25 min. Pace dispatches; don't poll-loop.
- **Default `REPO_URL`** clones `https://huggingface.co/spaces/rishabh16196/prompt_golf_env`. If you've forked the env, override: `REPO_URL=https://huggingface.co/spaces/your-name/your-fork bash ...`.
- **Default `TARGET_MODEL`** is `meta-llama/Llama-3.2-3B-Instruct`. Override per call: `TARGET_MODEL=microsoft/Phi-3-mini-4k-instruct bash training/hf_job_train.sh`.

That's the whole "user wants to reproduce this" path. The §1–§8 sections below go deeper if you want to understand individual steps, customize hyperparameters, or build the demo CSV / plots / Trackio replay locally after the jobs finish.

---

## 0. Prerequisites

- **HuggingFace account + token** with write access to a destination namespace (yours, not `rishabh16196/...`). Login locally:
  ```bash
  hf auth login
  ```
- **Python 3.10+** with the OpenEnv repo installed (`pip install -e .` from the repo root).
- **Llama-3.2-3B-Instruct access**. The model is gated — request access at <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> and confirm `hf auth whoami` shows your account before you dispatch jobs.
- **HF Jobs quota.** A full hero training run is ≈3h on L40S; multi-step is ≈3.5h. Eval pairs are ≈15 min each.

All `hf_job_*.sh` launchers honor these env vars; export them once at the top of your shell session and you're done:

```bash
export PUSH_TO_HUB=your-username/your-adapter-repo   # destination for adapter + plots + metrics + evals
export TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct # frozen target the agent learns to whisper to
export AGENT_MODEL=Qwen/Qwen3-1.7B                    # trainable agent base
```

---

## 1. (Optional but recommended) Profile target capability per task

Before burning GPU hours, check which tasks the target can actually do — tasks where the verbose prompt also fails contribute zero gradient and dilute the budget.

```bash
bash training/hf_job_profile.sh
# Pushes profile JSONL to ${PUSH_TO_HUB}/profiles/baseline_<target_slug>.jsonl
```

The output is a per-task `description_baseline` (verbose-prompt accuracy on that target). Any task at 0.0 means the target genuinely can't do it — drop those from your active task list before training.

---

## 2. Single-step training (the "hero" recipe)

`training/train_grpo.py` — vanilla TRL GRPO, single-step rollouts, LoRA r=16/α=32 on Qwen3-1.7B.

```bash
PUSH_TO_HUB=your-username/your-hero-adapter \
TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
ENABLE_THINKING=false \
  bash training/hf_job_train.sh
```

What this does:
- 500 GRPO steps × `num_generations=8` × 6 hidden test inputs per task → ≈24,000 target inferences.
- Reward per rollout: `raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped `[−0.5, 1.3]`.
- Anti-collapse: `MIN_TOKENS_FLOOR=5` (without it the agent finds 1-token tokenizer attacks).
- Pushes to the destination repo: `adapter_final/`, `train_metrics.jsonl`, `plots/{reward_curve,length_curve,breakdown}.png`.
- Wallclock: ≈3h on L40S.

Key flags (override via env vars in the launcher script):
- `--max-completion-length 256` — agent's per-turn output budget. Bump to 768+ if you enable thinking (see below).
- `--enable-thinking` / `--no-enable-thinking` — Qwen3's `<think>...</think>` chat template. We ship OFF; ON loses on reward and adds 30% tokens.
- `--num-generations 8` — GRPO group size. Smaller = faster, less stable.

---

## 3. Multi-step training (the 3-turn variant)

`training/train_grpo_multistep.py` — hand-rolled trajectory-level GRPO. TRL's GRPO can't do multi-turn credit assignment cleanly (it expects one prompt → one scalar reward), so this is a separate trainer.

```bash
SFT_ADAPTER=your-username/your-hero-adapter \
PUSH_TO_HUB=your-username/your-multistep-adapter \
TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  bash training/hf_job_train_multistep.sh
```

What's different:
- **Warm-start from the hero adapter** (non-optional in practice — cold starts burn 1000+ steps rediscovering single-turn behavior).
- 150 steps × 8 trajectories × 3 turns = 24 turns/step.
- Each task's 6 hidden test inputs split into a 2-example *feedback slice* (revealed across turns 1+2 with target outputs) and a 4-example *scoring slice* (final-turn prompt judged on these only).
- Reward = final-turn additive rubric. REINFORCE-style policy gradient over full-trajectory action tokens, KL penalty against a snapshot of the warm-start LoRA.
- Memory config (required for L40S): `--gradient-checkpointing` ON, `--update-micro-batch 2`, `--max-prompt-tokens 2048`, `--max-new-tokens 384`.
- Wallclock: ≈3.5h.

After training, the adapter ships to `${PUSH_TO_HUB}/adapter_final/`. **For the eval step to find it, you must promote it to the repo root** (PEFT looks for `adapter_config.json` at root):

```bash
mkdir -p /tmp/adapter && \
  hf download $PUSH_TO_HUB --include "adapter_final/*" --local-dir /tmp/adapter
cd /tmp/adapter/adapter_final
hf upload $PUSH_TO_HUB . . --commit-message "Promote adapter to repo root"
```

We hit this in the multi-step run; future versions of `train_grpo_multistep.py` should push to root directly.

---

## 4. Eval — base vs trained, on the same target

`training/eval_before_after.py` — runs the agent on every task with no adapter (`base`), then again with the trained adapter (`trained`). Output is one JSONL per label, one row per (task × seed). Both pushed to the adapter repo's `evals/` folder.

```bash
ADAPTER_REPO=your-username/your-hero-adapter \
TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
ENABLE_THINKING=false \
  bash training/hf_job_eval.sh both    # dispatches base + trained as two jobs
```

Modes:
- `base` — eval untrained Qwen3 only (cheaper, run once per target).
- `trained` — eval the adapter only (when you've added a new adapter and base eval already exists).
- `both` — convenience for a fresh adapter.

Each job is ≈15 min on L40S. Output rows include `task_id`, `category`, `agent_prompt`, `tokens`, `raw_task_score`, `reward`, `leakage_penalty`, `baseline_zero_shot`. Aggregate however you like — see `build_before_after_csv.py` for the canonical roll-up.

---

## 5. Build the demo CSV

`training/build_before_after_csv.py` — joins the verbose human prompts (from the env's task definitions) with the eval JSONLs to produce a 90-row demo CSV: `verbose / base / trained` prompts side by side, plus accuracy/token/reward deltas, plus 3 sample test inputs per task for the Gradio demo.

```bash
python training/build_before_after_csv.py \
  --base-jsonl    evals/eval_base.jsonl \
  --trained-jsonl evals/eval_trained.jsonl \
  --output-csv    evals/qwen_to_llama_demo.csv \
  --min-verbose-accuracy 0.0   # set to >0 to drop dead-target tasks
  # Optional: --push-to-hub your-name/your-adapter-repo to upload directly.
```

Then upload the CSV to the adapter repo so the demo Space can fetch it:

```bash
hf upload $ADAPTER_REPO evals/qwen_to_llama_demo.csv evals/qwen_to_llama_demo.csv
```

---

## 6. Plots (already auto-generated by the trainer)

`training/make_plots.py` is invoked by `hf_job_train.sh` at the end of training. If you want to re-render after the fact:

```bash
python training/make_plots.py \
  --metrics train_metrics.jsonl \
  --out-dir plots/
```

Produces `reward_curve.png`, `length_curve.png`, `breakdown.png`. Same plots are embedded in the BLOG_POST and Trackio dashboard.

---

## 7. Replay training metrics to Trackio (optional dashboard)

`training/replay_to_trackio.py` — post-hoc replay of `train_metrics.jsonl` files into the [Trackio](https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio) dashboard Space. Useful for cross-run comparison (hero vs multi-step vs Qwen→Qwen control).

Edit the `RUNS` dict at the top of the script to point at your adapter repos, then:

```bash
python training/replay_to_trackio.py
```

The script logs each run as a separate Trackio experiment with full per-step metrics. The dashboard auto-reflows.

---

## 8. End-to-end checklist

For a clean from-scratch reproduction of the hero release:

```bash
# 0) auth
hf auth login

# 1) (optional) profile
bash training/hf_job_profile.sh

# 2) train hero
PUSH_TO_HUB=your-name/prompt-golf-hero \
  bash training/hf_job_train.sh                        # ≈3h

# 3) eval base + trained
ADAPTER_REPO=your-name/prompt-golf-hero \
  bash training/hf_job_eval.sh both                    # 2 × ≈15min

# 4) build demo CSV
#    First pull the JSONLs locally, then merge them.
hf download your-name/prompt-golf-hero --include "evals/*.jsonl" --local-dir .
python training/build_before_after_csv.py \
  --base-jsonl    evals/eval_base.jsonl \
  --trained-jsonl evals/eval_trained.jsonl \
  --output-csv    evals/qwen_to_llama_demo.csv \
  --push-to-hub   your-name/prompt-golf-hero

# 5) (optional) Trackio replay
python training/replay_to_trackio.py
```

For the multi-step variant, swap step 2:

```bash
SFT_ADAPTER=your-name/prompt-golf-hero \
PUSH_TO_HUB=your-name/prompt-golf-multistep \
  bash training/hf_job_train_multistep.sh              # ≈3.5h
# then promote adapter_final/ to repo root before eval (see §3)
```

---

## File index

| Script | Role |
|---|---|
| [`train_grpo.py`](./train_grpo.py) | Single-step TRL GRPO trainer (the hero recipe). |
| [`train_grpo_multistep.py`](./train_grpo_multistep.py) | Trajectory-level GRPO for multi-turn episodes. |
| [`eval_before_after.py`](./eval_before_after.py) | Base + trained-adapter eval harness, one JSONL per label. |
| [`profile_baseline.py`](./profile_baseline.py) | Per-task target-capability profiler (verbose-prompt accuracy on a target). |
| [`build_before_after_csv.py`](./build_before_after_csv.py) | Merge eval JSONLs + verbose prompts into the demo CSV. |
| [`make_plots.py`](./make_plots.py) | Reward / length / breakdown curves from `train_metrics.jsonl`. |
| [`replay_to_trackio.py`](./replay_to_trackio.py) | Post-hoc Trackio dashboard replay across runs. |
| [`hf_job_train.sh`](./hf_job_train.sh) | HF Jobs launcher for single-step training. |
| [`hf_job_train_multistep.sh`](./hf_job_train_multistep.sh) | HF Jobs launcher for multi-step training. |
| [`hf_job_eval.sh`](./hf_job_eval.sh) | HF Jobs launcher for base + trained eval. |
| [`hf_job_profile.sh`](./hf_job_profile.sh) | HF Jobs launcher for the capability profiler. |
| [`requirements.txt`](./requirements.txt) | Pinned deps for the training jobs (matches OpenEnv-official torch/transformers/trl). |

---

## Common pitfalls (we hit all of these)

- **HF `whoami` rate limit** blocks back-to-back job dispatches. If you hit it, wait 5–25 min before retrying. Don't poll-loop.
- **Adapter pushed to a subfolder** (e.g. `adapter_final/`) breaks `PeftModel.from_pretrained()` which expects `adapter_config.json` at the repo root. Promote files to root before running eval.
- **OOM on L40S during multi-step training** if you forget `--gradient-checkpointing`. Set the four memory flags in §3 from the start.
- **Stale `TASK_NAMES` constant in `inference.py`** dropping new tasks from training. Use the lazy `_all_task_ids()` helper instead.
- **Cold-starting multi-turn** wastes compute. Always warm-start from a single-turn hero adapter.

If something else breaks, check the [`BLOG_POST.md`](../BLOG_POST.md) "Notes on training (for the curious)" section and the failure-mode bullets in "What the agent learned." Most issues we ran into are documented there.
