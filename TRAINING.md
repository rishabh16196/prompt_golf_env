# Training Guide — Prompt Golf on HF Jobs

End-to-end RL training for Prompt Golf using TRL's GRPO on a HuggingFace Jobs GPU. The whole pipeline is a single `hf jobs run` command.

---

## TL;DR

```bash
# 0. Push this env repo to HF Hub (or GitHub) so the job can clone it.
hf repo create rishabh16196/prompt_golf_env --repo-type=space --space-sdk=docker
git remote add hf https://huggingface.co/spaces/rishabh16196/prompt_golf_env
git push hf main

# 1. Login once with a WRITE token.
hf auth login

# 2. Train (one command; produces adapter pushed to Hub + plots).
bash training/hf_job_train.sh

# 3. Evaluate — base vs trained.
bash training/hf_job_eval.sh both

# 4. Pull results back locally for the demo.
hf jobs logs <job-id>          # inspect training
hf download rishabh16196/prompt-golf-grpo-1.5b --local-dir ./adapter
```

That's the whole loop.

---

## What Gets Trained

| Role | Model | Weights | Where it lives |
|---|---|---|---|
| **Agent** (learns) | Qwen/Qwen2.5-1.5B-Instruct + LoRA | Updated by GRPO | Inside the training process |
| **Target** (frozen) | Qwen/Qwen2.5-0.5B-Instruct | Never changes | Inside the env, same GPU |

The env loads the target in-process (no HTTP server needed for training). Both models share one GPU — on an `a10g-large` the combined memory is ~12 GB including the GRPO reference model.

---

## Task Split

- **16 training tasks** (the default)
- **4 held-out tasks**: `translate_numbers`, `reason_order`, `style_concise`, `refuse_unsafe`

Held-out tasks are never sampled during training. Eval shows generalization — if the trained agent only improves on training tasks, it overfit; if held-out scores also rise, it learned something about prompting in general.

Override the held-out split:

```bash
HELD_OUT=topic_news,arith_percent,ner_people,format_json_object \
  bash training/hf_job_train.sh
```

---

## Reward Recap

Each GRPO step runs `num_generations=8` candidate prompts for a task, scores each with:

```
reward = raw_task_score × length_factor × leakage_penalty
       + 0.3 × max(0, gain_over_baseline) × length_factor
```

GRPO updates the agent to make the higher-reward completions in each group more likely. See [server/rubrics.py](server/rubrics.py) for the exact formula.

---

## Training Stages

### Stage 1 — Launch the job

```bash
bash training/hf_job_train.sh
```

Under the hood this calls:

```
hf jobs run --flavor a10g-large --timeout 3h --secret HF_TOKEN \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  bash -lc "<clone + install + train>"
```

Knobs (all `VAR=value bash training/hf_job_train.sh`):

| var | default | meaning |
|---|---|---|
| `AGENT_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | the trainable LLM |
| `TARGET_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | the frozen LLM |
| `MAX_STEPS` | `500` | GRPO update steps |
| `NUM_GENS` | `8` | candidates per task per step |
| `BATCH_SIZE` | `2` | tasks per step (×num_gens = rollouts/step) |
| `GRAD_ACCUM` | `4` | grad accumulation steps |
| `LR` | `5e-6` | learning rate |
| `BETA` | `0.04` | KL penalty |
| `SEEDS_PER_TASK` | `4` | dataset rows per task |
| `FLAVOR` | `a10g-large` | `t4-medium \| l4x1 \| a10g-large \| a100-large` |
| `TIMEOUT` | `3h` | max job runtime |
| `PUSH_TO_HUB` | `rishabh16196/prompt-golf-grpo-1.5b` | where to save adapter |

### Stage 2 — Watch it

```bash
hf jobs ls
hf jobs logs <job-id> --follow
```

Expect one log line per GRPO step:
```
step   42  loss=+0.012  reward=0.612  avg_tokens=34.8  raw=0.61  lf=1.12  leak=0.98
```

### Stage 3 — Plots

The job renders plots at the end (`outputs/grpo/plots/*.png`). To re-render locally from the metrics file:

```bash
# after `hf download` or from a local training run
python training/make_plots.py \
  --metrics outputs/grpo/train_metrics.jsonl \
  --out-dir outputs/grpo/plots
```

Produces:
- `reward_curve.png` — mean reward per step (headline)
- `length_curve.png` — avg prompt tokens per step (compression story)
- `breakdown.png` — 2×2 grid: reward, tokens, raw score, length factor

### Stage 4 — Eval (before/after)

```bash
bash training/hf_job_eval.sh both
```

Runs two separate jobs — one with the base model, one with the trained adapter — each iterating over all 20 tasks × 3 seeds. Each emits an `eval_<label>.jsonl` plus a printed table:

```
task_id                    reward    raw  tokens     lf   leak
sentiment_basic             1.032   0.83    18.3   1.22   1.00
sentiment_nuanced           0.812   0.67    28.7   1.15   1.00
...
AVERAGE                     0.841              31.4
```

Diff the two averages for the "Average score: X → Y" headline.

---

## Hardware Picks

| Flavor | VRAM | Agent + Target | Notes |
|---|---|---|---|
| `t4-medium` | 16 GB | 1.5B agent + 0.5B target, tight | Cheapest; may OOM with `num_gens=8` |
| `l4x1` | 24 GB | Comfortable | Cheaper than a10g, decent speed |
| **`a10g-large`** | **24 GB** | **Comfortable, fast** | **Recommended** |
| `a100-large` | 80 GB | Fits 3B agent | Overkill unless scaling up |

500-step run at `a10g-large`, `batch=2`, `num_gens=8` ≈ **90–120 min**.

---

## Expected Curves

| Metric | Start | End (500 steps) |
|---|---|---|
| Mean reward | 0.25–0.40 | 0.75–0.95 |
| Avg prompt tokens | 50–70 | 15–25 |
| Raw task score | 0.30–0.45 | 0.65–0.85 |
| Held-out reward | 0.30 | 0.55–0.75 |

If reward plateaus near baseline after 100 steps, the KL penalty (`BETA`) is probably too high or the agent model is too small to explore meaningfully. Bump to `Qwen/Qwen2.5-3B-Instruct` on `a100-large`.

---

## Running It Locally (No HF Jobs)

If you want to iterate faster on a local GPU:

```bash
# Install
pip install -e .
pip install -r training/requirements.txt

# Small smoke test (20 steps, 4 gens, ~10 min on A10G)
python training/train_grpo.py \
  --max-steps 20 \
  --num-generations 4 \
  --output-dir outputs/grpo_smoke

# Eval
python training/eval_before_after.py \
  --label smoke \
  --output-json outputs/eval_smoke.jsonl
```

The scripts and the HF Jobs wrappers share the same CLI — local runs use the same code path as cloud runs.

---

## Files Produced

```
outputs/grpo/
├── config.json              # resolved args for the run
├── train_metrics.jsonl      # per-step metrics (plots read this)
├── checkpoint-XXX/          # HF Trainer checkpoints
├── adapter_final/           # final LoRA + tokenizer (pushed to Hub)
└── plots/
    ├── reward_curve.png
    ├── length_curve.png
    └── breakdown.png

outputs/
├── eval_base.jsonl          # per-episode records, base model
└── eval_trained.jsonl       # per-episode records, trained adapter
```

---

## Common Issues

**"ModuleNotFoundError: prompt_golf_env"**
- The job's `pip install -e .` step failed. Check `pyproject.toml` packages entries — they must point at the top-level package dir.

**"HF_TOKEN not found"**
- `--secret HF_TOKEN` requires `hf auth login` with a WRITE token BEFORE submitting the job. Re-login and resubmit.

**"target_model forward pass hangs"**
- The first reset of any episode triggers target download (~1 GB for 0.5B). Look for `Downloading` in the job logs; this is normal for the first ~60s.

**"reward stuck near baseline"**
- Usually `BETA` too high, `LR` too low, or `num_generations` too small. Try `BETA=0.02 LR=1e-5 NUM_GENS=8`.
- Also check: is the agent producing valid `<prompt>...</prompt>` blocks? If extraction fails, the fallback prompts are identical across generations and GRPO sees no signal. Inspect `train_metrics.jsonl` — if `avg_tokens` stays pinned near a fixed value, parsing is failing.

**"Out of memory"**
- Drop `BATCH_SIZE=1`, raise `GRAD_ACCUM=8`.
- Drop `NUM_GENS=4`.
- Use `--bf16` (default on GPU) and 4-bit agent loading (requires bitsandbytes).

---

## What This Does NOT Do Yet

- Multi-turn golf (agent gets K attempts per task with feedback) — all episodes are single-step.
- vLLM-accelerated rollouts — we use plain HF `generate` for simplicity. Easy upgrade: wrap the target with vLLM and add `--use-vllm` to the config.
- Curriculum scheduling (easy → hard) — all 16 training tasks are mixed uniformly. To add, gate task sampling on `state.global_step`.

Each of these is a one-file change if needed.
