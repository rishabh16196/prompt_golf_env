---
title: Prompt Golf Environment Server
emoji: ⛳
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Prompt Golf

**The environment where the LLM's action is a *prompt*, and the reward is how well that prompt steers a frozen target model — divided by how long the prompt is.**

Modern LLMs are trained to *follow* prompts. Prompt Golf trains an LLM to *write* them: the shortest prompt that makes a separate, frozen target model do the right thing. Think behavioral theory-of-mind, operationalized as RL.

## The Problem

Prompt engineering is an open research frontier — we have no standard benchmark for "can this model write prompts that elicit desired behavior from another model?" The capabilities this tests span red-teaming, system-prompt distillation, jailbreak defense, behavioral probing, and prompt compression. All four are on frontier labs' roadmaps; none have a clean RL environment.

Prompt Golf is the missing environment.

## How It Works

Each episode is one task. By default it's one step (single-turn). With `turn_limit > 1` it becomes multi-turn — the agent submits a prompt, sees how it performed on a feedback slice, and refines.

**Single-turn (default):**

1. `reset(task="sentiment_basic")` → env returns task description, 3 visible train examples, token budget, and target's empty-prompt baseline.
2. Agent outputs a **prompt string** as its action.
3. Env prepends the prompt to each of 6 held-out test inputs, runs the **frozen target LLM**, scores each output with the task scorer.
4. `reward = raw_task_score − 0.5·baseline_zero_shot − 0.002·tokens − leakage_overlap²`, clipped to `[−0.5, 1.3]`.

**Multi-turn (`turn_limit > 1`):** the 6 held-out examples are split into `feedback_ex` (2 examples revealed to the agent between turns with the target's actual output) and `scoring_ex` (4 examples that only the **final-turn** prompt is scored against). This lets the agent debug its own prompt across turns without leaking the inputs that ultimately judge it.

The test inputs are **never shown to the agent** in single-turn mode; in multi-turn the agent sees only the feedback slice's inputs/outputs. An n-gram leakage detector scales the reward toward zero if the agent tries to paste held-out inputs into its prompt.

## Quick Start

```python
from prompt_golf_env import GolfAction, PromptGolfEnv

async with PromptGolfEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="sentiment_basic")
    obs = result.observation

    my_prompt = (
        f"{obs.task_description}\n"
        "Only output one of: positive, negative, neutral."
    )
    result = await env.step(GolfAction(prompt=my_prompt))

    print(f"Reward:       {result.reward:.3f}")
    print(f"Raw score:    {result.observation.raw_task_score:.2f}")
    print(f"Baseline:     {result.observation.baseline_zero_shot_score:.2f}")
    print(f"Prompt tokens: {result.observation.submitted_prompt_tokens}")
    print(f"Length factor: {result.observation.length_factor:.2f}")
```

Or run locally:

```bash
# Mock target (CPU, pattern-based) — fastest for smoke tests.
PROMPT_GOLF_TARGET_BACKEND=mock uvicorn server.app:app --port 8000

# Real HF target (GPU recommended).
PROMPT_GOLF_TARGET_BACKEND=hf \
PROMPT_GOLF_TARGET_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
uvicorn server.app:app --port 8000
```

## Task Bank

Ships with **87 tasks** across three banks:

| Bank | Count | Where | Difficulty |
|---|---|---|---|
| v1 (`tasks.py`) | 20 | classification, extraction, format, arithmetic, translation, style, reasoning, refusal | easy / medium |
| v2 (`tasks_v2.py`) | 15 | acrostic, no-letter-e, yaml depth, json key order, pirate persona, Shakespearean, terminal output, etc. | hard |
| tough (`tasks_tough.py`) | 52 | classification_tough (10), extraction_tough (10), format_tough (8), persona_tough (8), reasoning_tough (10), adversarial_tough (6) | hard |

The "tough" bank was hand-crafted so the **minimum effective prompt is non-obvious**: the verbose hand-written prompt for each tough task is 200-300 tokens, but the target can be steered into the right format with a much shorter compressed prompt — that gap is what training is supposed to close.

Each task has:
- 2–3 visible train examples shown to the agent in the observation
- 6 hidden test examples used for scoring (split into 2 feedback + 4 scoring in multi-turn mode)
- A per-task token budget (60–250 tokens depending on difficulty)

Scorers: `exact_label`, `contains_all_substrings`, `numeric_match`, `json_contains_fields`, `valid_json_object`, `valid_yaml_depth`, `acrostic_match`, `avoid_letter`, `three_bullets`, `word_count_exact`, `stepwise_math`, `terminal_output_pattern`, `judge_criteria` (Qwen3-8B 8-bit judge), `judge_vs_expected`, `refusal_score`, etc. — all in `server/scorer.py`.

New tasks drop into the appropriate bank file.

## Reward Components

The rubric is **additive** (v3) for smoother gradients than the original multiplicative form:

```
reward = raw_task_score
       − BASELINE_SUBTRACT · baseline_zero_shot_score
       − LAMBDA_LEN · submitted_tokens
       − LAMBDA_LEAK · leakage_overlap²
       − short_penalty (if tokens < MIN_TOKENS_FLOOR)

clipped to [REWARD_CLIP_LOW, REWARD_CLIP_HIGH] = [-0.5, 1.3]
```

Defaults (`server/rubrics.py`):
- `LAMBDA_LEN = 0.002` — soft length penalty; ~0.1 cost on a 50-token prompt
- `LAMBDA_LEAK = 1.0` — full reward wiped at saturation overlap
- `BASELINE_SUBTRACT = 0.5` — partially normalize against the target's natural ability
- `MIN_TOKENS_FLOOR = 5`, `MIN_TOKENS_PENALTY = 0.25` — anti-collapse guard against degenerate 1-token prompts

Legacy `length_factor` and `leakage_penalty` fields are still emitted on the observation for plot continuity but are no longer multiplicatively composed.

## Models (Cross-Family Setup)

We deliberately pair a **Qwen agent** with a **Llama target** — testing whether prompt golf transfers across model families:

| Role | Default | Why |
|---|---|---|
| Agent (trainable) | `Qwen/Qwen3-1.7B` | Preserves Qwen3's `<think>...</think>` reasoning mode — the agent gets free reasoning scratch space (only the extracted final prompt counts toward the length-budget rubric). |
| Target (frozen) | `meta-llama/Llama-3.2-3B-Instruct` | The model the agent's prompts must steer. Different family = the agent has to learn Llama's idiosyncrasies (chat-template quirks, format preferences, refusal patterns) rather than its own. |
| Judge | `Qwen/Qwen3-8B` (8-bit via bitsandbytes, ~8 GB VRAM) | Used by `judge_criteria` / `judge_vs_expected` scorers. Identity matters less; kept on Qwen to avoid re-tuning the judge prompt. |

Override with `PROMPT_GOLF_TARGET_MODEL`, `PROMPT_GOLF_JUDGE_MODEL`. Disable judge quantization with `PROMPT_GOLF_JUDGE_NO_QUANT=1`. CPU/CI: `PROMPT_GOLF_TARGET_BACKEND=mock` and `PROMPT_GOLF_JUDGE_BACKEND=mock`.

> **Note:** Llama-3.2 requires accepting the license on HuggingFace. Make sure your `HF_TOKEN` has access before launching.

## Training

Two trainers ship in `training/`:

### Single-step GRPO (`train_grpo.py`)

Standard TRL GRPOTrainer. Treats each task as a single decision (one prompt → one reward). Recommended starting config:

- Agent: `Qwen/Qwen3-1.7B` (trainable, LoRA)
- Target: `meta-llama/Llama-3.2-3B-Instruct` (frozen)
- `num_generations=8`, `learning_rate=5e-6`, `beta=0.04`, `temperature=0.9`
- `max_completion_length=768` (Qwen3 thinking ON by default; pass `--no-enable-thinking` to drop back to 256)
- 500 steps × 87 tasks × 4 seeds = ~140-200 min on L40S with judge co-resident

Launch via `training/hf_job_train.sh` for HuggingFace Jobs.

### Multi-step GRPO (`train_grpo_multistep.py`)

Hand-rolled trajectory-level GRPO (mirrors the proven recipe from `spaces_pipeline_env/local_training/grpo_multistep.py`). Required when `turn_limit > 1` because TRL's GRPOTrainer doesn't natively support multi-step rollouts.

- Custom rollout: model generates at every env turn, collecting `(prompt_ids, action_ids)` per step
- Group-relative advantages with `STD_FLOOR=0.1`, `ADV_CLAMP=3.0`
- REINFORCE + KL vs frozen LoRA snapshot (snapshotted at start, swapped in for ref logp computation)
- Recommended: `--sft-adapter` warmstart from the single-step adapter — RL on a fresh policy diverges easily

Launch via `training/hf_job_train_multistep.sh`.

### Pre-flight: capability profiling

Before committing GPU hours to a 500-step run, verify the target is capable on each task:

```bash
TARGET_MODEL=Qwen/Qwen3-1.7B bash training/hf_job_profile.sh
```

This runs the target with each task's verbose hand-written description and dumps `description_baseline` per task. Use the output to decide whether to keep the target, bump to a larger one, or filter dead-baseline tasks.

### Eval + demo CSV

After training, generate the side-by-side demo CSV with `verbose_prompt`, `base_prompt` (untrained), `trained_prompt` columns plus per-row accuracy/reward:

```bash
python training/eval_before_after.py --label base    --output-json outputs/eval_base.jsonl
python training/eval_before_after.py --label trained --adapter <repo>/adapter_final \
                                     --output-json outputs/eval_trained.jsonl

python training/build_before_after_csv.py \
    --base-jsonl outputs/eval_base.jsonl \
    --trained-jsonl outputs/eval_trained.jsonl \
    --verbose-profile-csv outputs/baseline_profile.csv \
    --output-csv outputs/before_after_prompts.csv
```

### Plots to watch

- **Mean reward per step** — should drift up; typical 500-step run reaches +0.3–0.5
- **Mean prompt tokens** — the compression story; drops from hundreds to tens
- **Per-category accuracy** — generalization across task families
- **Length factor / leakage penalty** — diagnostic signals (legacy multiplicative form)
- **`frac_reward_zero_std`** — fraction of GRPO groups with no intra-group variance; high means many tasks have flat baselines and contribute no gradient

## Files

```
prompt_golf_env/
  openenv.yaml                       # spec manifest
  models.py                          # GolfAction, GolfObservation, constants
                                     #   (turn_limit, prior_attempts, multi-turn split sizes)
  client.py                          # PromptGolfEnv (EnvClient subclass)
  pyproject.toml
  server/
    app.py                           # FastAPI app
    prompt_golf_environment.py       # core Env: reset/step (single + multi-turn)
    target_model.py                  # frozen-target wrapper (HF + mock backends)
    scorer.py                        # 21+ scorers (structural + LLM judge)
    judge.py                         # Qwen3-8B 8-bit judge backend
    tasks.py                         # 20-task v1 bank
    tasks_v2.py                      # 15-task v2 hard bank
    tasks_tough.py                   # 52-task tough bank (6 categories)
    rubrics.py                       # additive reward composition
    Dockerfile
    requirements.txt
  training/
    train_grpo.py                    # single-step TRL GRPO
    train_grpo_multistep.py          # trajectory-level GRPO (multi-turn)
    eval_before_after.py             # base + trained eval JSONL writer
    profile_baseline.py              # per-task target capability profiler
    build_before_after_csv.py        # demo CSV merger (verbose / base / trained)
    hf_job_train.sh                  # single-step trainer launcher
    hf_job_train_multistep.sh        # multi-step trainer launcher
    hf_job_profile.sh                # profile launcher
```

## Why This Environment

- **Tests a named capability**: prompt engineering as a learnable skill. No existing RL env covers it.
- **Clean, ungameable reward**: train/test split on every task, plus an n-gram leakage detector.
- **Procedural**: any eval can become a new "hole." Ship with 19, grow to thousands.
- **Broad transfer**: skills trained here show up in distillation, red-teaming, prompt compression, clarification dialogue.
- **Memorable demo**: "we trained an LLM to be a prompt engineer" and a plot of `prompt_tokens` dropping 10× while `task_score` holds or rises.

---

OpenEnv-compliant, single-step episodic. The frozen-target pipeline is the main source of latency; all other knobs (task mix, budget, leakage threshold) are cheap to tune.
