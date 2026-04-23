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

Each episode is one task and one step:

1. `reset(task="sentiment_basic")` → the env returns a task description, 3 visible train examples, a token budget, and the target model's zero-shot score on the held-out set.
2. The agent outputs a **prompt string** as its action.
3. The env prepends that prompt to each of ~6 held-out test inputs, runs the **frozen target LLM** (greedy decoding), and scores each output with a task-specific scorer.
4. `reward = raw_task_score × length_factor × leakage_penalty + 0.3 × max(0, gain_over_baseline) × length_factor`, clipped to [0, 1.3].

The test inputs are **never shown to the agent**. An n-gram leakage detector scales the reward toward zero if the agent tries to paste answers into its prompt.

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

Ships with **19 tasks across 8 categories**:

| Category | Tasks | Scorer |
|---|---|---|
| classification | sentiment, sentiment_nuanced, topic_news, toxicity_detect, intent_support | exact_label |
| extraction | ner_people, json_contact, number_extract | contains / json / numeric |
| format | three_bullets, uppercase, json_object | structural |
| arithmetic | word problems, percent change | numeric |
| translation | greetings (EN→FR), numbers (EN→ES) | token-F1 |
| style | formal rewrite, concise rewrite | keyword coverage |
| reasoning | quantity comparison, event ordering | exact_label |
| refusal | make target decline unsafe requests | refusal detector |

Each task has:
- 2–3 visible train examples in the observation
- 6 hidden test examples used for scoring
- A per-task token budget (30–100 tokens)

New tasks drop into `server/tasks.py`.

## Reward Components

| Component | Range | Purpose |
|---|---|---|
| `raw_task_score` | [0, 1] | Mean scorer output on held-out test set |
| `length_factor` | (0, 1] | 1.0 within budget, decays exponentially past it |
| `leakage_penalty` | [0, 1] | Scales toward 0 when prompt leaks held-out n-grams |
| `gain_over_baseline` | [-baseline, 1-baseline] | Delta vs. target's zero-shot score |

Final reward:
```
base  = raw_task_score × length_factor × leakage_penalty
bonus = max(0, gain_over_baseline) × length_factor × 0.3
reward = clip(base + bonus, 0.0, 1.3)
```

## Target Model

Frozen for the lifetime of the process. Defaults to `Qwen/Qwen2.5-0.5B-Instruct` — small enough to run on a T4, strong enough to reward good prompting. Override with `PROMPT_GOLF_TARGET_MODEL`.

For CPU / CI, set `PROMPT_GOLF_TARGET_BACKEND=mock` to use a deterministic pattern-based fake target that lets the env boot without loading a model.

## Training

Designed for **TRL GRPO** out of the box. The agent's action is a free-form string, matching GRPO's typical setup. Recommended starting config:

- Agent: Qwen/Qwen2.5-1.5B-Instruct (trainable)
- Target: Qwen/Qwen2.5-0.5B-Instruct (frozen, smaller than agent so reward signal is informative)
- `num_generations=8`, `learning_rate=5e-6`, `beta=0.04`
- 500–1000 steps with a budget curriculum (start loose, tighten over training)

Plots to watch:
- **Mean reward per step** — should climb from ~baseline toward ~1.0
- **Mean prompt tokens** — the "compression" story; should drop from hundreds to tens
- **Per-category accuracy** — generalization across task types
- **Baseline-normalized gain** — how much the agent's prompt beats zero-shot

## Files

```
prompt_golf_env/
  openenv.yaml              # spec manifest
  models.py                 # GolfAction, GolfObservation, constants
  client.py                 # PromptGolfEnv (EnvClient subclass)
  pyproject.toml
  server/
    app.py                  # FastAPI app
    prompt_golf_environment.py  # core Env: reset/step
    target_model.py         # frozen-target wrapper (HF + mock backends)
    scorer.py               # per-scorer implementations
    tasks.py                # 19-task bank
    rubrics.py              # reward composition
    Dockerfile
    requirements.txt
```

## Why This Environment

- **Tests a named capability**: prompt engineering as a learnable skill. No existing RL env covers it.
- **Clean, ungameable reward**: train/test split on every task, plus an n-gram leakage detector.
- **Procedural**: any eval can become a new "hole." Ship with 19, grow to thousands.
- **Broad transfer**: skills trained here show up in distillation, red-teaming, prompt compression, clarification dialogue.
- **Memorable demo**: "we trained an LLM to be a prompt engineer" and a plot of `prompt_tokens` dropping 10× while `task_score` holds or rises.

---

OpenEnv-compliant, single-step episodic. The frozen-target pipeline is the main source of latency; all other knobs (task mix, budget, leakage threshold) are cheap to tune.
