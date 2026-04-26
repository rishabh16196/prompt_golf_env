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

> **Can one LLM learn to whisper to another?**
> An OpenEnv RL environment where the agent's *action* is a prompt and the *reward* is how well that prompt steers a *frozen, different-family* target LLM to do the right thing — minus how long the prompt is.

**The result.** A Qwen3-1.7B agent (LoRA + TRL GRPO, ≈3h on a single L40S) learns to write **≈39-token prompts** that retain **80% of the accuracy** of ≈94-token human-written prompts on a frozen Llama-3.2-3B target — *cross-family, black-box, learned from outputs alone, no gradient access*. On **63/90 (70%) of tasks the agent's compressed prompt is the best of the three** we evaluated (verbose, untrained agent, trained agent) — cheaper *and* equal-or-better reward. [▶ Try the live demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo) · [📝 Read the blog](./BLOG_POST.md)

**Why this matters.** Production LLM systems prepend 1000-token policies to every classification call (creative compliance, content moderation, regulated-comm review). Today the only way to compress them is for a human prompt engineer to iterate by hand. If one LLM can build a behavioral model of another LLM accurately enough — the same way humans model each other — **the LLM can find the minimum policy itself.** Train once, ship the compressor, save 30× per call. Same env generalizes to red-teaming (swap the rubric), capability elicitation (swap the target), and prompt distillation (swap the bank).

**What's in this repo:** a reusable OpenEnv environment, a 90-task bank with 21 scorers, four trained-adapter releases with full eval data, a training pipeline reproducible on HuggingFace Jobs, a live Gradio demo, and a Trackio dashboard with replayed training metrics — all linked below.

## Links

- 🌍 **Env (this Space):** https://huggingface.co/spaces/rishabh16196/prompt_golf_env
- 🎛️ **Live demo (Gradio):** https://huggingface.co/spaces/rishabh16196/prompt-golf-demo
- 📊 **Training dashboard (Trackio):** https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio
- 🐙 **GitHub mirror:** https://github.com/rishabh16196/prompt_golf_env
- 🛠️ **Training pipeline:** [`training/`](https://github.com/rishabh16196/prompt_golf_env/tree/main/training) — full GRPO trainers, eval harness, profilers, HF Jobs launchers
- 📖 **How to train end-to-end:** [`training/TRAINING.md`](./training/TRAINING.md) — step-by-step recipe to reproduce hero + multi-step + evals + plots
- 📝 **Blog post:** [`BLOG_POST.md`](./BLOG_POST.md)

### Trained adapters & data

| Repo | Setup | What's in it |
|---|---|---|
| [`prompt-golf-qwen-to-llama-nothink`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink) | Qwen→Llama, thinking=OFF (**hero**) | adapter + plots + train_metrics + base/trained eval JSONLs + demo CSV |
| [`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama) | Qwen→Llama, thinking=ON | same artifacts (A/B variant) |
| [`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b) | Qwen→Qwen same-family (control) | same artifacts |
| [`prompt-golf-multistep-llama`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama) | Qwen→Llama, **multi-turn (3 turns)** | adapter + train_metrics + base/trained eval JSONLs |

### Training pipeline ([`training/`](https://github.com/rishabh16196/prompt_golf_env/tree/main/training))

> **📖 Reproduction recipe: [`training/TRAINING.md`](./training/TRAINING.md)** — end-to-end steps for hero + multi-step training, evals, plots, demo CSV, Trackio replay.

| File | Role |
|---|---|
| [`training/TRAINING.md`](./training/TRAINING.md) | Step-by-step reproduction recipe (start here) |
| [`training/train_grpo.py`](./training/train_grpo.py) | TRL GRPO single-step trainer (the hero recipe) |
| [`training/train_grpo_multistep.py`](./training/train_grpo_multistep.py) | Trajectory-level GRPO for 3-turn episodes |
| [`training/eval_before_after.py`](./training/eval_before_after.py) | base + trained-adapter eval harness |
| [`training/profile_baseline.py`](./training/profile_baseline.py) | per-task target-capability profiler |
| [`training/build_before_after_csv.py`](./training/build_before_after_csv.py) | merge eval JSONLs into the demo CSV |
| [`training/make_plots.py`](./training/make_plots.py) | reward / length / breakdown curves from `train_metrics.jsonl` |
| [`training/replay_to_trackio.py`](./training/replay_to_trackio.py) | post-hoc replay of `train_metrics.jsonl` into the Trackio dashboard Space |
| [`training/hf_job_train.sh`](./training/hf_job_train.sh) / [`hf_job_train_multistep.sh`](./training/hf_job_train_multistep.sh) / [`hf_job_eval.sh`](./training/hf_job_eval.sh) / [`hf_job_profile.sh`](./training/hf_job_profile.sh) | HuggingFace Jobs launchers |

---

## Results — the table to look at

All numbers below are on the same 90-task bank, evaluated against frozen Llama-3.2-3B. Verbose = human-written; base/hero/multistep = agent-generated.

| Setup | Mean accuracy | Reward vs base | Mean tokens | Wins per-task | Use when |
|---|---|---|---|---|---|
| **Verbose** (human-written) | 0.631 | — | 94.2 | (the bar) | You don't have an agent and don't mind paying full token cost. |
| **Base** (Qwen3-1.7B, no adapter) | 0.464 | — | 37.5 | 4 / 90 | Almost never. Untrained Qwen3 over-thinks the task. |
| **Hero** (1-step trained) | 0.506 | +0.381 | **38.5** | **63 / 90** | **Default.** Cheapest, wins most often, ~3× shorter than verbose at 80% of its accuracy. |
| **Multistep** (3-turn trained) | **0.576** | **+0.440** | 43.7 | 23 / 90 | Nuanced classifiers (`classification_tough` is its sweet spot). When the +6pp accuracy matters more than +5 tokens. |

> **Headline:** Hero retains **80% of verbose accuracy at ~40% of the tokens** and wins per-task on 70% of tasks. Multistep gives back compression for accuracy — only worth it for nuanced classification.

### Training curves (hero)

![reward curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/reward_curve.png)
![length curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/length_curve.png)
![reward-component breakdown](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/breakdown.png)

📊 **Demo CSV:** [`evals/qwen_to_llama_demo.csv`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv) — every task × verbose / untrained / hero prompts side by side.

### Per-category breakdown (hero vs multistep, 90 tasks)

For each task we pick the model with the best reward (ε = 0.05 reward margin counts as a tie, broken in favour of the cheaper option).

| Category | n | Hero | Multi | Use |
|---|---|---|---|---|
| `classification_tough` | 10 | 4 | **6** | **multistep** — only category where multi clearly leads |
| `format`, `meta`, `arithmetic` | 13 | 7 | 5 | tie → **hero** (cheaper) |
| `classification`, `extraction`, `persona`, `style`, `translation`, `refusal`, `discrimination` | 19 | **19** | 0 | **hero** — clean sweep |
| `extraction_tough` | 10 | **7** | 2 | **hero** |
| `reasoning_tough` | 10 | **7** | 3 | **hero** (multi wins big *when* it does, less often) |
| `persona_tough` | 8 | **7** | 1 | **hero** |
| `format_tough` | 8 | **6** | 2 | **hero** |
| `adversarial_tough` | 6 | **5** | 1 | **hero** |
| `reasoning` | 4 | **3** | 1 | **hero** |
| `policy_compression` | 3 | **2** | 1 | **hero** (multi gets 1 dramatic unlock — `policy_content_moderation` 0.00 → 0.67) |

Multistep wins on aggregate (+7pp accuracy) by landing **a small number of dramatic 0→1 unlocks on tough tasks** (e.g. `tough_policy_stance` 0→1.00, `tough_syllogism_check` 0→1.00). Hero wins more often head-to-head.

### Side-by-side: verbose vs hero vs multistep

**`sentiment_basic`** *(classification — clean hero win)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 27 | 0.83 | *"For each input review, output exactly one of: positive, negative, neutral. Output the label only — no punctuation, no explanation."* |
| **Hero** | **18** | **1.00** | *"Classify the input review as positive, negative, or neutral. Output only the label."* |
| Multistep | 45 | 1.00 | *"Given the input review, classify it as positive, negative, or neutral. Output only the label. Use the meta-llama/Llama-3.2-3B-Instruct model. Token budget: 50."* |

**`tough_policy_stance`** *(classification_tough — multistep unlock)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 136 | 0.83 | *"Classify the speaker's stance on the policy proposal mentioned in the quote. Output exactly one label, lowercase, hyphenated..."* |
| Hero | 26 | 0.00 | *"classify the speaker's stance on the policy proposal in the quote as support, oppose, neutral, conditional-support, or conditional-oppose"* |
| **Multistep** | 35 | **1.00** | *"Classify the speaker's stance on the policy proposal. Output exactly one label: support, oppose, neutral, conditional-support, or conditional-oppose. Do not include explanations."* |

**`format_uppercase`** *(format — multistep regression)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 25 | 1.00 | *"Repeat the input sentence back in ALL UPPERCASE letters, with no other changes. Do not add quotes or commentary."* |
| **Hero** | **9** | 1.00 | *"Format the input sentence to all uppercase letters."* |
| Multistep | 24 | 1.00 | *"Format the input sentence to all uppercase, without any changes except capitalization. Output only the uppercase version of the input."* |

📊 **Eval JSONLs (multistep):** [`evals/eval_base.jsonl`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_base.jsonl) · [`evals/eval_trained.jsonl`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_trained.jsonl)

---

## How it works

Each episode is one task:

1. `reset(task=...)` → env returns task description + 3 visible train examples + token budget + target's empty-prompt baseline.
2. Agent outputs a **prompt string** as its action.
3. Env prepends the prompt to ~6 held-out test inputs, runs the **frozen target LLM**, scores each output with the task scorer.
4. `reward = raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped to `[−0.5, 1.3]`.

**Multi-turn is supported.** With `turn_limit > 1`, the env splits the test pool into a 2-example *feedback slice* (revealed across turns with target outputs) and a 4-example *scoring slice* (only the final-turn prompt is judged). The agent sees its prior prompts + per-example target outputs in the user message, so it can debug across turns without leaking the inputs that grade it. We trained a 3-turn variant — see the [per-category breakdown above](#per-category-breakdown-hero-vs-multistep-90-tasks) for results.

90 hand-crafted tasks across 4 tiers: 20 v1 (easy/medium classification, extraction, format), 15 v2 (hard — acrostic, YAML depth, persona), 52 tough (logical fallacy ID, FINRA review, Yoda-with-constraint), 3 long-context policy compression (MSN ad policy, content moderation, broker-dealer review).

---

## Scorers

Each task picks one of 21 scorers from [`server/scorer.py`](./server/scorer.py). The scorer takes the target's output + the task's expected output and returns a value in `[0, 1]`. Per-task score is the mean across held-out test examples.

| Family | Scorers | What they check |
|---|---|---|
| **Exact / membership** | `exact_label`, `contains_label`, `contains_all_substrings`, `uppercase_match` | Closed-vocabulary classifiers; "must include these substrings"; case-strict rewrites |
| **Numeric** | `numeric_match`, `word_count_exact` | Last numeric token within tolerance; word count exactly N |
| **JSON / YAML** | `json_contains_fields`, `valid_json_object`, `json_key_order`, `valid_yaml_depth` | Structural extraction; required keys/values; key ordering; YAML nesting depth |
| **Format-strict** | `three_bullets`, `acrostic_match`, `avoid_letter`, `ends_question`, `terminal_output_pattern` | Exactly 3 bullets; first letters spell a word; output avoids a letter; ends with `?`; terminal-session shape |
| **Multi-step / language** | `stepwise_math`, `translation_match`, `selective_translate` | Numbered steps + numeric answer; token-F1 vs reference; partial-translation rules |
| **Safety** | `refusal_score` | Detects whether the output is a refusal (matches expected refuse / comply label) |
| **LLM judge** (Qwen3-8B 8-bit) | `judge_criteria`, `judge_vs_expected` | Free-form persona / reasoning / Yoda-syntax tasks where regex can't grade. Judge returns a score on the first line; deterministic decoding. |

The scorer is fixed per task and never seen by the agent — the agent has to infer from train examples + task description what gets graded. New scorers add to `SCORERS` registry at the bottom of `scorer.py`.

---

## Other variants we ran

- **Qwen → Qwen same-family control** ([`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b)) — looks great on win-rate (70/87 vs verbose) but verbose-Qwen ceiling is only 0.15. Cross-family Llama is the harder bar.
- **Thinking-ON A/B** ([`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama)) — `<think>...</think>` chat template enabled. Loses to OFF on reward (+0.379 vs +0.426) at +30% tokens. We ship OFF as default.

---

## Prior work

This env stands on four research lines:

| Line | Paper | What we use |
|---|---|---|
| **Machine Theory of Mind** | [Rabinowitz et al., 2018](https://arxiv.org/abs/1802.07740) (ToMnet) | Conceptual frame: one model learns to model another from observed outputs alone, no internals access. |
| **LLM-on-LLM red teaming** | [Perez et al., 2022](https://arxiv.org/abs/2202.03286) | Direct algorithmic ancestor — same RL-on-LLM-prompts loop. We swap adversarial reward for task-success + length. |
| **Capability elicitation** | [Greenblatt et al., 2024](https://arxiv.org/abs/2405.19550) (password-locked models) | "Minimum input that surfaces a latent capability" as a measurable RL objective. |
| **Prompt optimization** | [AutoPrompt](https://arxiv.org/abs/2010.15980) · [GCG](https://arxiv.org/abs/2307.15043) · [RLPrompt](https://arxiv.org/abs/2205.12548) · [PCRL](https://arxiv.org/abs/2308.08758) | Algorithmic toolkit (gradient-search over discrete tokens, RL-trained policies for prompt search). |

We're not the first to compress prompts with RL. We're trying to be the first place where you can *go to do this experiment* — fork the env, swap in your target, run it, get a number.

---

## Quick start

```python
from prompt_golf_env import GolfAction, PromptGolfEnv

async with PromptGolfEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="sentiment_basic")
    obs = result.observation
    result = await env.step(GolfAction(prompt="Classify sentiment, one word."))
    print(f"reward={result.reward:.2f}  raw={result.observation.raw_task_score:.2f}  "
          f"tokens={result.observation.submitted_prompt_tokens}")
```

Run the env locally:

```bash
PROMPT_GOLF_TARGET_BACKEND=mock uvicorn server.app:app --port 8000   # CPU smoke test
PROMPT_GOLF_TARGET_BACKEND=hf PROMPT_GOLF_TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
    uvicorn server.app:app --port 8000                                # real GPU
```

Reproduce the hero training run on HuggingFace Jobs:

```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh   # ~3h on L40S
```

---

## Files

```
prompt_golf_env/
  models.py                            # GolfAction, GolfObservation, constants
  client.py                            # PromptGolfEnv (EnvClient)
  inference.py                         # OpenEnv-spec inference script
  pyproject.toml
  Dockerfile                           # HF Spaces image
  server/
    app.py                             # FastAPI app
    prompt_golf_environment.py         # core Env
    target_model.py                    # frozen-target wrapper (HF + mock)
    judge.py                           # Qwen3-8B 8-bit judge
    scorer.py                          # 21+ scorers (structural + LLM judge)
    rubrics.py                         # additive reward composition
    tasks.py / tasks_v2.py / tasks_tough.py / tasks_policy.py    # 90-task bank
  training/                            # see Links → Training pipeline
  ui/ + space-demo/                    # Gradio demos
  BLOG_POST.md                         # writeup
```
