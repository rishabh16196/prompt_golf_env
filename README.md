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

> An RL environment where the agent's *action* is a prompt and the *reward* is how well that prompt steers a frozen target LLM — divided by how long the prompt is.

A Qwen3-1.7B agent (trained via TRL GRPO) learns to write **35-token prompts** that get a frozen Llama-3.2-3B target to **80% of the accuracy** of human-written 250-token prompts.

## Links

- 🌍 **Env (this Space):** https://huggingface.co/spaces/rishabh16196/prompt_golf_env
- 🎛️ **Live demo (Gradio):** https://huggingface.co/spaces/rishabh16196/prompt-golf-demo
- 📊 **Training dashboard (Trackio):** https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio
- 🐙 **GitHub mirror:** https://github.com/rishabh16196/prompt_golf_env
- 🛠️ **Training pipeline:** [`training/`](https://github.com/rishabh16196/prompt_golf_env/tree/main/training) — full GRPO trainers, eval harness, profilers, HF Jobs launchers
- 📝 **Blog post:** [`BLOG_POST.md`](./BLOG_POST.md)

### Trained adapters & data

| Repo | Setup | What's in it |
|---|---|---|
| [`prompt-golf-qwen-to-llama-nothink`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink) | Qwen→Llama, thinking=OFF (**hero**) | adapter + plots + train_metrics + base/trained eval JSONLs + demo CSV |
| [`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama) | Qwen→Llama, thinking=ON | same artifacts (A/B variant) |
| [`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b) | Qwen→Qwen same-family (control) | same artifacts |
| [`prompt-golf-multistep-llama`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama) | Qwen→Llama, **multi-turn (3 turns)** | adapter + train_metrics + base/trained eval JSONLs |

### Training pipeline ([`training/`](https://github.com/rishabh16196/prompt_golf_env/tree/main/training))

| File | Role |
|---|---|
| [`training/train_grpo.py`](./training/train_grpo.py) | TRL GRPO single-step trainer |
| [`training/eval_before_after.py`](./training/eval_before_after.py) | base + trained-adapter eval harness |
| [`training/profile_baseline.py`](./training/profile_baseline.py) | per-task target-capability profiler |
| [`training/build_before_after_csv.py`](./training/build_before_after_csv.py) | merge eval JSONLs into the demo CSV |
| [`training/replay_to_trackio.py`](./training/replay_to_trackio.py) | post-hoc replay of `train_metrics.jsonl` into the Trackio dashboard Space |
| [`training/hf_job_train.sh`](./training/hf_job_train.sh) / [`hf_job_eval.sh`](./training/hf_job_eval.sh) / [`hf_job_profile.sh`](./training/hf_job_profile.sh) | HuggingFace Jobs launchers |

---

## Hero result — Qwen3-1.7B agent → Llama-3.2-3B target

500 GRPO steps on a 90-task bank. Same setup as the demo Space.

![reward curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/reward_curve.png)

![length curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/length_curve.png)

![reward-component breakdown](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/breakdown.png)

| Stage | Mean accuracy | Mean tokens |
|---|---|---|
| Verbose human prompt | **0.65** | ~63 |
| Untrained Qwen3-1.7B | 0.48 | ~38 |
| **Trained Qwen3-1.7B + LoRA** | **0.52** | **35** |

→ **80% accuracy retention at 55% of the verbose token count.**

The trained prompt **beats the human verbose prompt on 48 of 87 tasks (55%)** under the same rubric (`raw_score − 0.5·baseline − 0.002·tokens`). On the rest, the accuracy drop on hard tasks outweighs the length savings — those are the cases where the trained agent compressed too aggressively to keep up with Llama's verbose-prompt capability ceiling.

📊 **Demo CSV:** [`evals/qwen_to_llama_demo.csv`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv) — 90 rows × verbose / untrained / trained prompts side by side, with per-task accuracy + reward-advantage columns.

---

## How it works

Each episode is one task:

1. `reset(task=...)` → env returns task description + 3 visible train examples + token budget + target's empty-prompt baseline.
2. Agent outputs a **prompt string** as its action.
3. Env prepends the prompt to ~6 held-out test inputs, runs the **frozen target LLM**, scores each output with the task scorer.
4. `reward = raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped to `[−0.5, 1.3]`.

Multi-turn (`turn_limit > 1`) splits the test pool into a 2-example feedback slice (revealed across turns) and a 4-example scoring slice (only the final-turn prompt is judged). The agent sees its prior prompts + per-example target outputs in the user message, so it can debug across turns without leaking the inputs that judge it.

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

## Variants

### Same-family control: Qwen→Qwen

Same agent (Qwen3-1.7B + LoRA) but target is also Qwen3-1.7B. Different framing: because Qwen target is weak on strict-format tasks, **verbose prompts only get 0.15 accuracy on average** — a much lower bar to beat.

![Qwen→Qwen reward curve](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/resolve/main/plots/reward_curve.png)

| | Qwen→Qwen | Qwen→Llama (hero) |
|---|---|---|
| Trained beats verbose on | **70 / 87 tasks (80%)** | 48 / 87 (55%) |
| Mean reward advantage vs verbose | **+0.085** | -0.057 |
| Verbose accuracy ceiling | 0.15 | 0.65 |

Qwen target's weakness makes this the easier comparison to "win" — the trained agent's compressed prompts beat verbose on 80% of tasks because verbose itself is failing. Cross-family Llama target is a much harder bar to clear, but the absolute accuracy is far higher.

📊 **Demo CSV:** [`evals/qwen_to_qwen_demo.csv`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/blob/main/evals/qwen_to_qwen_demo.csv)

### Cross-family thinking=ON variant

Identical training setup but with Qwen3's `<think>...</think>` chat template enabled.

![Qwen→Llama thinking-ON reward curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/resolve/main/plots/reward_curve.png)

| | thinking=OFF (hero) | thinking=ON |
|---|---|---|
| Trained accuracy | 0.523 | **0.539** |
| Trained reward | **+0.426** | +0.379 |
| Mean tokens | **35** | 46 |

OFF wins on reward and compression; ON wins on accuracy by 1.6pp at a 30% token cost. We ship OFF as the default; ON is a different operating point on the accuracy/length frontier.

📊 **Demo CSV:** [`evals/qwen_to_llama_demo.csv`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/blob/main/evals/qwen_to_llama_demo.csv)

### Multi-turn variant: 3 turns per episode

Same agent + target as the hero, but each episode runs `turn_limit=3` — the agent sees its prior prompts + per-example feedback on a 2-example feedback slice, and only the final-turn prompt is judged on a held-out 4-example scoring slice. Trained with the trajectory-level GRPO trainer (`train_grpo_multistep.py`), 150 steps × 8 trajectories × 3 turns.

| | Single-step hero | Multi-step (3 turns) |
|---|---|---|
| Trained accuracy | 0.523 | **0.576** |
| Trained reward | +0.426 | **+0.440** |
| Mean tokens | **35** | 43.7 |
| Trained beats untrained on | — | 29 / 90 tasks |

**Multi-step is a conditional improvement, not a strict upgrade:**

| Where it wins | Where it loses |
|---|---|
| `reasoning_tough` (5 wins / 0 losses) | `format` (2 wins / 5 losses) |
| `classification_tough` (7 / 2) | `format_three_bullets`, `format_uppercase` (bloat with no accuracy gain) |
| `policy_compression` (1 win — `policy_content_moderation` 0.00 → 0.67) | Dead-target tasks (agent burns tokens trying anyway, e.g. `policy_finreg_communication_review` 17→112 tokens, both 0.00) |

The intuition: multi-turn relaxes length pressure because the agent has room to debug across turns. That helps tasks where the agent needs reasoning room (tough reasoning, tough classification, complex extraction). It hurts tasks where short single-shot prompts already win (format-strict tasks). **Single-step is the right default for cost-sensitive deployments; multi-step is the right pick when accuracy ceilings matter more than token count.**

📊 **Eval JSONLs:** [`evals/eval_base.jsonl`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_base.jsonl) + [`evals/eval_trained.jsonl`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_trained.jsonl)

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
