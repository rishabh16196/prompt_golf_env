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
| [`prompt-golf-multistep-llama`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama) | Qwen→Llama multi-turn | trajectory-level GRPO adapter |
| [`prompt-golf-llama-self`](https://huggingface.co/rishabh16196/prompt-golf-llama-self) | Llama→Llama self-improvement | adapter where Llama writes prompts for itself |

### Training pipeline ([`training/`](https://github.com/rishabh16196/prompt_golf_env/tree/main/training))

| File | Role |
|---|---|
| [`training/train_grpo.py`](./training/train_grpo.py) | TRL GRPO single-step trainer |
| [`training/train_grpo_multistep.py`](./training/train_grpo_multistep.py) | hand-rolled trajectory-level GRPO (multi-turn) |
| [`training/eval_before_after.py`](./training/eval_before_after.py) | base + trained-adapter eval harness |
| [`training/profile_baseline.py`](./training/profile_baseline.py) | per-task target-capability profiler |
| [`training/build_before_after_csv.py`](./training/build_before_after_csv.py) | merge eval JSONLs into the demo CSV |
| [`training/replay_to_trackio.py`](./training/replay_to_trackio.py) | post-hoc replay of `train_metrics.jsonl` into the Trackio dashboard Space |
| [`training/hf_job_train.sh`](./training/hf_job_train.sh) / [`hf_job_train_multistep.sh`](./training/hf_job_train_multistep.sh) / [`hf_job_eval.sh`](./training/hf_job_eval.sh) / [`hf_job_profile.sh`](./training/hf_job_profile.sh) | HuggingFace Jobs launchers |

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

→ **80% accuracy retention at 55% of the verbose token count.**  Peak compression: **37× on long-context policy tasks** (e.g. a 737-token MSN ad-creative policy → a 20-token classifier prompt).

The trained prompt **beats the human verbose prompt on 48 of 87 tasks (55%)** under the same rubric (`raw_score − 0.5·baseline − 0.002·tokens`). On the rest, the accuracy drop on hard tasks outweighs the length savings — those are the cases where the trained agent compressed too aggressively to keep up with Llama's verbose-prompt capability ceiling.

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

### Cross-family thinking=ON variant

Identical training setup but with Qwen3's `<think>...</think>` chat template enabled.

![Qwen→Llama thinking-ON reward curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/resolve/main/plots/reward_curve.png)

| | thinking=OFF (hero) | thinking=ON |
|---|---|---|
| Trained accuracy | 0.523 | **0.539** |
| Trained reward | **+0.426** | +0.379 |
| Mean tokens | **35** | 46 |

OFF wins on reward and compression; ON wins on accuracy by 1.6pp at a 30% token cost. We ship OFF as the default; ON is a different operating point on the accuracy/length frontier.

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
    prompt_golf_environment.py         # core Env (single + multi-turn)
    target_model.py                    # frozen-target wrapper (HF + mock)
    judge.py                           # Qwen3-8B 8-bit judge
    scorer.py                          # 21+ scorers (structural + LLM judge)
    rubrics.py                         # additive reward composition
    tasks.py / tasks_v2.py / tasks_tough.py / tasks_policy.py    # 90-task bank
  training/                            # see Links → Training pipeline
  ui/ + space-demo/                    # Gradio demos
  BLOG_POST.md                         # writeup
```
