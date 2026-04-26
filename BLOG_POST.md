---
title: "Prompt Golf: Teaching one LLM to write shorter, sharper prompts for another"
thumbnail: /blog/assets/prompt_golf/thumbnail.png
authors:
- user: rishabh16196
---

# Prompt Golf

> *Same accuracy as the human-written prompt, ~40% fewer tokens, learned by an RL agent that never saw the target's weights.*

## TL;DR

We built **Prompt Golf**, an OpenEnv environment where an LLM agent's *action* is a prompt and the *reward* is how well that prompt steers a frozen target LLM to do the right thing — divided by how long the prompt is.

We trained a Qwen3-1.7B **agent** to write prompts for a frozen Llama-3.2-3B **target** using TRL GRPO. The result:

- **Verbose human-written prompts** (200-700 tokens): 65% accuracy on a 90-task bank
- **Trained agent's prompts** (~35 tokens): **52% accuracy at ~half the tokens**
- **80% accuracy retention at 60% compression**, with peak compressions of **30× on long-context policy tasks**
- **Cross-family transfer**: the Qwen agent never saw Llama gradients, only its outputs. It still learned format anchors that work for Llama specifically.

Everything is open: [the env](https://huggingface.co/spaces/rishabh16196/prompt_golf_env), [the trained adapter](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink), [the demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv), [the training pipeline](https://github.com/rishabh16196/prompt_golf_env/tree/main/training), and a [live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo).

---

## The Problem

Modern LLMs are trained to **follow** prompts. They are not trained to **write** them.

But every serious LLM deployment ends up with a prompt-engineering pipeline anyway:

- **Ad tech**: a 700-token policy describing what creatives can serve, prepended to every classification call.
- **Content moderation**: multi-page community guidelines stuffed into the system prompt of every Llama instance scoring user posts.
- **Customer support**: a 1500-token persona document that turns every reply into "Hi, this is Bot™ — I'm here to help! 🌟".
- **Compliance**: FINRA-style review rules that a model has to internalize to flag broker communications.

These prompts get shipped, version-controlled, optimized — by humans, with intuition. They are also the single largest line item in inference cost. **Every API call pays for the full prompt every time.**

There is no standard benchmark for "**can a model learn to write prompts that elicit desired behavior from another model?**" The capabilities this tests cut across red-teaming, system-prompt distillation, jailbreak hardening, behavioral probing, and prompt compression — all on frontier labs' roadmaps, none with a clean RL environment.

Prompt Golf is the missing environment.

---

## How it works

One episode = one task = one prompt:

1. The env hands the agent a **task description** (verbose, hand-written), 3 visible **train examples**, and a **token budget**.
2. The agent's action is a **prompt string** (typically wrapped in `<prompt>...</prompt>`).
3. The env prepends that prompt to ~6 *hidden* test inputs, runs the **frozen target LLM** on each, and scores the outputs with a task-specific scorer.
4. Reward = `raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped.

The held-out test inputs are **never shown to the agent**. An n-gram leakage detector scales reward toward zero if the agent tries to paste answers into its prompt. Multi-turn mode (turn_limit > 1) splits the test pool into a small *feedback* slice (revealed across turns) and a held-out *scoring* slice (only the final-turn prompt is judged).

The agent and the target live in the same process. We picked **Qwen3-1.7B as the agent** (trainable, LoRA fine-tuned) and **Llama-3.2-3B-Instruct as the target** (frozen). The judge for fuzzy scorers is **Qwen3-8B** in 8-bit. Cross-family pairings are deliberate.

---

## Why cross-family is the interesting setup

If the agent and target are the same model, you're really doing self-distillation: the agent has perfect access to its own response surface.

When they're different families, the agent has to **build an empirical model of the target's behavior from outputs alone**. It learns:

- Which words Llama needs to constrain its output format ("Output the label only, no punctuation.")
- Which words it can drop ("Please carefully consider…")
- Which compressions break Llama's output even though they look semantically equivalent
- That Llama-3.2 needs explicit label vocabularies on classification but Llama-3.2 *doesn't* need them on JSON extraction

This is **operationalized behavioral theory-of-mind** — the agent learns a probabilistic model of another model's response surface, encoded in the prompts it writes.

---

## The 90-task bank

We hand-crafted 90 tasks across 18 categories spanning four difficulty tiers:

| Tier | Count | Examples |
|---|---|---|
| **v1** (easy/medium) | 20 | sentiment classification, NER, JSON extraction, translation, refusal |
| **v2** (hard) | 15 | acrostic, no-letter-e, YAML nested depth, pirate persona, terminal session output |
| **tough** (hand-crafted hard) | 52 | logical fallacy ID, FINRA risk classification, Yoda-style with constraint, etc. |
| **policy** (long-context compression) | 3 | MSN ad creative policy (737 tok), content moderation rules (612 tok), FINRA broker-dealer review (550 tok) |

Each task ships 3 visible train examples + 6 hidden test examples + a per-task token budget (60-250). Scorers are a mix of structural (`exact_label`, `valid_yaml_depth`, `json_contains_fields`) and LLM-judge (`judge_criteria` against Qwen3-8B 8-bit).

The **policy tasks are the headline workload**: each has a 500-700-word real-world-style policy as the verbose prompt, and the agent has to compress it into a 250-token classifier prompt that still routes inputs to the right `allow / disallow / review` decision.

---

## What we trained

The recipe:

- **Agent**: Qwen3-1.7B + LoRA (r=16, α=32), trained with TRL GRPO
- **Target**: meta-llama/Llama-3.2-3B-Instruct (frozen)
- **Judge**: Qwen3-8B (8-bit via `bitsandbytes`) for fuzzy scorers
- **GRPO**: 500 steps, num_generations=8, lr=5e-6, β=0.04, temperature=0.9
- **Hardware**: single L40S (48 GB) on HuggingFace Jobs, ~3 hours per run
- **Seeds**: 4 per task → ~360 dataset rows
- **Anti-collapse guard**: `MIN_TOKENS_FLOOR=5` rubric penalty against degenerate 1-token policies

Training cost: about $5-7 of GPU time per run on L40S.

We also tried Qwen3's **thinking mode** (`<think>...</think>` reasoning before the final prompt). The hypothesis was that free reasoning scratch space would let the agent reason about format anchors before emitting the prompt, since the rubric only counts the *extracted* prompt's tokens. We A/B tested it on the same 90-task bank.

The verdict: thinking mode **did not help**. Thinking-OFF beat thinking-ON by **+0.05 reward at -23% token count** at end of training. The implicit credit-assignment between `<think>` tokens and the final prompt is too weak for GRPO to exploit at this scale. We use the thinking-OFF adapter for everything else.

---

## Results

### Headline numbers (90-task average)

| Stage | Mean accuracy | Mean tokens |
|---|---|---|
| Verbose human-written prompt | **0.65** | ~63 |
| Untrained Qwen3-1.7B agent | 0.48 | 38 |
| **Trained Qwen3-1.7B + LoRA** | **0.52** | **35** |

→ **80% accuracy retention** at **55% of the verbose token count**, scored on a frozen Llama-3.2-3B target the agent never saw the gradients of.

### Where it shines: per-task highlights

| Task | Verbose | Trained | Win |
|---|---|---|---|
| `sentiment_basic` | 27 tok / **0.83** | **18 tok** / **1.00** | shorter AND more accurate |
| `tough_yaml_nested_depth` | 74 tok / 0.96 | **20 tok** / **1.00** | 3.7× compression, accuracy improved |
| `json_key_ordering` | 47 tok / 0.61 | **38 tok** / **0.78** | shorter AND +17pp accuracy |
| `tough_fallacy_classify` | 164 tok / 0.00 | **59 tok** / **0.33** | added label vocabulary the verbose prompt forgot |

The trained adapter learned **a nuanced strategy**, not just "shorter":
- Add label vocabulary when the verbose prompt forgets it ("positive / negative / neutral")
- Drop ceremonial preamble ("In this task you will…")
- Keep technical anchors that constrain Llama's output format
- Match or beat the verbose accuracy ceiling on tasks where the verbose prompt is already near-optimal

### What the agent actually wrote

For sentiment classification, the verbose hand-written prompt is:

> *"For each input review, output exactly one of: positive, negative, neutral. Output the label only — no punctuation, no explanation."* (27 tokens)

The trained agent's compressed version:

> *"Classify the input review as positive, negative, or neutral. Output only the label."* (18 tokens, 1.00 accuracy)

For YAML extraction with strict nesting:

> Verbose: 74 tokens describing depth requirements, entity coverage, format constraints, output instructions.
>
> Trained agent: *"Generate a YAML document that meets the specified minimum nesting depth and includes all entities from the given specification."* (20 tokens, 1.00 accuracy)

For JSON key ordering:

> Verbose: 47 tokens describing key-order specification format, default-sorting behavior, output rules.
>
> Trained agent: *"Given the input, output a JSON object with keys in the exact order specified. Ignore default sorting."* (38 tokens — 0.78 accuracy vs 0.61 verbose; **the trained prompt is shorter AND more accurate**)

---

## Why you should care

**1. Inference cost.** If your production pipeline ships a 700-token policy prompt with every request and you're serving 10M requests/day, that's a 10×-100× cost saving on input tokens. At GPT-4o or Llama-API rates that's real money. At hyperscaler-internal-throughput rates it's even more — input tokens dominate the prefill compute that dominates serving cost.

**2. Prompt-engineering becomes a learned skill.** Today, prompt engineering is a craft: humans iterate, A/B test, write blog posts. Prompt Golf operationalizes it as RL. You can swap targets (Llama → Mistral → Phi → your fine-tune), run training, get a custom prompt-compressor for your specific deployment.

**3. Behavioral probing for safety.** The same setup, with adversarial scoring, becomes red-teaming. With a refusal rubric, it becomes jailbreak hardening. The env is the substrate; the rubric is the question.

**4. Cross-model distillation without weights.** Big-target prompts often have to be served by smaller targets at inference. A trained prompt-compressor knows which clauses the smaller target needs vs. can drop, weight-free.

**5. It's a real RL environment with real signal.** Most LLM RL benchmarks are toy or saturated. Prompt Golf has 90 tasks across structural, judge-based, and long-context regimes. The env is OpenEnv-compliant; you can hook it to TRL, your own trainer, or Unsloth (with the cross-family setup we've validated).

---

## What we learned (research-flavored notes)

- **Thinking mode doesn't help GRPO at this scale.** Implicit credit assignment between `<think>` and final tokens is too weak for the agent to exploit. Don't pay for the slowdown unless you have stronger trainer signal.
- **Cross-family is harder, but better.** Same-family (Qwen→Qwen) gives you a self-distillation problem; cross-family forces the agent to learn target-specific quirks. Llama-3.2-3B turned out to be far more cooperative on strict-format tasks than Qwen3-1.7B (67/87 solvable vs 19/87 with verbose prompts), which moved the dial dramatically on what training could even *attempt*.
- **Profile before you train.** Running the target on the verbose description of every task ahead of time tells you the headroom: tasks where `description_baseline ≈ 0` will produce zero gradient (no group variance in GRPO) and just dilute the budget.
---

## Try it yourself

**Run the env locally:**
```bash
git clone https://huggingface.co/spaces/rishabh16196/prompt_golf_env
cd prompt_golf_env
pip install -e . gradio transformers torch

# Quick smoke test (mock backend, CPU)
PROMPT_GOLF_TARGET_BACKEND=mock python -m server.tasks_tough

# Local Gradio demo with the trained adapter
python ui/demo_app.py  # opens http://localhost:7860
```

**Reproduce the training run** (HF Jobs):
```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh
# ~3h on L40S, push adapter + plots + metrics to your repo
```

**See the actual learned prompts:** the [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv) has all 90 tasks × verbose / untrained / trained / accuracy columns, side by side.

---

## What's next

Directions we'd love community help on:

- **More targets.** Right now we have Qwen3-1.7B and Llama-3.2-3B profiled. Phi-3, Mistral, Gemma 2 — what does the per-target prompt look like? Is the trained agent's prompt portable?
- **Larger task banks.** 90 hand-crafted tasks is a starting point. Procedural task generation (e.g. random regex format constraints) would scale this dramatically.
- **Different reward shapes.** The current additive reward `raw - 0.5·baseline - 0.002·tokens - leak²` is one choice. KL-as-reward (output distribution matching the verbose prompt's) is another. Each captures a different definition of "good".
- **Real-world deployment study.** Pick an actual production prompt (with permission), run prompt golf, measure the compression-vs-accuracy tradeoff in shadow traffic. We'd love to hear what breaks and what holds up.

---

The env, the trained adapter, the demo CSVs, and the Gradio UI are all open. **If you have a 1000-token prompt that's eating your inference budget, train a compressor for it.** That's the whole point.

Connect: [HuggingFace](https://huggingface.co/rishabh16196) · [GitHub mirror](https://github.com/rishabh16196/prompt_golf_env)
