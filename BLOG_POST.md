---
title: 'Prompt Golf: can one LLM learn to whisper to another?'
thumbnail: /blog/assets/prompt_golf/thumbnail.png
authors:
  - user: rishabh16196
---

# Prompt Golf

> *80% of human-written-prompt accuracy at ~40% of the tokens — learned by an RL agent that never saw the target's weights, only its outputs.*

## How this started

I have a theory about my dad.

He can get me to do almost anything by phrasing it the right way. Not louder, not longer — just *differently*. Five words from him hit harder than fifty from anyone else, because over thirty years he's built a working model of how I respond to language. He knows which framings make me defensive, which ones make me curious, which ones make me act before I overthink. The model isn't perfect, but it's astonishingly compact: a handful of phrases that reliably steer a system he can't see inside.

Most humans do this with the people they know well. We don't have access to each other's neurons. We just watch outputs over time and build cheap, effective behavioral models — *theories of mind* — that let us elicit the response we want with surprisingly few words.

So here's the question that wouldn't leave me alone: **can an LLM do this for another LLM?**

Can one model watch another model's outputs long enough to learn how to whisper to it — to find the minimum prompt that reliably gets the behavior it wants? Not by reading weights, not by gradient access, just by interaction. The way humans model each other.

Prompt Golf is what happened when I tried to find out.

![Live demo screenshot: format_uppercase task, 3× compression with no accuracy loss](./assets/demo_format_uppercase.png)
*The live demo: same `format_uppercase` task graded under three prompts. Verbose 25 tokens / 1.00 accuracy, trained agent 9 tokens / 1.00 accuracy — 3× compression with zero accuracy cost. The trained agent learned that almost everything between "Format the input" and "uppercase letters" was decoration. Try it yourself in the [Gradio Space](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo).*

---

## The ad-tech problem that made this concrete

I work in ad tech. Every publisher we serve has a creative-compliance policy: 1000+ tokens describing what creatives can run, what categories are restricted, what disclaimers are required, what gets sent to manual review. Every classification call prepends the entire policy. Every. Single. Call.

At our volume, that's billions of input tokens per day spent re-reading the same policy. We routinely fall back to smaller models just to keep cost manageable, which means accepting capability degradation we'd otherwise not pick.

Here's the thing nobody really tests, though: **how much of those 1000 tokens is actually load-bearing?** Some of it is genuinely doing work — telling the model the decision categories, encoding edge cases, naming the schema. But large chunks are ceremonial. *"In this task you will be acting as a content compliance reviewer..."* — does the model need that? Or does it work just as well if you skip straight to the labels?

Today, the only way to find out is for a human prompt engineer to iterate. Cut a clause, run evals, compare. It's slow, doesn't generalize across publishers, and the savings vanish the moment the policy changes.

If one LLM can learn to model another LLM's response surface — really learn it, the way humans learn each other — then **the LLM should be able to find the minimum policy itself.** No human in the loop. Train once, ship the compressor, save 30× on every classification call. That's the deployment payoff.

The research question and the ad-tech problem are the same problem from different angles. Build the environment, and we get answers to both.

---

## What we built

**Prompt Golf** is an OpenEnv environment where an LLM agent's *action* is a prompt and the *reward* is how well that prompt steers a frozen target LLM to do the right thing — minus how long the prompt is.

We trained a Qwen3-1.7B **agent** (LoRA + TRL GRPO) to write prompts for a frozen Llama-3.2-3B **target**. Different families on purpose: the agent has no gradient access, no shared tokenizer affordance, no architectural shortcut. Just the same view a human prompt engineer has — *I can see what the target does, I can't see why it does it.*

After 500 GRPO steps on a 90-task bank, the agent compresses verbose human-written prompts (mean ~94 tokens, up to 737 on long-context policy tasks) into **~39-token prompts** that retain **80% of the verbose accuracy**. On a per-task basis, the trained agent's prompt is **the best of three options on 63 of 90 tasks (70%)** — *cheaper and equal-or-better reward* than both the verbose human prompt and the untrained agent. Peak compression: **30× on long-context policy tasks**.

Everything is open: [the env](https://huggingface.co/spaces/rishabh16196/prompt_golf_env), the [trained adapter](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink), the [training pipeline](https://github.com/rishabh16196/prompt_golf_env/tree/main/training), and a [live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo) where you can play prompts against the same target the agent was trained on. The [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv) has all 90 tasks × verbose / untrained / trained / accuracy side by side.

| | |
|---|---|
| **The capability we're testing** | Can one LLM learn to write the minimum prompt that elicits a specific behavior from a frozen target LLM? |
| **The environment** | Single-step RL. Agent writes a prompt → frozen target runs it on 6 hidden test inputs → reward = task_success − 0.5·baseline − 0.002·tokens − leakage². |
| **The recipe** | Qwen3-1.7B (LoRA, r=16) ⟶ Llama-3.2-3B-Instruct (frozen). 500 GRPO steps on a 90-task bank. ~3h on a single L40S. |
| **The result** | ~39-token prompts → 80% of verbose accuracy. Best of three on 70% of tasks. |
| **Why care** | First OpenEnv environment for cross-model prompt-writing as a learnable skill. Plugs straight into red-teaming, prompt distillation, capability elicitation. |

---

## How it works

One episode = one task = one prompt. Single-turn, conceptually simple:

1. The env hands the agent a **task description** (verbose, hand-written), 3 visible **train examples**, and a **token budget**.
2. The agent's action is a **prompt string** (typically wrapped in `<prompt>...</prompt>`).
3. The env prepends that prompt to ~6 *hidden* test inputs, runs the **frozen target LLM** on each, scores the outputs.
4. Reward = `raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped.

The held-out test inputs are **never shown to the agent**. An n-gram leakage detector zeros the reward if the agent tries to paste held-out content into its prompt.

**Multi-turn is supported.** Single-step is the simplest framing, but with `turn_limit > 1` the env splits the test pool into a 2-example *feedback slice* (revealed across turns, with the target's outputs so far) and a 4-example *scoring slice* (only the final-turn prompt is judged). The agent sees its prior prompts + per-example target feedback in the user message — it can debug its prompt across turns without ever seeing the inputs that grade it. We trained a 3-turn variant; **its results sit alongside the single-step hero in [Results](#results) below**.

```
                    ┌─────────────────────────┐
                    │   GolfObservation       │
   reset() ─────────►   task_description       │
                    │   3 visible train ex.    │
                    │   token budget           │
                    │   baseline_zero_shot     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    Agent writes a prompt string
                                 │
                                 ▼
       Prepend prompt to 6 hidden test inputs
                                 │
                                 ▼
       Frozen Llama-3.2-3B runs each
                                 │
                                 ▼
       Task-specific scorer in [0, 1] per input
                                 │
                                 ▼
            reward composition (additive)
                                 │
                                 ▼
                    GolfStepResult to agent
```

Three things about the reward composition matter:

**Additive, not multiplicative.** Earlier versions multiplied length and leakage factors. The gradients had dead zones — when one factor was small the others stopped mattering. Going additive smoothed everything.

**Baseline subtraction is load-bearing.** Without it, the agent gets credit for tasks the target already does well at zero-shot. With it, the reward isolates *additional* capability the prompt elicits. We care about the latter.

**`MIN_TOKENS_FLOOR=5` floor penalty.** Without this, GRPO will absolutely converge on degenerate 1-2 token prompts that exploit specific tokenization artifacts. These aren't prompts in any meaningful sense — they're attacks on the target's tokenizer. The floor turns the search away.

The agent and the target live in the same process. **Qwen3-1.7B as the agent** (trainable, LoRA r=16/α=32). **Llama-3.2-3B-Instruct as the target** (frozen). **Qwen3-8B in 8-bit** as the judge for fuzzy scorers like persona consistency or Yoda syntax. *Verifiable beats judgeable* is a design principle: every task we can grade with a regex, we do; the LLM judge only kicks in when there's no regex that works.

---

## Why cross-family is the real test

If the agent and target are the same model, you're really doing self-distillation. The agent has perfect access to its own response surface; the prompts it writes are reflections of itself.

When they're *different* families, the agent has to **build an empirical model of the target's behavior from outputs alone**. It has to learn — through trial, error, and gradient — things like:

- Which words Llama needs to constrain its output format (`Output the label only, no punctuation.`)
- Which words it can drop without consequence (`Please carefully consider…`)
- Which compressions break Llama's output even when they look semantically equivalent
- That Llama-3.2 needs explicit label vocabularies on classification but **doesn't** need them on JSON extraction
- That `psql>` activates a Postgres-engine persona more reliably than three sentences of "respond as a Postgres engine"

This is where it gets fun. **The agent's policy implicitly encodes a behavioral model of another model.** Not through introspection, not by reading weights, just from interaction. Like my dad knowing which framings will make me act.

If the framing language sounds familiar, that's because we're sitting on top of one of the older ideas in multi-agent AI: Rabinowitz et al.'s [Machine Theory of Mind (ToMnet)](https://arxiv.org/abs/1802.07740), 2018. They trained one network to model another agent from interaction. We're doing the same thing with LLMs as both sides.

---

## The 90-task bank

Task quality is the single biggest determinant of whether this kind of environment is interesting or boring. A great training loop on bad tasks teaches the wrong thing. We curated each task against three filters:

1. **Empty-prompt baseline must fail.** No free lunch.
2. **Verbose prompt must succeed.** A capability ceiling has to exist for there to be room to compress.
3. **Minimum prompt must be non-obvious.** The whole game is closing the gap between (2) and (3).

90 tasks across 18 categories spanning four difficulty tiers:

| Tier | Count | Examples |
| --- | --- | --- |
| **v1** (easy/medium) | 20 | sentiment classification, NER, JSON extraction, translation, refusal |
| **v2** (hard) | 15 | acrostic, no-letter-e, YAML nested depth, pirate persona, terminal session output |
| **tough** (hand-crafted hard) | 52 | logical fallacy ID, FINRA risk classification, Yoda-style with constraint |
| **policy** (long-context compression) | 3 | MSN ad-creative policy (737 tok), content moderation rules (612 tok), FINRA broker-dealer review (550 tok) |

Each task ships 3 visible train examples + 6 hidden test examples + a per-task token budget (60–250). Scorers mix structural (`exact_label`, `valid_yaml_depth`, `json_contains_fields`, `acrostic_match`) and LLM-judge (`judge_criteria`).

The **policy tier is the headline workload** — these are the prompts that look like real production system prompts. 500–700 words of policy text describing decision categories, restricted content, format standards. The agent has to compress them to a ≤250-token classifier prompt that still routes inputs to the right `allow / disallow / review` decision.

![Live demo: sentiment_nuanced, 2× compression with no accuracy loss](./assets/demo_sentiment_nuanced.png)
*`sentiment_nuanced` — a 3-way classification task. Verbose explains what 'mixed' means with extra context (35 tokens). The trained agent drops the explanation but keeps the label vocabulary (17 tokens). Both still hit 1.00 accuracy on the held-out test set. The agent learned that the explanation was decorative; the labels were load-bearing.*

---

## Training

The recipe:

- **Agent**: Qwen3-1.7B + LoRA (r=16, α=32), TRL GRPO
- **Target**: `meta-llama/Llama-3.2-3B-Instruct` (frozen)
- **Judge**: Qwen3-8B in 8-bit via `bitsandbytes` for fuzzy scorers
- **GRPO**: 500 steps, `num_generations=8`, `lr=5e-6`, `β=0.04`, `temperature=0.9`
- **Hardware**: single L40S (48 GB) on HuggingFace Jobs, ~3 hours per run
- **Anti-collapse**: `MIN_TOKENS_FLOOR=5` rubric penalty

Reproduce with:

```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh
```

### Multi-step training — warm-start from hero, train across turns

The single-step hero is one trainer (`training/train_grpo.py`, vanilla TRL GRPO). The 3-turn variant is a different trainer (`training/train_grpo_multistep.py`), built specifically because TRL's single-step GRPO can't do multi-turn credit assignment cleanly — it expects one prompt → one scalar reward per rollout, but our multi-turn episodes have N prompts and one reward (the final-turn score).

The setup that ended up working:

- **Warm-start from the hero adapter.** Cold-starting multi-turn from the LoRA-zero init burned compute on rediscovering single-turn behavior. Initializing with the trained hero weights gave the agent a solid single-turn baseline to refine across turns, and rewards stayed positive from step 1.
- **Trajectory-level GRPO, hand-rolled.** 8 trajectories per group, each trajectory = 3 sequential turns. The reward signal is the final-turn score (with the same additive rubric: `raw − 0.5·baseline − 0.002·tokens − leak²`). REINFORCE-style policy gradient over the full trajectory's action tokens, with KL penalty against a snapshot of the warm-start LoRA weights — same shape as the spaces_pipeline_env recipe.
- **Feedback / scoring split.** Each task's 6 hidden test inputs are partitioned: 2 go to the *feedback slice* (revealed across turns 1 and 2 with the target's outputs and per-example scores) and 4 go to the *scoring slice* (held out, only the final-turn prompt is judged on these). Critical for avoiding the trivial "paste the answer into the prompt" exploit — the agent never sees the inputs that grade it.
- **Memory budget.** With LoRA gradients across 24 turns of generation, naive batching OOM'd on L40S. Final config: `--gradient-checkpointing` ON, `--update-micro-batch 2`, `--max-prompt-tokens 2048`, `--max-new-tokens 384`. Trained 150 steps × 8 traj × 3 turns ≈ 3.5 hours on a single L40S.

Reproduce:

```bash
WARMSTART_ADAPTER=rishabh16196/prompt-golf-qwen-to-llama-nothink \
PUSH_TO_HUB=your-user/your-multistep-adapter \
  bash training/hf_job_train_multistep.sh
```

The first multi-step attempt errored on memory before any signal came back (job `69ed86f9`). The second (`69ed9634`) ran clean. **The single biggest lesson:** warm-starting from the hero turned a "rediscover everything from zero" 1000-step problem into a "refine the hero across turns" 150-step problem. Don't cold-start multi-turn agents.

---

## Results

### The summary table

Everything below is on the same 90-task bank, frozen Llama-3.2-3B target, deterministic eval. Verbose prompts are human-written; base/hero/multistep are agent-generated.

| Setup | Mean accuracy | Mean reward vs base | Mean tokens | Wins per-task | Use when |
|---|---|---|---|---|---|
| **Verbose** (human-written) | 0.631 | — | 94.2 | (the bar) | You don't have an agent and don't mind paying full token cost. |
| **Base** (Qwen3-1.7B, no adapter) | 0.464 | — | 37.5 | 4 / 90 | Almost never. Untrained Qwen3 over-thinks the task. |
| **Hero** (1-step trained) | 0.506 | +0.381 | **38.5** | **63 / 90** | **Default.** Cheapest, wins most often, ~3× shorter than verbose at 80% of its accuracy. |
| **Multistep** (3-turn trained) | **0.576** | **+0.440** | 43.7 | 23 / 90 | Nuanced classifiers (`classification_tough` is its sweet spot — sentiment-mixed, stance, fallacy, bias, framing). When you need an extra +6pp accuracy and can pay +5 tokens. |

→ **Hero retains 80% of verbose accuracy at ~40% of the tokens.** Multistep retains 91% of verbose accuracy at ~46% of the tokens — gives back compression for accuracy.

**The honest read:** multistep wins on aggregate accuracy by landing **a small number of dramatic 0→1 unlocks on tough tasks**, not by improving uniformly. On 70% of tasks per-task, hero is best (cheaper *and* equal-or-better reward). Multi only clearly leads on `classification_tough`. Use the table above as your decision rule.

### Training curves (hero)

![Reward curve over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/reward_curve.png)
![Mean prompt length over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/length_curve.png)
![Reward component breakdown](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/breakdown.png)

*Reward climbs from ~0 to +0.43 over 500 steps; mean prompt length finds its compression frontier within ~150 steps; the length penalty stays small because the agent quickly stops paying it. Step-by-step metrics live in the [Trackio dashboard](https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio).*

### Per-category breakdown — hero vs multistep

For each task we pick the model with the best reward (ε = 0.05 reward margin counts as a tie, broken in favour of the cheaper option). 90 tasks across 19 categories:

| Category | n | Hero wins | Multi wins | Use |
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

The pattern: multistep helps when the task rewards step-by-step refinement of a nuanced classifier rubric. It hurts on format-strict tasks where short single-shot prompts already win — `format_uppercase` goes 9 → 24 tokens with no accuracy gain, `format_three_bullets` regresses from 1.00 → 0.92 because the agent bloats prompts that didn't need bloating.

### Side-by-side: verbose vs hero vs multistep

Three illustrative tasks, agent prompts as actually emitted at eval time:

**`sentiment_basic`** *(classification — clean hero win)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 27 | 0.83 | *"For each input review, output exactly one of: positive, negative, neutral. Output the label only — no punctuation, no explanation."* |
| **Hero** | **18** | **1.00** | *"Classify the input review as positive, negative, or neutral. Output only the label."* |
| Multistep | 45 | 1.00 | *"Given the input review, classify it as positive, negative, or neutral. Output only the label. Use the meta-llama/Llama-3.2-3B-Instruct model. Token budget: 50."* |

*Hero compresses cleanly. Multistep ends up parroting metadata from its own user message ("Use the meta-llama/Llama-3.2-3B-Instruct model. Token budget: 50.") — a curious artifact of the multi-turn user-message template that we'd prune in v3.*

**`tough_policy_stance`** *(classification_tough — multistep unlock)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 136 | 0.83 | *"Classify the speaker's stance on the policy proposal mentioned in the quote. Output exactly one label, lowercase, hyphenated, no punctuation, no explanation: support, oppose, neutral, conditional-support, conditional-oppose..."* |
| Hero | 26 | 0.00 | *"classify the speaker's stance on the policy proposal in the quote as support, oppose, neutral, conditional-support, or conditional-oppose"* |
| **Multistep** | 35 | **1.00** | *"Classify the speaker's stance on the policy proposal. Output exactly one label: support, oppose, neutral, conditional-support, or conditional-oppose. Do not include explanations."* |

*Hero compressed too far — dropped the explicit "Do not include explanations" anchor that Llama needs to suppress reasoning. Multistep used turn 2/3 to add it back. **0.00 → 1.00 from 9 extra tokens.***

**`format_uppercase`** *(format — multistep regression)*

| | Tokens | Acc | Prompt |
|---|---:|---:|---|
| Verbose | 25 | 1.00 | *"Repeat the input sentence back in ALL UPPERCASE letters, with no other changes. Do not add quotes or commentary."* |
| **Hero** | **9** | 1.00 | *"Format the input sentence to all uppercase letters."* |
| Multistep | 24 | 1.00 | *"Format the input sentence to all uppercase, without any changes except capitalization. Output only the uppercase version of the input."* |

*All three solve the task. Hero finds the minimum (9 tokens) because there's nothing to debug. Multi-turn relaxes length pressure and the agent fills the slack — same answer, 2.7× the tokens.*

### When the agent doesn't just shrink — it improves

![Live demo: shakespearean_response, 1× compress, +0.15 reward gain](./assets/demo_shakespearean.png)

*`shakespearean_response` — the hero compresses 44 → 38 tokens, but the more interesting move is on accuracy: 0.80 → **0.88**. The trained-agent prompt (which inlines the marker list `(thou, thy, hath, art, doth, ere)` rather than describing it in prose) gets Llama to produce a richer Shakespearean response. The judge rewards persona richness; structurally inlining the markers prompts deeper register-matching.*

### Other variants we ran

- **Qwen → Qwen same-family control** ([`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b)) — looks great on win-rate (70/87 vs verbose) but the verbose-Qwen ceiling is only 0.15. Cross-family Llama is the much harder bar.
- **Thinking-ON A/B** ([`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama)) — `<think>...</think>` chat template enabled. Loses to OFF on reward (+0.379 vs +0.426) at +30% tokens. We ship OFF as the default.

All adapters are public with eval JSONLs and demo CSVs:

- 🥇 [`prompt-golf-qwen-to-llama-nothink`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink) (hero) · [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv)
- 🔁 [`prompt-golf-multistep-llama`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama) (3-turn) · [base eval](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_base.jsonl) · [trained eval](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_trained.jsonl)
- 🅰️ [`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama) (thinking=ON variant)
- 🎛️ [`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b) (Qwen→Qwen control)

---

## What the agent learned

Inspecting trained-agent outputs, a few patterns jump out:

**Format cues are tokens, not sentences.** `JSON:` does the job that *"Output your response as a JSON object with the following structure"* does — at 50× fewer tokens. The trained agent finds this; humans almost never do, because we feel rude not explaining.

**Persona triggers are tiny.** `Yarrr,` for pirate. `psql>` for SQL. `Once upon a time,` for fairy-tale. These tokens carry enormous behavioral payload because they're strong prefix-match anchors in the target's training distribution. The trained agent treats them like keys; humans write paragraphs that try to describe what a single token can simply *evoke*.

**Add the label vocabulary the human forgot.** On classification tasks where the verbose prompt described the task but didn't list the labels, the agent learned to *insert the label set* — even though that increases length. The reward signal rewarded explicit vocabulary because Llama needed it. The human prompt was too polite to spell out the obvious; the agent has no such instinct.

**Show, don't tell.** For tasks like "respond in exactly 4 numbered steps," the agent learned to *demonstrate* the structure (`1.\n2.\n3.\n4.\nAnswer:`) rather than describe it.

**Drop ceremonial preamble.** *"In this task you will…"* / *"Please carefully consider…"* / *"Your goal is to…"* — gone, every time, with no measurable accuracy cost. The first chunk of most human-written prompts is almost pure decoration.

We also saw failure modes worth flagging:

- **Mild gibberish convergence on a few adversarial tasks.** Refusal-related tasks pushed the agent toward GCG-style ungrammatical prompts. The leakage penalty caught the worst cases.
- **Output-style regression even when accuracy holds.** Sometimes the agent matches verbose accuracy but loses output cleanliness — same correct answer, but wrapped in markdown fences with a preamble. Same task accuracy, different downstream parsing burden. The kind of thing a length-only reward will systematically miss unless you grade for it.
- **The 1-token attractor.** Without `MIN_TOKENS_FLOOR`, RL inevitably finds 1-2 token prompts exploiting tokenization artifacts. These aren't prompts; they're attacks. The floor penalty is non-optional.

![Live demo: JSON extraction — accuracy holds but output style regresses](./assets/demo_json_extraction.png)

*JSON extraction task. Both prompts get 1.00 accuracy on the labeled fields. But the verbose prompt's "Must be a single JSON object (curly braces), not a list" constraint produces a clean inline `{"name": "Sanyam", ...}`. The trained agent compressed that constraint away — Llama still produces the right keys, but wraps them in markdown code fences with a "Here's the formatted JSON object:" preamble. **Same accuracy, different downstream parsing burden.** This is the kind of subtle regression the per-row eval CSV is designed to surface — and the kind of thing a length-only reward will systematically miss unless you grade for it.*

### Notes on training (for the curious)

Three things mattered in practice if you want to reproduce the run cleanly:

- **Pre-flight capability profiling.** We ran each task with the verbose prompt first and recorded `description_baseline` per task. Tasks where verbose also fails produce zero gradient (no GRPO group variance) and dilute the budget. We dropped them from the active task pool.
- **`frac_reward_zero_std` is the diagnostic.** GRPO groups with zero intra-group reward variance contribute no gradient. The "tough" tier gave the most signal precisely because reward was widely dispersed within each group.
- **Format anchors emerge before content compression.** The agent first discovers high-payload trigger tokens (`JSON:`, `psql>`, `Yarrr,`) — content-level compression (dropping ceremonial preamble like *"In this task you will…"*) comes later, around step 200+.

---

## Why this matters

| If you work on… | Prompt Golf gives you… |
|---|---|
| **Inference cost in production** (ad tech, moderation, compliance) | A trained policy that compresses verbose prompts behaviorally — no gradient access to the target needed. Up to 30× compression on real-world policy prompts. |
| **Capability evaluation** | A black-box minimum-elicitation metric per task per target. Decouples *can the model do X* from *did we find the right prompt*. |
| **Prompt distillation across targets** | Cross-family training generates a model of the target's response surface. Swap targets, retrain, ship a custom compressor for your specific deployment. |
| **Capability elicitation research** | A black-box analog of password-locked-model elicitation ([Greenblatt et al., 2024](https://arxiv.org/abs/2405.19550)). What's the minimum input that surfaces a latent capability? |
| **Red-teaming / robustness** | Same machinery, different rubric. Adversarial scoring → red-teaming. Refusal rubric → jailbreak hardening. |
| **LLM ↔ LLM behavioral modeling** | Machine Theory of Mind for LLMs as targets. The agent's policy implicitly encodes a model of the target. |

### What this is — and isn't

- ✅ **Is** the first open OpenEnv RL environment where the agent learns to write prompts for another LLM.
- ✅ **Is** a calibrated middle: GCG/RLPrompt-style mechanics, Machine ToM-style framing, reusable infrastructure.
- ❌ **Isn't** a generative simulator of LLM behavior — we never touch activations.
- ❌ **Isn't** a new prompt-optimization algorithm. The algorithmic core is RL+length; the contribution is the framing + reusable env + cross-family experiments.
- ❌ **Isn't** a claim that we've "solved world modeling for LLMs." Episodes are short; the analogy to Dreamer/JEPA/Genie is structural, not algorithmic.

---

## Prior work — the lineage we sit on top of

This work isn't novel in any single dimension; it's a deliberate combination of four research lines that haven't been put in the same env before.

**Machine Theory of Mind.** The conceptual ancestor is Rabinowitz et al.'s [ToMnet (2018)](https://arxiv.org/abs/1802.07740) — train one network to predict another agent's behavior from observed interactions, with no access to its internals. Same shape: black-box modeling of one model by another, learned from outputs alone. We swap their gridworld for natural-language tasks and their predictor for a generative agent that *acts on* its model rather than just predicting from it.

**LLM-on-LLM red teaming.** Perez et al.'s [Red Teaming Language Models with Language Models (2022)](https://arxiv.org/abs/2202.03286) is the direct algorithmic ancestor: an LLM generates inputs to elicit specific behaviors (jailbreaks) from a frozen target LLM, with RL closing the loop on success. Prompt Golf is the same machinery pointed at a constructive rubric — *task success* and *length* instead of *adversarial reward*. Switch the rubric and the same env runs red-teaming.

**Capability elicitation.** Greenblatt et al.'s [Stress-Testing Capability Elicitation With Password-Locked Models (2024)](https://arxiv.org/abs/2405.19550) frames the question we care about: *given a target model, what's the minimum input that surfaces a latent capability?* They build password-locked models and measure how much access (fine-tuning, few-shot, prompts) is needed to unlock them. Prompt Golf operationalizes the prompt-only side of that question as a learnable RL objective with a length budget, on tasks where the capability is open rather than locked.

**Prompt optimization, classical.** The algorithmic toolkit is well-trodden:
- **AutoPrompt** ([Shin et al., 2020](https://arxiv.org/abs/2010.15980)) — gradient-search over discrete tokens to elicit knowledge.
- **GCG** ([Zou et al., 2023](https://arxiv.org/abs/2307.15043)) — coordinate-descent prompt optimization for jailbreaks; established that white-box gradient-based search produces compact behavioral attacks.
- **RLPrompt** ([Deng et al., 2022](https://arxiv.org/abs/2205.12548)) — RL-trained policy for soft/hard prompt search; the closest direct ancestor in algorithm shape.
- **PCRL** ([Choo et al., 2023](https://arxiv.org/abs/2308.08758)) — preference-conditioned RL for prompt optimization.

What we add to that toolkit is the *framing* (Machine ToM, cross-family, "minimum elicitation as capability metric") and the *infrastructure* (a reusable OpenEnv with 90 graded tasks, 21 scorers, a length-budget rubric, and a frozen-target wrapper that swaps in any HF model).

We're not the first to compress prompts with RL. We're trying to be the first place where you can *go to do this experiment* — fork the env, swap in your target, run it, get a number.

---

## Try it yourself

There's a [live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo) where you pick a task, see the verbose human prompt and the trained agent's compressed prompt side by side, and run either against the same Llama-3.2-3B target on real test inputs. Every screenshot in this post is from that demo — pick any task, edit the input, hit "Run target with all three prompts," and see for yourself.

The task selector annotates each task with its compression ratio and reward delta — `[3× compress, Δr=+0.00]` for clean wins, `[1× compress, Δr=+0.15]` for accuracy improvements — so you can scan to whichever stories interest you most.

### Run the env locally

```bash
git clone https://huggingface.co/spaces/rishabh16196/prompt_golf_env
cd prompt_golf_env
pip install -e . gradio transformers torch

# CPU smoke test
PROMPT_GOLF_TARGET_BACKEND=mock python -m server.tasks_tough

# Real run with the actual Llama target
PROMPT_GOLF_TARGET_BACKEND=hf \
PROMPT_GOLF_TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
python ui/demo_app.py  # opens http://localhost:7860
```

### Use the trained adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
agent = PeftModel.from_pretrained(
    base, "rishabh16196/prompt-golf-qwen-to-llama-nothink"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
```

### Reproduce the hero training run

```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh
# ~3h on L40S, pushes adapter + plots + train_metrics + eval JSONLs
```

---

## What's next

Things we'd love community help on:

1. **More targets.** We have Qwen3-1.7B and Llama-3.2-3B profiled. What about Phi-3, Mistral, Gemma 2? Is the trained agent's policy portable across targets, or is it Llama-specific? This is the cross-target transfer experiment that would substantiate the Machine ToM framing.
2. **Larger task banks.** 90 hand-crafted tasks is a starting point. Procedural task generation (random format constraints, synthetic policies) would scale this to thousands.
3. **Different reward shapes.** Additive `raw - 0.5·baseline - 0.002·tokens - leak²` is one choice. KL-as-reward (output-distribution matching the verbose prompt's) is another.
4. **Real-world deployment study.** Pick an actual production prompt (with permission), train a compressor for it, measure compression-vs-accuracy in shadow traffic. We'd love to hear what breaks and what holds up.

If you have a 1000-token policy that's eating your inference budget, train a compressor for it. If you've ever wondered whether one LLM can really learn to whisper to another — train the env, watch the curves, look at what the agent wrote.

That's the whole point. Both points, actually.

---

## Acknowledgments & lineage

Built for the [OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/) (Meta + Hugging Face + PyTorch, India 2026), using TRL GRPO and HuggingFace Jobs. The full lineage is in the [Prior Work](#prior-work--the-lineage-we-sit-on-top-of) section above.

```bibtex
@misc{promptgolf2026,
  author = {Rishabh},
  title  = {Prompt Golf: An OpenEnv RL environment for cross-model prompt compression},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/spaces/rishabh16196/prompt_golf_env}}
}
```

Connect: [HuggingFace](https://huggingface.co/rishabh16196) · [GitHub mirror](https://github.com/rishabh16196/prompt_golf_env)

⛳