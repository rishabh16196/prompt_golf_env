---
title: 'Prompt Golf: can one LLM learn to whisper to another?'
thumbnail: /blog/assets/prompt_golf/thumbnail.png
authors:
  - user: rishabh16196
---

# Prompt Golf

> *Same accuracy as the human-written prompt at ~55% of the tokens — learned by an RL agent that never saw the target's weights, only its outputs.*

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

After 500 GRPO steps on a 90-task bank, the agent compresses verbose human-written prompts (mean ~63 tokens, up to 737 on long-context policy tasks) into **35-token prompts** that retain **80% of the verbose accuracy** and **beat the human prompt outright on 48 of 87 tasks (55%)**. Peak compression: **30× on long-context policy tasks**.

Everything is open: [the env](https://huggingface.co/spaces/rishabh16196/prompt_golf_env), the [trained adapter](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink), the [training pipeline](https://github.com/rishabh16196/prompt_golf_env/tree/main/training), and a [live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo) where you can play prompts against the same target the agent was trained on. The [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv) has all 90 tasks × verbose / untrained / trained / accuracy side by side.

| | |
|---|---|
| **The capability we're testing** | Can one LLM learn to write the minimum prompt that elicits a specific behavior from a frozen target LLM? |
| **The environment** | Single-step RL. Agent writes a prompt → frozen target runs it on 6 hidden test inputs → reward = task_success − 0.5·baseline − 0.002·tokens − leakage². |
| **The recipe** | Qwen3-1.7B (LoRA, r=16) ⟶ Llama-3.2-3B-Instruct (frozen). 500 GRPO steps on a 90-task bank. ~3h on a single L40S. |
| **The result** | 35-token prompts → 80% of verbose accuracy. Wins on 55% of tasks. |
| **Why care** | First OpenEnv environment for cross-model prompt-writing as a learnable skill. Plugs straight into red-teaming, prompt distillation, capability elicitation. |

---

## How it works

One episode = one task = one prompt. Single-turn, conceptually simple:

1. The env hands the agent a **task description** (verbose, hand-written), 3 visible **train examples**, and a **token budget**.
2. The agent's action is a **prompt string** (typically wrapped in `<prompt>...</prompt>`).
3. The env prepends that prompt to ~6 *hidden* test inputs, runs the **frozen target LLM** on each, scores the outputs.
4. Reward = `raw_task_score − 0.5·baseline − 0.002·tokens − leakage_overlap²`, clipped.

The held-out test inputs are **never shown to the agent**. An n-gram leakage detector zeros the reward if the agent tries to paste held-out content into its prompt. Multi-turn mode (`turn_limit > 1`) splits the test pool into a small *feedback* slice (revealed across turns with target outputs) and a held-out *scoring* slice (only the final-turn prompt is judged).

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

A few practical things that mattered:

**Pre-flight capability profiling is non-negotiable.** Before committing GPU hours, we ran each task with the verbose hand-written description and recorded `description_baseline` per task. Tasks where the verbose prompt also fails produce zero gradient (no GRPO group variance) and dilute the budget. Profile first, train second. We dropped tasks the target couldn't solve with *any* prompt — there's nothing for golf to compress *toward* if there's no peak.

**`frac_reward_zero_std` is the diagnostic to watch.** GRPO groups with zero intra-group variance contribute no gradient. The "tough" tier gave the most signal precisely because reward was widely dispersed within each group.

**Format anchors emerge before content compression.** Watching intermediate checkpoints, the agent first discovers that certain trigger tokens — `JSON:`, `psql>`, `Yarrr,` — carry enormous behavioral payload. Content-level compression (dropping ceremonial preamble like *"In this task you will…"*) comes later, around step 200+. The order of capabilities is itself interesting.

### The thinking-mode A/B that didn't work

Qwen3 has an optional `<think>...</think>` chat template — free reasoning scratch space before the final output. Hypothesis: free reasoning would let the agent strategize about format anchors before emitting the prompt, since the rubric only counts the *extracted* prompt's tokens. Should be free intelligence.

It wasn't. Identical training setup, thinking ON vs OFF:

|  | thinking=OFF (hero) | thinking=ON |
|---|---|---|
| Trained accuracy | 0.523 | **0.539** |
| Trained reward | **+0.426** | +0.379 |
| Mean tokens | **35** | 46 |

OFF wins on reward and compression by a clear margin. ON wins on accuracy by 1.6 percentage points at 30% more tokens. **The credit assignment between `<think>` tokens and the final extracted prompt is too weak for GRPO to exploit at this scale** — the gradient just doesn't flow cleanly across the thinking block. We ship OFF as the hero.

---

## Results

### Headline numbers (90-task average)

| Stage | Mean accuracy | Mean tokens |
|---|---|---|
| Verbose human-written prompt | **0.65** | ~63 |
| Untrained Qwen3-1.7B agent | 0.48 | 38 |
| **Trained Qwen3-1.7B + LoRA** | **0.52** | **35** |

→ **80% accuracy retention at 55% of the verbose token count**, scored on a frozen Llama-3.2-3B target the agent never had gradient access to.

The trained agent **beats the human verbose prompt on 48 of 87 evaluated tasks (55%)** under the same rubric. On the 39 it loses, the failure mode is consistent: it compressed too aggressively to keep up with Llama's verbose-prompt capability ceiling. **On those tasks the verbose prompt's extra tokens were doing real cognitive work**, not just adding decoration. We're honest about this — the demo CSV shows every row.

### The training curves

![Reward curve over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/reward_curve.png)

*Mean reward per step, climbing from ~0 to +0.43 over 500 steps. The plateau around step 350 is where length compression saturates against accuracy preservation.*

![Mean prompt length over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/length_curve.png)

*Mean prompt tokens per step. The agent finds the compression frontier within the first ~150 steps and then refines it. The trajectory is monotonic but uneven — bigger compression jumps happen on tasks where the agent discovers a new format anchor.*

![Reward component breakdown](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/breakdown.png)

*Decomposition of the additive reward. The length penalty stays small because the agent quickly stops paying it; the gain comes from raw task score climbing while baseline-subtracted reward stays positive.*

For step-by-step exploration, the **[Trackio dashboard](https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio)** has the full per-step metrics replayed from `train_metrics.jsonl`.

### Where it shines

| Task | Verbose | Trained | Win |
| --- | --- | --- | --- |
| `sentiment_basic` | 27 tok / **0.83** | **18 tok** / **1.00** | shorter AND more accurate |
| `tough_yaml_nested_depth` | 74 tok / 0.96 | **20 tok** / **1.00** | 3.7× compression, accuracy improved |
| `json_key_ordering` | 47 tok / 0.61 | **38 tok** / **0.78** | shorter AND +17pp accuracy |
| `tough_fallacy_classify` | 164 tok / 0.00 | **59 tok** / **0.33** | added the label vocabulary the verbose prompt forgot |
| `policy_msn_ad_creative` | **737 tok** / 0.00 | **20 tok** / 0.00 | 37× compression — both fail because Llama-3.2-3B can't reason over the policy hierarchy, but the compression is *free* |

That last row is the ad-tech case in miniature. Llama-3.2-3B can't actually solve `policy_msn_ad_creative` — it's not a strong enough target. So both prompts get 0 accuracy. **But the verbose prompt was charging 737 tokens of prefill on every request to deliver that 0.** The trained agent does it for 20. **Pair the compressed prompt with a stronger target and you'd ship the same behavior at 37× lower input-token cost.** That's exactly the production payoff that motivated this in the first place.

### What the agent actually wrote

Sentiment classification:

> *Verbose:* "For each input review, output exactly one of: positive, negative, neutral. Output the label only — no punctuation, no explanation." (27 tokens)
>
> *Trained:* "Classify the input review as positive, negative, or neutral. Output only the label." (18 tokens, **1.00 accuracy**)

YAML extraction with strict nesting:

> *Verbose:* 74 tokens describing depth requirements, entity coverage, format constraints, output instructions.
>
> *Trained:* "Generate a YAML document that meets the specified minimum nesting depth and includes all entities from the given specification." (20 tokens, **1.00 accuracy**)

Policy compliance — the long-context money case:

> *Verbose:* 737 tokens of MSN ad-creative policy.
>
> *Trained:* "Classify the input creative as allow, disallow, or review based on the given policy guidelines." (20 tokens)

### When the agent doesn't just shrink — it improves

![Live demo: shakespearean_response, 1× compress, +0.15 reward gain](./assets/demo_shakespearean.png)

*`shakespearean_response` — the trained agent compresses 44 → 38 tokens, but the more interesting move is on accuracy: 0.80 → **0.88**. Inspecting the target outputs shows why. The trained-agent prompt (which inlines the marker list `(thou, thy, hath, art, doth, ere)` rather than describing it in prose) gets Llama to produce a richer Shakespearean response — "Fair patron, thou dost inquire about thy canine companion, dost thou? Ere I respond, I pray thee, tell me more of thy hound's plight..." — vs. the more conservative verbose output. The judge rewards persona richness; structurally inlining the markers prompts deeper register-matching.*

### The same-family control: Qwen → Qwen

|  | Qwen→Qwen (control) | Qwen→Llama (hero) |
|---|---|---|
| Trained beats verbose on | **70/87 tasks (80%)** | 48/87 (55%) |
| Mean reward advantage vs verbose | **+0.085** | -0.057 |
| Verbose accuracy ceiling | 0.15 | 0.65 |

This looks like a "win" for Qwen→Qwen on win-rate, but the framing is misleading. Qwen3-1.7B as a target only achieves 0.15 average accuracy with verbose prompts — the bar to beat is on the floor. **Cross-family Llama is a much harder bar to clear (0.65 verbose ceiling), but the absolute accuracy delivered by the trained agent is dramatically higher.** This is the kind of nuance you only see when you actually run the comparison.

![Qwen→Qwen reward curve](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/resolve/main/plots/reward_curve.png)

*Same training recipe with Qwen3-1.7B as both agent and target. The curve looks great because the verbose baseline is weak — but absolute accuracy is much lower than the cross-family run.*

### The thinking-ON variant

![Qwen→Llama thinking-ON reward curve](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/resolve/main/plots/reward_curve.png)

*Identical training setup with Qwen3's `<think>...</think>` chat template enabled. Trajectory is similar in shape but absolute reward plateaus lower — the extra tokens spent inside `<think>` cost more than the accuracy gain they buy.*

All trained adapters are public, with their own demo CSVs:

- 🥇 **[`prompt-golf-qwen-to-llama-nothink`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink)** (thinking=OFF, hero) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv)
- 🅰️ **[`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama)** (thinking=ON variant) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/blob/main/evals/qwen_to_llama_demo.csv)
- 🎛️ **[`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b)** (Qwen→Qwen control) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/blob/main/evals/qwen_to_qwen_demo.csv)
- 🔁 **[`prompt-golf-multistep-llama`](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama)** (multi-turn, 3 turns/episode) — [base eval](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_base.jsonl) · [trained eval](https://huggingface.co/rishabh16196/prompt-golf-multistep-llama/blob/main/evals/eval_trained.jsonl)

### The multi-turn variant: same agent, 3 turns to get it right

The hero is single-step — the agent writes one prompt and the episode ends. Multi-turn loosens that. We ran a 3-turn variant where the agent sees its prior prompts plus per-example feedback on a *separate* feedback slice (2 examples), then the **final-turn** prompt is judged on a held-out 4-example scoring slice. Same task bank. Same target. Trained with a hand-rolled trajectory-level GRPO (`train_grpo_multistep.py`) for 150 steps × 8 trajectories × 3 turns.

| | Single-step hero | Multi-step (3 turns) |
|---|---|---|
| Trained accuracy | 0.523 | **0.576** |
| Trained reward | +0.426 | **+0.440** |
| Mean tokens | **35** | 43.7 |
| Trained beats untrained on | — | 29 / 90 tasks |

Six points of accuracy and a slight reward gain, paid for with ~25% more tokens. But the average hides the interesting structure — **multi-step is a conditional improvement, not a strict upgrade**:

| Category | Multi-step wins | Multi-step losses |
|---|---|---|
| `reasoning_tough` | **5 / 10** | 0 |
| `classification_tough` | **7 / 10** | 2 |
| `extraction_tough` | 4 / 10 | 4 (mixed) |
| `policy_compression` | 1 / 3 (`policy_content_moderation` 0.00 → **0.67**) | 1 (`policy_finreg_communication_review` 17 → 112 tokens, still 0.00) |
| `format` | 2 / 8 | **5 / 8** |

Top single-task wins:
- `tough_policy_stance`: 0.00 → **1.00** (+0.98 reward)
- `tough_syllogism_check`: 0.00 → **1.00** (+0.97)
- `sentiment_basic`: 0.00 → **1.00** (+0.93)
- `json_key_ordering`: 0.11 → **1.00** (+0.89)
- `tough_contract_obligation`: 0.06 → **0.89** (+0.69; 43 → 117 tokens)
- `policy_content_moderation`: 0.00 → **0.67** — multi-step *unlocked* a previously dead policy task by using turn 2/3 to refine the categorization prompt

Where it hurts: format-strict tasks where short single-shot prompts already win. `format_uppercase` goes 9 → 24 tokens with no accuracy gain. `format_three_bullets` regresses 1.00 → 0.92. The agent uses its turn-2 / turn-3 budget to bloat prompts that didn't need bloating.

**The intuition:** multi-turn relaxes length pressure because the agent has room to debug across turns. That helps tasks where the agent needs reasoning room (tough reasoning, tough classification, complex extraction, policy-style compression). It hurts tasks where the optimal answer is "just say the right 8 words." We predicted this exact split in the v3 design plan before the run completed — it's a clean confirmation that multi-turn isn't a free upgrade, it's a different operating point on the accuracy/length curve.

**Operationally:** single-step is still the better default for inference-cost-sensitive deployments. Multi-step is the right pick when accuracy ceilings matter more than token count — and especially when individual tasks reward step-by-step refinement.

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

This work draws on four converging research lines:

- **[Machine Theory of Mind](https://arxiv.org/abs/1802.07740)** (Rabinowitz et al., 2018) — the conceptual ancestor.
- **[Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)** (Perez et al., 2022) — the direct algorithmic ancestor.
- **[Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550)** (Greenblatt et al., 2024) — the motivation for treating minimum elicitation as a meaningful capability metric.
- **[AutoPrompt](https://arxiv.org/abs/2010.15980)**, **[GCG](https://arxiv.org/abs/2307.15043)**, **[RLPrompt](https://arxiv.org/abs/2205.12548)**, **[PCRL](https://arxiv.org/abs/2308.08758)** — the algorithmic toolkit.

Built for the [OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/) (Meta + Hugging Face + PyTorch, India 2026), using TRL GRPO and HuggingFace Jobs.

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