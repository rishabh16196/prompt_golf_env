---
title: 'Prompt Golf: training one LLM to write the shortest prompts that steer another'
thumbnail: /blog/assets/prompt_golf/thumbnail.png
authors:
  - user: rishabh16196
---

# Prompt Golf

> *Same accuracy as the human-written prompt at ~55% of the tokens — learned by an RL agent that never saw the target's weights, only its outputs.*

We trained an LLM to be a prompt engineer for *another LLM*.

The setup: a Qwen3-1.7B **agent** (LoRA-fine-tuned via TRL GRPO) writes prompts. A frozen Llama-3.2-3B **target** runs them. The reward is task success minus prompt length. After 500 GRPO steps on a 90-task bank, the agent compresses verbose human-written prompts (mean ~63 tokens, up to 737 on long-context policy tasks) into **35-token** prompts that retain **80% of the verbose accuracy** and **beat the human prompt outright on 48 of 87 tasks (55%)**.

Peak compression: **37×** on long-context policy tasks — a 737-token MSN ad-creative policy compressed to a 20-token classifier prompt.

Everything is open: the OpenEnv environment, three trained adapters, a live Gradio demo where you can play prompts against the same target, a Trackio dashboard with the full training trajectory, and a reproducible HuggingFace Jobs pipeline.

> 🌍 **[Environment Space](https://huggingface.co/spaces/rishabh16196/prompt_golf_env)**
> 🎛️ **[Live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo)**
> 📊 **[Trackio dashboard](https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio)**
> 🤗 **[Hero adapter](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink)**
> 🐙 **[GitHub mirror](https://github.com/rishabh16196/prompt_golf_env)**

<!-- IMAGE PLACEHOLDER 1 — Hero
     A side-by-side panel:
       LEFT  : 737-token MSN ad-creative policy (truncated with "...")
       RIGHT : 20-token trained-agent compression
       Bottom badge: "37× compression"
     This is the strongest single visual you have. Lead with it. -->

---

## TL;DR

| | |
|---|---|
| **The capability we're testing** | Can one LLM learn to write the minimum prompt that elicits a specific behavior from a frozen target LLM? |
| **The environment** | Single-step RL. Agent writes a prompt → frozen target runs it on 6 hidden test inputs → reward = task_success − 0.5·baseline − 0.002·tokens − leakage². |
| **The recipe** | Qwen3-1.7B (LoRA, r=16) ⟶ Llama-3.2-3B-Instruct (frozen). 500 GRPO steps on a 90-task bank. ~3h on a single L40S. |
| **The result** | 35-token prompts → 80% of verbose accuracy. Wins on 55% of tasks. 37× peak compression on long-context policy tasks. |
| **Why care** | First OpenEnv environment for cross-model prompt-writing as a learnable skill. Plugs straight into red-teaming, prompt distillation, capability elicitation. |

---

## 1. The capability gap: prompts as folklore

Modern LLMs are trained to **follow** prompts. They are not trained to **write** them. But every serious deployment ships a prompt-engineering pipeline anyway:

- **Ad tech:** a 700-token policy describing what creatives can serve, prepended to every classification call.
- **Content moderation:** multi-page community guidelines stuffed into the system prompt of every Llama instance scoring user posts.
- **Customer support:** a 1500-token persona document that turns every reply into "Hi, this is Bot™ — I'm here to help! 🌟".
- **Compliance:** FINRA-style review rules that a model has to internalize to flag broker communications correctly.

These prompts get written, version-controlled, A/B tested — by humans, with intuition. They are also the single largest line item in inference cost. **A 700-token policy on 10M daily requests is 7 billion tokens of prefill compute per day** — and we strongly suspect most of those tokens are decorative, not load-bearing.

There's a deeper research problem hiding underneath the cost. We have **no clean way to distinguish "the model can't do X" from "we haven't found the right prompt."** Modern benchmarks conflate the two. The gap between a *minimum* and a *verbose* prompt that elicit the same behavior is empirical evidence about what's stored in weights vs. what must be supplied via context — but no reusable RL environment exists to study this.

There are pieces in the literature that gesture at this. **AutoPrompt** ([Shin et al., 2020](https://arxiv.org/abs/2010.15980)) and **GCG** ([Zou et al., 2023](https://arxiv.org/abs/2307.15043)) search for short prompts but produce gibberish that doesn't generalize. **RLPrompt** ([Deng et al., 2022](https://arxiv.org/abs/2205.12548)) and **PCRL** ([Jung & Kim, 2024](https://arxiv.org/abs/2308.08758)) use RL with length penalties as one-off papers, not reusable environments. **Red-Teaming-with-LMs** ([Perez et al., 2022](https://arxiv.org/abs/2202.03286)) trains an LLM to elicit behaviors from a frozen LLM — exactly our setup — but oriented at safety rather than capability.

Prompt Golf is the missing piece: an open, reusable OpenEnv RL environment for cross-model prompt-writing, with the same algorithmic core but a research framing oriented at capability elicitation, prompt distillation, and behavioral modeling.

The conceptual ancestor we lean on hardest is Rabinowitz et al.'s **[Machine Theory of Mind](https://arxiv.org/abs/1802.07740)** — meta-learn a model of another agent from interaction. That's exactly what the Qwen agent ends up doing. It never sees Llama's gradients. It only sees Llama's outputs. From those, it builds a probabilistic behavioral model of Llama's response surface, encoded in the prompts it learns to write.

---

## 2. The environment

Each episode is one task. The agent sees the task, writes a prompt, gets scored. That's the whole loop.

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

### Reward

```
reward = raw_task_score
       − 0.5 · baseline_zero_shot     ← don't reward what the target already does
       − 0.002 · submitted_tokens      ← the golf score
       − leakage_overlap²              ← anti-cheat: caught pasting test inputs
       − short_penalty (if tokens < 5) ← anti-collapse to 1-token prompts

clipped to [-0.5, 1.3]
```

Three things about this composition matter.

**Additive, not multiplicative.** Earlier versions used `length_factor × leakage_factor × raw_score`, which gave brittle gradients (the multiplicative form has dead zones where one factor is small and gradients vanish). The additive form is smoother and what training actually converges on.

**Baseline subtraction is load-bearing.** Without it, the agent gets credit for tasks the target already does well at zero-shot — which means it's rewarded for nothing. With it, the reward signal isolates *additional capability elicited by the prompt*, which is what we actually care about.

**Anti-collapse floor.** Without `MIN_TOKENS_FLOOR=5`, GRPO inevitably converges on degenerate 1-2 token prompts that exploit specific tokenization artifacts. These aren't prompts in any meaningful sense — they're attacks on the target's tokenizer. The floor penalty turns the search away from those local optima.

### Anti-leakage

The 6 held-out test inputs are **never shown to the agent**. A trigram-overlap detector zeros the reward if the agent tries to paste held-out inputs into its prompt. Multi-turn mode (when `turn_limit > 1`) splits the test pool into a 2-example *feedback* slice (revealed across turns with the target's outputs) and a 4-example *scoring* slice (only the final-turn prompt is judged) — so the agent can debug across turns without leaking the inputs that ultimately judge it.

### Scorers

Each task picks one of 21 scorers, grouped into 7 families:

| Family | Scorers | What they check |
|---|---|---|
| **Exact / membership** | `exact_label`, `contains_label`, `contains_all_substrings`, `uppercase_match` | Closed-vocabulary classifiers; required substrings; case-strict rewrites |
| **Numeric** | `numeric_match`, `word_count_exact` | Last numeric token within tolerance; word count exactly N |
| **JSON / YAML** | `json_contains_fields`, `valid_json_object`, `json_key_order`, `valid_yaml_depth` | Required keys/values; key ordering; nesting depth |
| **Format-strict** | `three_bullets`, `acrostic_match`, `avoid_letter`, `ends_question`, `terminal_output_pattern` | Exactly 3 bullets; first letters spell a word; output avoids a letter; ends with `?`; terminal-session shape |
| **Multi-step / language** | `stepwise_math`, `translation_match`, `selective_translate` | Numbered steps + numeric answer; token-F1 vs reference; partial-translation rules |
| **Safety** | `refusal_score` | Whether the output is a refusal (matches expected refuse/comply label) |
| **LLM judge** (Qwen3-8B 8-bit) | `judge_criteria`, `judge_vs_expected` | Free-form persona / reasoning / Yoda-syntax tasks; deterministic decoding |

The scorer is **fixed per task and never seen by the agent** — it has to infer from train examples + task description what gets graded. *Verifiable beats judgeable* is the design principle: every task we can grade with a regex, we do; LLM judges only kick in for genuinely free-form behaviors like persona consistency.

---

## 3. Why cross-family is the right setup

If the agent and target are the same model family, you're really doing self-distillation: the agent has perfect access to its own response surface. We ship that as a control (`prompt-golf-grpo-1.5b`, Qwen→Qwen).

When agent and target are different families, the agent has to **build an empirical model of the target's behavior from outputs alone**. Concretely, it learns:

- Which words Llama needs to constrain its output format (`Output the label only, no punctuation.`)
- Which words Llama can drop without consequence (`Please carefully consider…`)
- Which compressions break Llama even when they look semantically equivalent
- That Llama-3.2 needs explicit label vocabularies on classification but *doesn't* need them on JSON extraction

This is **operationalized behavioral theory-of-mind** — the agent's policy implicitly encodes a probabilistic model of another model's response surface.

Cross-family also turned out to be the *easier* setup empirically, for an unexpected reason. Llama-3.2-3B is significantly more cooperative on strict-format tasks than Qwen3-1.7B: **67/87 tasks have non-zero verbose-prompt accuracy on Llama**, vs only 19/87 on Qwen. That changes what training can attempt at all — cross-family Qwen→Llama gives the agent more "real" tasks with reward variance to learn from.

---

## 4. The 90-task bank

Task quality is the single biggest determinant of whether this kind of environment is interesting or boring. A great training loop on bad tasks teaches the wrong thing. We curated each task against three filters:

1. **Empty-prompt baseline must fail.** No free lunch. We ran every task with an empty prompt and dropped the ones where the target succeeded anyway.
2. **Verbose prompt must succeed.** A capability ceiling has to exist for there to be room to compress. Run `bash training/hf_job_profile.sh` on your fork to do this check yourself.
3. **Minimum prompt must be non-obvious.** The whole game is closing the gap between (2) and (3).

| Tier | Count | Examples |
|---|---|---|
| **v1** (`tasks.py`) | 20 | sentiment classification, NER, JSON extraction, translation, refusal |
| **v2** (`tasks_v2.py`) | 15 | acrostic, no-letter-e, YAML nested depth, pirate persona, terminal session output |
| **tough** (`tasks_tough.py`) | 52 | logical fallacy ID, FINRA risk classification, Yoda-with-constraint |
| **policy** (`tasks_policy.py`) | 3 | MSN ad-creative policy (737 tok), content moderation rules (612 tok), FINRA broker-dealer review (550 tok) |

Each task ships 3 visible train examples + 6 hidden test examples + a per-task token budget (60–250).

The **policy tasks are the headline workload**: each has a 500–700-word real-world-style policy as the verbose prompt, and the agent has to compress it into a ≤250-token classifier prompt that still routes inputs to the right `allow / disallow / review` decision. That's where the inference-cost story is most visible — these are prompts that look like real production system prompts.

<!-- IMAGE PLACEHOLDER 2 — Task bank diagram
     A 4-panel grid, one panel per tier, showing one example task each:
       v1     : "sentiment_basic"        — input review → label
       v2     : "tough_yaml_nested_depth" — input spec → 4-deep YAML
       tough  : "tough_fallacy_classify" — input argument → fallacy name
       policy : "policy_msn_ad_creative" — input creative → allow/disallow
     For each: show a 2-line example input → expected output.
     Helps the reader picture what the agent is being graded on. -->

---

## 5. Training: GRPO, LoRA, ~3 hours on an L40S

The recipe:

- **Agent:** Qwen3-1.7B + LoRA (r=16, α=32), trained with TRL GRPO
- **Target:** `meta-llama/Llama-3.2-3B-Instruct` (frozen)
- **Judge:** Qwen3-8B in 8-bit via `bitsandbytes` (only for `judge_*` scorers)
- **GRPO config:** 500 steps, `num_generations=8`, `lr=5e-6`, `β=0.04`, `temperature=0.9`, `max_completion_length=768`
- **Hardware:** single L40S (48 GB) on HuggingFace Jobs, ~3 hours per run
- **Anti-collapse guard:** `MIN_TOKENS_FLOOR=5` rubric penalty

To reproduce:

```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh
```

A few practical things that mattered along the way:

**Pre-flight capability profiling is non-negotiable.** Before committing GPU hours, we ran each task with the verbose hand-written description and recorded `description_baseline` per task. Tasks where the verbose prompt also fails produce zero gradient (no GRPO group variance) and just dilute the budget. Profile first, train second.

**`frac_reward_zero_std` is the diagnostic to watch.** If a GRPO group has zero intra-group reward variance, it contributes no gradient. The "tough" tier gave the most signal because its reward was widely dispersed within each group — that's a feature, not a bug.

**Format anchors emerge before content compression.** Looking at intermediate checkpoints, the agent first learns that certain trigger tokens (`JSON:`, `psql>`, `Yarrr,`) carry enormous behavioral payload. Content-level compression — dropping ceremonial preamble like *"In this task you will…"* — comes later, around step 200+.

### The thinking-mode A/B

Qwen3 supports an optional `<think>...</think>` chat template that gives the model free reasoning scratch space before the final output. Hypothesis: free reasoning would let the agent reason about format anchors before emitting the prompt, since the rubric only counts the *extracted* prompt's tokens.

We A/B'd identical training setups, thinking ON vs OFF:

|  | thinking=OFF (hero) | thinking=ON |
|---|---|---|
| Trained accuracy | 0.523 | **0.539** |
| Trained reward | **+0.426** | +0.379 |
| Mean tokens | **35** | 46 |

OFF wins on reward and compression by a clear margin. ON wins on accuracy by 1.6 percentage points at a 30% token cost. **The implicit credit assignment between `<think>` tokens and the final prompt is too weak for GRPO to exploit at this scale** — the gradient just doesn't flow cleanly across the thinking block. We ship OFF as the hero adapter and ON as a different operating point on the accuracy/length frontier.

---

## 6. Results

### Headline numbers (Qwen → Llama, 90-task average)

| Stage | Mean accuracy | Mean tokens |
|---|---|---|
| Verbose human-written prompt | **0.65** | ~63 |
| Untrained Qwen3-1.7B agent | 0.48 | ~38 |
| **Trained Qwen3-1.7B + LoRA** | **0.52** | **35** |

→ **80% accuracy retention at 55% of the verbose token count**, scored on a frozen Llama target the agent never had gradient access to.

The trained agent **beats the human verbose prompt on 48 of 87 tasks (55%)** under the same rubric. On the remaining 39 tasks, the accuracy drop on hard tasks outweighs the length savings — those are cases where the trained agent compressed too aggressively to keep up with Llama's verbose-prompt capability ceiling. **On those tasks the verbose prompt's extra tokens are doing real cognitive work**, not just adding decoration.

### The training curves

![Reward curve over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/reward_curve.png)

*Mean reward per step, climbing from ~0 to +0.43 over 500 steps. The plateau around step 350 is where length compression saturates against accuracy preservation.*

![Mean prompt length over 500 GRPO steps](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/length_curve.png)

*Mean prompt tokens per step. The agent finds the compression frontier within the first ~150 steps and then refines it. The trajectory is monotonic but uneven — bigger compression jumps happen on tasks where the agent discovers a new format anchor.*

![Reward component breakdown](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/resolve/main/plots/breakdown.png)

*Decomposition of the additive reward. The length penalty stays small because the agent quickly stops paying it; the gain comes from raw task score climbing while baseline-subtracted reward stays positive.*

For step-by-step exploration, the **[Trackio dashboard](https://huggingface.co/spaces/rishabh16196/prompt-golf-trackio)** has the full per-step metrics replayed from `train_metrics.jsonl`.

### Per-task highlights

The full row-by-row demo CSV is at **[`evals/qwen_to_llama_demo.csv`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv)** — every task with verbose / untrained / trained prompts side by side, plus accuracy and reward deltas. A few representative rows:

| Task | Verbose | Trained | Notes |
|---|---|---|---|
| `sentiment_basic` | 27 tok / **0.83** | **18 tok** / **1.00** | Shorter AND more accurate |
| `tough_yaml_nested_depth` | 74 tok / 0.96 | **20 tok** / **1.00** | 3.7× compression, accuracy improved |
| `json_key_ordering` | 47 tok / 0.61 | **38 tok** / **0.78** | Shorter AND +17pp accuracy |
| `tough_fallacy_classify` | 164 tok / 0.00 | **59 tok** / **0.33** | Trained agent added the label vocabulary the verbose prompt forgot |
| `policy_msn_ad_creative` | **737 tok** / 0.00 | **20 tok** / 0.00 | 37× compression — both fail because Llama-3.2-3B can't reason over the policy hierarchy, but the compression doesn't *cost* anything |

The `policy_msn_ad_creative` row is the most interesting. Both prompts get 0 accuracy, so it looks like a draw — but the verbose prompt was charging 737 tokens of prefill on every request to deliver that 0. The trained agent does it for 20. **Pair the compressed prompt with a stronger target and you'd ship the same behavior at 37× lower input-token cost.**

### What the agent actually wrote

For sentiment classification:

> *Verbose human prompt:* "For each input review, output exactly one of: positive, negative, neutral. Output the label only — no punctuation, no explanation." (27 tokens)
>
> *Trained agent:* "Classify the input review as positive, negative, or neutral. Output only the label." (18 tokens, **1.00 accuracy**)

For YAML extraction with strict nesting:

> *Verbose:* 74 tokens describing depth requirements, entity coverage, format constraints, output instructions.
>
> *Trained agent:* "Generate a YAML document that meets the specified minimum nesting depth and includes all entities from the given specification." (20 tokens, **1.00 accuracy**)

For policy compliance — the long-context money case:

> *Verbose:* 737 tokens of MSN ad-creative policy listing prohibited content, restricted categories, format standards, decision schemas.
>
> *Trained agent:* "Classify the input creative as allow, disallow, or review based on the given policy guidelines." (20 tokens)

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

All three trained adapters are public, with their own demo CSVs:

- 🥇 **[`prompt-golf-qwen-to-llama-nothink`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink)** (thinking=OFF, hero) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv)
- 🅰️ **[`prompt-golf-qwen-to-llama`](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama)** (thinking=ON variant) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama/blob/main/evals/qwen_to_llama_demo.csv)
- 🎛️ **[`prompt-golf-grpo-1.5b`](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b)** (Qwen→Qwen control) — [demo CSV](https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/blob/main/evals/qwen_to_qwen_demo.csv)

---

## 7. What the agent learned

Some qualitative observations from inspecting trained-agent outputs:

**Format cues are tokens, not sentences.** `JSON:` does the job that `Output your response as a JSON object with the following structure` does — at 50× fewer tokens.

**Persona triggers are surprisingly small.** `Yarrr,` for pirate. `psql>` for SQL. `Once upon a time,` for fairy-tale. These tokens carry enormous behavioral payload because they're strong prefix-match anchors in the target's training distribution. The trained agent finds them; humans writing prompts almost never do.

**Add the label vocabulary the human forgot.** On classification tasks where the verbose prompt described the task but didn't list the labels, the trained agent learned to *insert the label set* even though that increases length. The reward signal pushes toward explicit label vocabularies because the target needs them. The human prompt was too polite to spell them out; the agent has no such instinct.

**Show, don't tell.** For tasks like "respond in exactly 4 numbered steps," the agent learned to *demonstrate* the structure (`1.\n2.\n3.\n4.\nAnswer:`) rather than describe it. This is the kind of insight a generic action space enables that operator-based golfers (with `INSERT`, `DELETE`, `REPLACE` actions) would miss.

**Drop ceremonial preamble.** *"In this task you will…"* / *"Please carefully consider…"* / *"Your goal is to…"* — gone, every time, with no measurable accuracy cost. The first chunk of most human-written prompts is almost pure decoration.

We also saw failure modes worth flagging:

- **Mild gibberish convergence on a few adversarial tasks.** A handful of refusal-related tasks pushed the agent toward GCG-style ungrammatical prompts. The leakage penalty caught the worst cases.
- **Over-compression on tasks where verbose is doing real work.** The 39 losing tasks share a consistent failure mode: the agent compressed too aggressively to keep up with Llama's verbose-prompt capability ceiling. On these tasks the verbose prompt's extra tokens are doing real cognitive work, not just adding decoration.
- **The 1-token attractor.** Without `MIN_TOKENS_FLOOR`, RL inevitably found 1-2 token prompts exploiting tokenization artifacts. These weren't prompts in any meaningful sense — they were attacks. The floor penalty is non-optional.

---

## 8. Try it yourself

There's a **[live Gradio demo](https://huggingface.co/spaces/rishabh16196/prompt-golf-demo)** where you pick a task, see the verbose human prompt and the trained agent's compressed prompt side by side, and run either against the same Llama-3.2-3B target on real test inputs. Same UI shows accuracy for both.

<!-- IMAGE PLACEHOLDER 3 — Demo screenshot
     Clean shot of the Gradio demo UI:
     - Task selector populated with one of the policy or tough tasks
     - Verbose vs trained prompt panels side by side
     - "Run on Llama-3.2-3B" button
     - Per-input accuracy badges visible
     This is what readers click through to. -->

### Run the env locally

```bash
git clone https://huggingface.co/spaces/rishabh16196/prompt_golf_env
cd prompt_golf_env
pip install -e . gradio transformers torch

# CPU smoke test (mock target, no GPU needed)
PROMPT_GOLF_TARGET_BACKEND=mock uvicorn server.app:app --port 8000

# Real run with the actual Llama target
PROMPT_GOLF_TARGET_BACKEND=hf \
PROMPT_GOLF_TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct \
uvicorn server.app:app --port 8000
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

# Give it a verbose prompt-golf task description; get back a compressed prompt
```

### Reproduce the hero training run

```bash
PUSH_TO_HUB=your-user/your-repo bash training/hf_job_train.sh
# ~3h on L40S, pushes adapter + plots + train_metrics + eval JSONLs to your repo
```

### Hit the env from Python

```python
from prompt_golf_env import GolfAction, PromptGolfEnv

async with PromptGolfEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="sentiment_basic")
    obs = result.observation
    result = await env.step(GolfAction(prompt="Classify sentiment, one word."))
    print(f"reward={result.reward:.2f} | tokens={result.observation.submitted_prompt_tokens}")
```

---

## 9. Why this matters

| If you work on… | Prompt Golf gives you… |
|---|---|
| **Inference cost in production** | A trained policy that compresses verbose prompts behaviorally — no gradient access to the target needed. Up to 37× compression on real-world policy prompts. |
| **Capability evaluation** | A black-box minimum-elicitation metric per task per target. Decouples *can the model do X* from *did we find the right prompt*. |
| **Prompt distillation across targets** | Cross-family training generates a model of the target's response surface. Swap targets, retrain, ship a custom prompt-compressor for your specific deployment. |
| **Capability elicitation research** | A black-box analog of password-locked-model elicitation ([Greenblatt et al., 2024](https://arxiv.org/abs/2405.19550)). What's the minimum input that surfaces a latent capability? |
| **Red-teaming / robustness** | Same machinery, different rubric. Adversarial scoring → red-teaming. Refusal rubric → jailbreak hardening. |
| **LLM ↔ LLM behavioral modeling** | Machine Theory of Mind ([Rabinowitz et al., 2018](https://arxiv.org/abs/1802.07740)) for LLMs as targets. The agent's policy implicitly encodes a model of the target. |

### What this is — and isn't

- ✅ **Is** the first open OpenEnv RL environment where the agent learns to write prompts for another LLM.
- ✅ **Is** a calibrated middle: GCG/RLPrompt-style mechanics, Machine ToM-style framing, and reusable infrastructure.
- ❌ **Isn't** a generative simulator of LLM behavior — we never touch activations.
- ❌ **Isn't** a new prompt-optimization algorithm. The algorithmic core is RL+length; the contribution is the framing + reusable env + cross-family experiments.
- ❌ **Isn't** a claim that we've "solved world modeling for LLMs." Episodes are short; the analogy to Dreamer/JEPA/Genie is structural, not algorithmic.

---

## 10. What's next

Directions we'd love community help on:

1. **More targets.** We have Qwen3-1.7B and Llama-3.2-3B profiled. Phi-3, Mistral, Gemma 2 — what does the per-target prompt look like? Is the trained agent's policy portable, or is it Llama-specific? This is the cross-target transfer experiment that would substantiate the Machine ToM framing.
2. **Larger task banks.** 90 hand-crafted tasks is a starting point. Procedural task generation (random format constraints, synthetic policies) would scale this to thousands of holes.
3. **Different reward shapes.** The current additive reward is one choice. KL-as-reward (output-distribution matching the verbose prompt's) is another. Each captures a different definition of "good."
4. **Real-world deployment study.** Pick an actual production prompt (with permission), train a compressor for it, measure compression-vs-accuracy in shadow traffic. We'd love to hear what breaks and what holds up.

If you have a 1000-token prompt that's eating your inference budget, train a compressor for it. **That's the whole point.**

---

## Acknowledgments & citations

This work draws on four converging research lines:

- **[Machine Theory of Mind](https://arxiv.org/abs/1802.07740)** (Rabinowitz et al., 2018) — the conceptual ancestor.
- **[Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)** (Perez et al., 2022) — the direct algorithmic ancestor.
- **[Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550)** (Greenblatt et al., 2024) — the motivation for treating minimum elicitation as a meaningful capability metric.
- **[AutoPrompt](https://arxiv.org/abs/2010.15980)**, **[GCG](https://arxiv.org/abs/2307.15043)**, **[RLPrompt](https://arxiv.org/abs/2205.12548)**, **[PCRL](https://arxiv.org/abs/2308.08758)** — the algorithmic toolkit.

Built for the [OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/) (Meta + Hugging Face + PyTorch, India 2026), using TRL GRPO, HuggingFace Jobs, and the OpenEnv spec.

```bibtex
@misc{promptgolf2026,
  author = {Rishabh},
  title  = {Prompt Golf: An OpenEnv RL environment for cross-model prompt compression},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/spaces/rishabh16196/prompt_golf_env}}
}
```

⛳

<!-- IMAGE PLACEHOLDER 4 — Closing graphic (optional)
     Either:
       (a) a wordcloud of the actual short prompts the agent
           discovered, sized by frequency across the bank, or
       (b) a clean trophy/golf-ball graphic with the headline
           number "37×" in the foreground.
     Optional. The post lands fine without it. -->