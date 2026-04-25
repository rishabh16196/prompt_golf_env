"""
GRPO training for Prompt Golf.

Runs the Prompt Golf env in-process (no HTTP server needed) and trains an
agent LLM to write short, effective prompts for a separate, frozen target
LLM. Designed to run as a single HuggingFace Jobs invocation.

Both models live on the same GPU:
  - AGENT (trainable)  — e.g. Qwen/Qwen2.5-1.5B-Instruct, LoRA-fine-tuned
  - TARGET (frozen)    — e.g. Qwen/Qwen2.5-0.5B-Instruct, via env.target_model

Reward (see server/rubrics.py):
    raw_task_score * length_factor * leakage_penalty
    + 0.3 * max(0, gain_over_baseline) * length_factor

Usage (local smoke test):
    python training/train_grpo.py \
        --agent-model Qwen/Qwen2.5-1.5B-Instruct \
        --target-model Qwen/Qwen2.5-0.5B-Instruct \
        --max-steps 20 \
        --num-generations 4 \
        --output-dir outputs/grpo_local

Usage (HF Jobs):
    bash training/hf_job_train.sh   # see file for full command
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple


# Make the env importable whether this script is invoked from the repo root
# (python training/train_grpo.py) or from inside training/.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))


PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a prompt engineer. Your job: write a system prompt that makes a
    separate, frozen target LLM solve the task on HIDDEN test inputs.

    Rules:
      - Output ONLY your prompt, wrapped in <prompt>...</prompt>.
      - Keep it SHORT. Shorter prompts score higher.
      - DO NOT copy train examples verbatim into your prompt — a leakage
        detector scales the reward toward zero if you do.
      - Use imperative voice. Anchor the output format tightly.
    """
).strip()


def build_agent_user_message(obs) -> str:
    examples_block = "\n".join(
        f"- input: {ex.get('input','')!r}  expected: {ex.get('expected','')!r}"
        for ex in (obs.train_examples or [])
    )
    return textwrap.dedent(
        f"""
        TASK: {obs.task_id}  (category: {obs.task_category})
        DESCRIPTION: {obs.task_description}
        TOKEN BUDGET: {obs.prompt_budget_tokens}
        TARGET: {obs.target_model_id}
        BASELINE (empty prompt) SCORE: {obs.baseline_zero_shot_score:.2f}

        Visible train examples (do not copy verbatim):
        {examples_block}

        Write your prompt inside <prompt>...</prompt>.
        """
    ).strip()


def build_chat_prompt(tokenizer, obs) -> str:
    """Apply chat template → single string the agent's tokenizer will see.

    Passes enable_thinking=False for Qwen3 models so the agent emits its
    prompt directly instead of a <think>...</think> reasoning trace
    followed by output. Thinking-mode output would also blow past our
    max_completion_length budget.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_agent_user_message(obs)},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            # Qwen3 / Qwen3.5 support this kwarg; other models ignore it.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return f"{SYSTEM_PROMPT}\n\n{build_agent_user_message(obs)}\n\nAssistant:"


def build_prompt_dataset(env, tokenizer, task_ids: List[str], seeds_per_task: int):
    """Build a HF Dataset where each row is (chat-formatted prompt, task_id, seed)."""
    from datasets import Dataset

    rows: List[Dict] = []
    for task_id in task_ids:
        for seed in range(seeds_per_task):
            obs = env.reset(task=task_id, seed=seed)
            rows.append(
                {
                    "prompt": build_chat_prompt(tokenizer, obs),
                    "task_id": task_id,
                    "seed": seed,
                }
            )
    return Dataset.from_list(rows)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_prompt(text: str) -> str:
    """Pull the <prompt>...</prompt> body from the agent's completion.

    Robust to Qwen3-style <think>...</think> reasoning blocks that may
    slip through even with enable_thinking=False by stripping those
    blocks first. Falls back to the first non-empty non-'<think>' line,
    then to a canned instruction if the completion is unusable.
    """
    text = text or ""
    # Strip any thinking blocks first so the content after them is what we inspect.
    stripped = _THINK_BLOCK_RE.sub("", text).strip()
    m = PROMPT_TAG_RE.search(stripped)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Fallback: first non-empty line that isn't a stray think tag.
    for line in stripped.split("\n"):
        line = line.strip()
        if line and not line.lower().startswith(("<think>", "</think>")):
            return line
    return "Follow the instruction. Output only the answer."


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_reward_fn(env, log_state: Dict):
    """Create a TRL-compatible reward function closed over the env.

    TRL GRPO passes `prompts`, `completions`, and any extra columns from the
    dataset as kwargs. We use `task_id` and `seed` to re-seed the env
    deterministically per group, and accept the completion as the agent's
    submitted prompt (after <prompt> extraction).
    """
    from prompt_golf_env.models import GolfAction

    def reward_fn(prompts, completions, task_id=None, seed=None, **kwargs):
        if task_id is None or seed is None:
            raise ValueError("task_id and seed must be columns in the dataset")

        rewards: List[float] = []
        tokens_log: List[int] = []
        raw_log: List[float] = []
        lf_log: List[float] = []
        lp_log: List[float] = []

        for comp, tid, s in zip(completions, task_id, seed):
            # Support either OpenAI-style list-of-dicts or plain strings
            if isinstance(comp, list):
                text = "".join(m.get("content", "") for m in comp)
            else:
                text = str(comp)
            agent_prompt = extract_prompt(text)

            env.reset(task=tid, seed=int(s))
            obs = env.step(GolfAction(prompt=agent_prompt))

            rewards.append(float(obs.reward or 0.0))
            tokens_log.append(int(obs.submitted_prompt_tokens or 0))
            raw_log.append(float(obs.raw_task_score or 0.0))
            lf_log.append(float(obs.length_factor or 0.0))
            lp_log.append(float(obs.leakage_penalty or 0.0))

        # Roll up for the step metrics file.
        n = max(1, len(rewards))
        log_state["last_batch"] = {
            "avg_reward": sum(rewards) / n,
            "avg_tokens": sum(tokens_log) / n,
            "avg_raw_score": sum(raw_log) / n,
            "avg_length_factor": sum(lf_log) / n,
            "avg_leakage_penalty": sum(lp_log) / n,
            "n": n,
        }
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Metrics logging via a Trainer callback
# ---------------------------------------------------------------------------

def make_callback(log_state: Dict, output_dir: Path):
    """Write per-step training metrics to train_metrics.jsonl."""
    from transformers import TrainerCallback

    metrics_path = output_dir / "train_metrics.jsonl"

    class PromptGolfMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            entry = {
                "step": state.global_step,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "reward": logs.get("reward"),
                "kl": logs.get("kl"),
                **(log_state.get("last_batch") or {}),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            # Echo a tidy one-liner to stdout.
            print(
                f"step {state.global_step:>4d}  "
                f"loss={entry.get('loss') or 0.0:+.4f}  "
                f"reward={entry.get('reward') or 0.0:.3f}  "
                f"avg_tokens={entry.get('avg_tokens') or 0:.1f}  "
                f"raw={entry.get('avg_raw_score') or 0:.2f}  "
                f"lf={entry.get('avg_length_factor') or 0:.2f}  "
                f"leak={entry.get('avg_leakage_penalty') or 0:.2f}",
                flush=True,
            )

    return PromptGolfMetricsCallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for Prompt Golf")
    p.add_argument("--agent-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output-dir", default="outputs/grpo")

    # Task split — held out spans v1 AND v2 for honest generalization eval
    p.add_argument(
        "--held-out-tasks",
        default=(
            "translate_numbers,reason_order,style_concise,refuse_unsafe,"
            "yaml_nested_3levels,socrates_question,sarcasm_vs_literal"
        ),
        help="Comma-separated task ids excluded from training (used for eval).",
    )
    p.add_argument("--seeds-per-task", type=int, default=4,
                   help="How many dataset rows to build per training task.")

    # GRPO knobs
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-generations", type=int, default=8,
                   help="Must divide per_device_batch_size * grad_accum_steps (default 2*4=8).")
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty")
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=1024)

    # Rollout sampling — explicit so we don't silently inherit Qwen3's
    # baked-in generation_config (temperature=0.6, top_p=0.95), which
    # is too conservative for GRPO exploration.
    p.add_argument("--rollout-temperature", type=float, default=0.9)
    p.add_argument("--rollout-top-p", type=float, default=1.0)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Misc
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push-to-hub", default=None,
                   help="If set, push the LoRA adapter to this HF repo id at end.")
    p.add_argument("--log-interval", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Set the target for the env BEFORE importing the env module.
    os.environ.setdefault("PROMPT_GOLF_TARGET_MODEL", args.target_model)
    os.environ.setdefault("PROMPT_GOLF_TARGET_BACKEND", "hf")

    # Heavy imports.
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from prompt_golf_env.server.prompt_golf_environment import PromptGolfEnvironment
    from prompt_golf_env.server.tasks import list_task_ids as list_task_ids_v1
    from prompt_golf_env.server.tasks_v2 import list_task_ids_v2
    from prompt_golf_env.server.tasks_tough import list_task_ids_tough

    # NOTE: we deliberately do NOT import Unsloth here. Unsloth patches
    # Qwen2Attention at import time, which breaks the target model
    # (also Qwen2) that we load via vanilla transformers inside the env.
    # See target_model.py — target uses AutoModelForCausalLM. Co-hosting
    # agent (would-be Unsloth) + target (vanilla) in one process requires
    # Unsloth to be absent. The proven-working path is plain TRL + PEFT
    # LoRA; on L40S a 500-step 1.5B run finishes in ~140 min.

    set_seed(args.seed)

    # ----- tokenizer (agent) -----
    tokenizer = AutoTokenizer.from_pretrained(args.agent_model)
    tokenizer.padding_side = "left"  # decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- env (target loaded lazily on first forward pass) -----
    env = PromptGolfEnvironment()
    all_tasks = list_task_ids_v1() + list_task_ids_v2() + list_task_ids_tough()
    held_out = {t.strip() for t in args.held_out_tasks.split(",") if t.strip()}
    train_tasks = [t for t in all_tasks if t not in held_out]
    print(f"[setup] tasks total={len(all_tasks)} train={len(train_tasks)} held_out={len(held_out)}", flush=True)

    # ----- dataset -----
    train_ds = build_prompt_dataset(env, tokenizer, train_tasks, args.seeds_per_task)
    eval_ds = build_prompt_dataset(env, tokenizer, sorted(held_out), seeds_per_task=2) if held_out else None
    print(f"[setup] train rows={len(train_ds)}  eval rows={len(eval_ds) if eval_ds else 0}", flush=True)

    # ----- reward + callback -----
    log_state: Dict = {}
    reward_fn = make_reward_fn(env, log_state)
    MetricsCallback = make_callback(log_state, output_dir)

    # ----- GRPO config -----
    # NOTE on temperature / top_p: Qwen3 ships a generation_config with
    # temperature=0.6, top_p=0.95. Without explicit overrides TRL picks
    # those up and our 10 candidate prompts per group come out too
    # similar -> low advantage variance -> slow GRPO learning. Setting
    # temperature=0.9 and top_p=1.0 gives the agent room to explore
    # diverse prompts per task, which is what GRPO needs.
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        bf16=args.bf16 and torch.cuda.is_available(),
        logging_steps=args.log_interval,
        save_steps=max(50, args.max_steps // 4),
        save_total_limit=2,
        seed=args.seed,
        report_to=[],  # plug in "wandb" if you want — set WANDB_API_KEY
        remove_unused_columns=False,  # keep task_id / seed in batch
    )

    # ----- LoRA -----
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ----- Train -----
    trainer = GRPOTrainer(
        model=args.agent_model,
        processing_class=tokenizer,
        args=grpo_cfg,
        reward_funcs=[reward_fn],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        callbacks=[MetricsCallback()],
    )

    t0 = time.time()
    trainer.train()
    print(f"[train] done in {time.time() - t0:.1f}s", flush=True)

    # ----- Save -----
    adapter_dir = output_dir / "adapter_final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[save] adapter at {adapter_dir}", flush=True)

    # ----- Render plots inline so they land in output_dir/plots/ -----
    metrics_path = output_dir / "train_metrics.jsonl"
    plots_dir = output_dir / "plots"
    if metrics_path.exists():
        try:
            import subprocess
            subprocess.run(
                ["python", "-u", "training/make_plots.py",
                 "--metrics", str(metrics_path),
                 "--out-dir", str(plots_dir)],
                check=False,
            )
            print(f"[plots] rendered to {plots_dir}", flush=True)
        except Exception as e:
            print(f"[plots] render failed: {e}", flush=True)

    # ----- Push adapter + metrics + plots + config to hub -----
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True, repo_type="model")

        # 1. Adapter files at repo root (so PeftModel.from_pretrained(repo_id) works)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=args.push_to_hub,
            repo_type="model",
            commit_message=f"GRPO adapter, steps={args.max_steps}",
        )

        # 2. Training artifacts (metrics, config) at root
        for fname in ("train_metrics.jsonl", "config.json"):
            src = output_dir / fname
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=fname,
                    repo_id=args.push_to_hub,
                    repo_type="model",
                    commit_message=f"upload {fname}",
                )

        # 3. Plots under plots/
        if plots_dir.exists():
            api.upload_folder(
                folder_path=str(plots_dir),
                repo_id=args.push_to_hub,
                path_in_repo="plots",
                repo_type="model",
                commit_message="training plots",
            )

        print(f"[push] uploaded adapter + artifacts to https://huggingface.co/{args.push_to_hub}", flush=True)


if __name__ == "__main__":
    main()
