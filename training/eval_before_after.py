"""
Before/After evaluation for Prompt Golf.

Loads the agent model (optionally with a LoRA adapter), runs it on every
task in the task bank (or a specified subset), captures the prompt it
produces, runs that prompt through the env, and writes a results JSONL +
a readable summary table.

Run twice — once without --adapter (base), once with --adapter (trained)
— then diff the tables for the hackathon demo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List

import torch


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the agent system prompt + user message + extractor from train_grpo
# so eval mirrors training exactly.
from training.train_grpo import (  # noqa: E402
    SYSTEM_PROMPT,
    build_agent_user_message,
    extract_prompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt Golf eval harness")
    p.add_argument("--agent-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None,
                   help="Optional LoRA adapter dir or HF repo id.")
    p.add_argument("--target-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tasks", default="all",
                   help="'all' or comma-separated task ids.")
    p.add_argument("--seeds-per-task", type=int, default=3)
    p.add_argument("--output-json", default="outputs/eval_results.jsonl")
    p.add_argument("--label", default="base",
                   help="Label to tag this eval run (e.g. 'base', 'trained').")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_agent(agent_model: str, adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(agent_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        agent_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        print(f"[load] agent + adapter = {agent_model} + {adapter}", flush=True)
    else:
        print(f"[load] agent (base) = {agent_model}", flush=True)
    return model, tok


def build_chat_string(tok, obs) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_agent_user_message(obs)},
    ]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{build_agent_user_message(obs)}\n\nAssistant:"


@torch.inference_mode()
def generate_prompt(model, tok, chat_str: str, max_new_tokens: int, temperature: float) -> str:
    enc = tok(chat_str, return_tensors="pt").to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=1.0,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = gen[0][enc["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_prompt(text)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PROMPT_GOLF_TARGET_MODEL", args.target_model)
    os.environ.setdefault("PROMPT_GOLF_TARGET_BACKEND", "hf")

    from prompt_golf_env.models import GolfAction
    from prompt_golf_env.server.prompt_golf_environment import PromptGolfEnvironment
    from prompt_golf_env.server.tasks import TASKS, list_task_ids

    # Load agent
    model, tok = load_agent(args.agent_model, args.adapter)

    # Pick tasks
    if args.tasks == "all":
        task_ids = list_task_ids()
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Env
    env = PromptGolfEnvironment()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    for task_id in task_ids:
        for seed in range(args.seeds_per_task):
            obs = env.reset(task=task_id, seed=seed)
            chat_str = build_chat_string(tok, obs)
            agent_prompt = generate_prompt(
                model, tok, chat_str,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            obs_after = env.step(GolfAction(prompt=agent_prompt))
            row = {
                "label": args.label,
                "task_id": task_id,
                "category": TASKS[task_id].category,
                "difficulty": TASKS[task_id].difficulty,
                "seed": seed,
                "agent_prompt": agent_prompt,
                "tokens": obs_after.submitted_prompt_tokens,
                "budget": obs_after.prompt_budget_tokens,
                "raw_task_score": obs_after.raw_task_score,
                "length_factor": obs_after.length_factor,
                "leakage_penalty": obs_after.leakage_penalty,
                "baseline_zero_shot": obs_after.baseline_zero_shot_score,
                "reward": obs_after.reward,
            }
            rows.append(row)

    # Write JSONL
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Print summary
    elapsed = time.time() - t0
    print(f"\n[eval] {len(rows)} episodes in {elapsed:.1f}s  →  {out_path}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    # Aggregate by task
    by_task: Dict[str, List[Dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    header = f'{"task_id":24s}  {"reward":>7s}  {"raw":>5s}  {"tokens":>6s}  {"lf":>5s}  {"leak":>5s}'
    print(header)
    print("-" * len(header))
    total_reward = 0.0
    total_tokens = 0.0
    for tid, items in by_task.items():
        avg = {
            k: sum((it.get(k) or 0.0) for it in items) / len(items)
            for k in ("reward", "raw_task_score", "tokens", "length_factor", "leakage_penalty")
        }
        print(
            f'{tid:24s}  {avg["reward"]:7.3f}  {avg["raw_task_score"]:5.2f}  '
            f'{avg["tokens"]:6.1f}  {avg["length_factor"]:5.2f}  {avg["leakage_penalty"]:5.2f}'
        )
        total_reward += avg["reward"]
        total_tokens += avg["tokens"]
    n = len(by_task)
    print("-" * len(header))
    print(
        f'{"AVERAGE":24s}  {total_reward/n:7.3f}  {"":>5s}  '
        f'{total_tokens/n:6.1f}'
    )


if __name__ == "__main__":
    main()
