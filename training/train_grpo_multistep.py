"""
Multi-step GRPO for Prompt Golf — model in the env loop at every turn.

Adapted from spaces_pipeline_env/local_training/grpo_multistep.py (the
proven trajectory-level GRPO recipe used in the Spaces env). Differences
for Prompt Golf:

  - Action is a free-form prompt string (not a JSON action).
  - Trajectory length = `turn_limit` (typically 2 or 3).
  - Trajectory grade = final-turn reward (`obs.reward` after step where
    `obs.done == True`). Intermediate turns are unrewarded; the agent
    only sees feedback in the next observation's `prior_attempts`.

Why this exists:  TRL's GRPOTrainer treats one prompt -> one completion.
For multi-turn we need the model to generate at every env step, observe
the resulting feedback, and refine. This script runs a custom
trajectory-level GRPO loop (REINFORCE + KL vs frozen LoRA snapshot).

Memory cost: trainable LoRA + a snapshot dict of those LoRA weights as
the reference. Both fit easily on L40S (48 GB) alongside Qwen3-1.7B
target + Qwen3-8B 8-bit judge.

Usage:
    python -u training/train_grpo_multistep.py \
        --max-steps 200 --num-gens 4 --batch-size 2 \
        --turn-limit 3 \
        --enable-thinking \
        --push-to-hub rishabh16196/prompt-golf-grpo-multistep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the prompt format + extract_prompt from the single-step trainer
# so the multi-step rollouts match the agent's training distribution
# bit-for-bit (same SYSTEM_PROMPT, same chat template, same parsing).
from training.train_grpo import (  # noqa: E402
    SYSTEM_PROMPT,
    build_agent_user_message,
    build_chat_prompt,
    extract_prompt,
)


# ---------------------------------------------------------------------------
# Trajectory containers
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_ids: torch.Tensor      # [seq_len] — chat-templated prompt
    action_ids: torch.Tensor      # [act_len] — generated tokens
    action_text: str              # extracted prompt (post-extract_prompt)


@dataclass
class Trajectory:
    task_id: str
    seed: int
    steps: List[StepRecord]
    grade: float                  # final-turn reward
    raw_task_score: float         # final-turn raw_task_score (accuracy)
    submitted_tokens: int         # final-turn prompt token count
    turns_taken: int


# ---------------------------------------------------------------------------
# Rollout: model in the loop at every env step
# ---------------------------------------------------------------------------

def rollout_episode(
    env, model, tokenizer, task_id: str, seed: int, *,
    turn_limit: int,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    device: str,
    max_prompt_tokens: int = 4096,
) -> Trajectory:
    """Run one episode. Model generates at every turn until env.done.

    Returns a Trajectory with per-turn (prompt_ids, action_ids) pairs
    used by the policy-gradient update.
    """
    from prompt_golf_env.models import GolfAction

    obs = env.reset(task=task_id, seed=seed, turn_limit=turn_limit)
    steps: List[StepRecord] = []
    grade: float = 0.0
    raw_task_score: float = 0.0
    submitted_tokens: int = 0

    model.eval()
    while not obs.done:
        # Build chat prompt — multi-turn obs carries prior_attempts which
        # build_agent_user_message folds into the user message.
        chat_str = build_chat_prompt(tokenizer, obs, enable_thinking=enable_thinking)
        prompt_ids = tokenizer(chat_str, return_tensors="pt").input_ids[0]
        if prompt_ids.shape[0] > max_prompt_tokens:
            # Left-truncate (preserve the tail with the "write your prompt" hint)
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        prompt_ids = prompt_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                prompt_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0][prompt_ids.shape[0]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        action_text = extract_prompt(gen_text)

        steps.append(StepRecord(
            prompt_ids=prompt_ids.detach().cpu(),
            action_ids=gen_ids.detach().cpu(),
            action_text=action_text,
        ))

        obs = env.step(GolfAction(prompt=action_text))
        if obs.done:
            grade = float(obs.reward or 0.0)
            raw_task_score = float(obs.raw_task_score or 0.0)
            submitted_tokens = int(obs.submitted_prompt_tokens or 0)

    return Trajectory(
        task_id=task_id,
        seed=seed,
        steps=steps,
        grade=grade,
        raw_task_score=raw_task_score,
        submitted_tokens=submitted_tokens,
        turns_taken=len(steps),
    )


# ---------------------------------------------------------------------------
# Log-prob computation (batched left-padding for memory efficiency)
# ---------------------------------------------------------------------------

def compute_logprobs_batched(
    model, records: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str, pad_id: int,
) -> List[torch.Tensor]:
    """Per-record action-token logprobs in one batched forward pass.

    Records are list of (prompt_ids, action_ids). We left-pad each
    [prompt_ids | action_ids] sequence to the max length, then read the
    a_len logits that predict each action token.
    """
    if not records:
        return []
    prompt_lens = [p.shape[0] for p, _ in records]
    action_lens = [a.shape[0] for _, a in records]
    seq_lens = [pl + al for pl, al in zip(prompt_lens, action_lens)]
    max_len = max(seq_lens)
    K = len(records)

    input_ids = torch.full((K, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((K, max_len), dtype=torch.long, device=device)
    for i, (p, a) in enumerate(records):
        full = torch.cat([p.to(device), a.to(device)], dim=0)
        input_ids[i, max_len - full.shape[0]:] = full
        attn_mask[i, max_len - full.shape[0]:] = 1

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [K, T, V]

    results: List[torch.Tensor] = []
    for i, (p, a) in enumerate(records):
        p_len, a_len = prompt_lens[i], action_lens[i]
        pad_prefix = max_len - (p_len + a_len)
        start = pad_prefix + p_len - 1
        action_logits = logits[i, start : start + a_len]  # [a_len, V]
        logprobs = F.log_softmax(action_logits.float(), dim=-1)
        action_ids_dev = a.to(device)
        token_logp = logprobs.gather(1, action_ids_dev.unsqueeze(-1)).squeeze(-1)
        results.append(token_logp)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-step GRPO for Prompt Golf")
    p.add_argument("--agent-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--target-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    p.add_argument("--sft-adapter", default=None,
                   help="Optional LoRA adapter to warm-start from "
                        "(e.g. baseline single-turn adapter). Strongly "
                        "recommended — RL on a freshly initialized "
                        "policy diverges easily.")
    p.add_argument("--output-dir", default="outputs/grpo_multistep")
    p.add_argument("--push-to-hub", default=None,
                   help="HF model repo id; pushes adapter + metrics here.")

    # Trajectory shape
    p.add_argument("--turn-limit", type=int, default=3,
                   help="Turns per episode. >1 enables multi-turn.")
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-enable-thinking", dest="enable_thinking",
                   action="store_false")

    # GRPO knobs
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-gens", type=int, default=4,
                   help="Trajectories per task per GRPO step.")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Tasks sampled per GRPO step.")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL penalty vs frozen LoRA snapshot.")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Per-turn agent generation cap. Trim from 768 to "
                        "halve forward+backward memory. Bump back if "
                        "thinking-mode answers get truncated.")
    p.add_argument("--max-prompt-tokens", type=int, default=2048,
                   help="Trim from 4096 — turn-3 prompts with "
                        "prior_attempts can hit 3-5k tokens; truncating "
                        "to 2k drops the longest prior turn first.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-micro-batch", type=int, default=2,
                   help="Records per batched forward pass. 2 halves "
                        "activation memory vs the default 4.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   default=True,
                   help="Recompute forward activations during backward "
                        "instead of caching. ~80%% activation memory "
                        "saving at ~30%% extra compute. Default ON for "
                        "multi-step because trajectory rollouts blow up "
                        "activation memory.")
    p.add_argument("--no-gradient-checkpointing",
                   dest="gradient_checkpointing", action="store_false")
    p.add_argument("--save-every", type=int, default=50)

    # LoRA (used when --sft-adapter is not given — fresh LoRA init)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Task selection
    p.add_argument("--held-out-tasks", default="",
                   help="Comma-separated task ids to exclude from training.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Run one rollout and print, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env vars consumed by the env's lazy backends
    os.environ.setdefault("PROMPT_GOLF_TARGET_MODEL", args.target_model)
    os.environ.setdefault("PROMPT_GOLF_TARGET_BACKEND", "hf")
    os.environ.setdefault("PROMPT_GOLF_JUDGE_MODEL", args.judge_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Multi-step GRPO (Prompt Golf, trajectory-level) ===", flush=True)
    print(f"  device:           {device}", flush=True)
    print(f"  agent:            {args.agent_model}", flush=True)
    print(f"  target:           {args.target_model}", flush=True)
    print(f"  judge:            {args.judge_model}", flush=True)
    print(f"  warmstart:        {args.sft_adapter or '(fresh LoRA init)'}", flush=True)
    print(f"  turn_limit:       {args.turn_limit}", flush=True)
    print(f"  enable_thinking:  {args.enable_thinking}", flush=True)
    print(f"  max_steps:        {args.max_steps}", flush=True)
    print(f"  tasks/step (B):   {args.batch_size}", flush=True)
    print(f"  gens/task (G):    {args.num_gens}", flush=True)
    print(f"  trajectories/step:{args.batch_size * args.num_gens}", flush=True)
    print(f"  lr / beta:        {args.lr} / {args.beta}", flush=True)

    # ---- Env (lazy-loads target on first use) ----
    from prompt_golf_env.server.prompt_golf_environment import (
        PromptGolfEnvironment,
        _ALL_TASKS,
    )
    env = PromptGolfEnvironment()

    # ---- Tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.agent_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model + LoRA ----
    print("\nLoading agent base model (bf16)...", flush=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    if args.sft_adapter:
        print(f"Loading adapter from {args.sft_adapter} (trainable)...", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    else:
        print("Initializing fresh LoRA adapter (no warmstart)...", flush=True)
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base, lora_cfg)
    model = model.to(device) if not torch.cuda.is_available() else model
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_tr:,}", flush=True)

    # ---- Gradient checkpointing (default ON for multi-step) ----
    # Saves ~80% activation memory at ~30% extra compute. Critical for
    # multi-step because trajectory rollouts (B × G × turn_limit records)
    # blow up activation memory during the backward pass.
    if args.gradient_checkpointing:
        # PEFT models need use_reentrant=False on modern PyTorch
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older transformers/peft don't take the kwarg
            model.gradient_checkpointing_enable()
        # PEFT requires inputs to require grad when checkpointing the base
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("  gradient_checkpointing: ENABLED", flush=True)
    else:
        print("  gradient_checkpointing: disabled", flush=True)

    # ---- Snapshot trainable weights as the KL reference ----
    print("Snapshotting trainable weights as KL reference...", flush=True)
    ref_state: Dict[str, torch.Tensor] = {
        k: v.detach().clone()
        for k, v in model.named_parameters() if v.requires_grad
    }

    # ---- Training task pool ----
    held_out = {t.strip() for t in args.held_out_tasks.split(",") if t.strip()}
    train_task_ids = [tid for tid in _ALL_TASKS.keys() if tid not in held_out]
    print(f"  task pool: {len(train_task_ids)} tasks "
          f"(held out: {len(held_out)})", flush=True)

    # ---- Optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    if args.dry_run:
        print("\n[DRY-RUN] one rollout...", flush=True)
        task = train_task_ids[0]
        traj = rollout_episode(
            env, model, tokenizer, task_id=task, seed=args.seed,
            turn_limit=args.turn_limit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            device=device,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        print(f"  task={traj.task_id} turns={traj.turns_taken} "
              f"grade={traj.grade:.3f} raw={traj.raw_task_score:.2f} "
              f"tokens={traj.submitted_tokens}", flush=True)
        for i, sr in enumerate(traj.steps):
            print(f"  turn {i+1}: action_text='{sr.action_text[:80]}' "
                  f"({sr.action_ids.shape[0]} action tokens)", flush=True)
        print("[DRY-RUN] done — no training.", flush=True)
        return

    # ---- Training loop ----
    print("\n=== starting multi-step GRPO ===\n", flush=True)
    t_train = time.time()
    metrics: List[Dict[str, Any]] = []
    STD_FLOOR = 0.1
    ADV_CLAMP = 3.0

    def swap_weights(target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Copy target_state into trainable params; return prior snapshot."""
        old: Dict[str, torch.Tensor] = {}
        for k, v in model.named_parameters():
            if v.requires_grad and k in target_state:
                old[k] = v.detach().clone()
                with torch.no_grad():
                    v.copy_(target_state[k])
        return old

    for step in range(args.max_steps):
        step_t0 = time.time()
        tasks_this_step = random.sample(
            train_task_ids, min(args.batch_size, len(train_task_ids))
        )
        seed_base = args.seed + step * 1000

        # ---- Phase 1: rollouts (no grad) ----
        all_groups: List[List[Trajectory]] = []
        for ti, task in enumerate(tasks_this_step):
            group: List[Trajectory] = []
            for g in range(args.num_gens):
                traj = rollout_episode(
                    env, model, tokenizer,
                    task_id=task, seed=seed_base + ti * 100 + g,
                    turn_limit=args.turn_limit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                    device=device,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                group.append(traj)
            all_groups.append(group)

        # ---- Group-relative advantages with std floor + clamp ----
        flat_records: List[Tuple[StepRecord, float]] = []
        group_stats = []
        n_groups_skipped = 0
        for group in all_groups:
            rewards = torch.tensor([t.grade for t in group], dtype=torch.float32)
            mean_r = rewards.mean().item()
            raw_std = rewards.std(unbiased=False).item()
            if raw_std < 0.02:  # all trajectories scored equal -> no signal
                n_groups_skipped += 1
                group_stats.append((rewards.tolist(), mean_r, 0.0))
                continue
            std_r = max(raw_std, STD_FLOOR)
            group_stats.append((rewards.tolist(), mean_r, std_r))
            for traj in group:
                adv = (traj.grade - mean_r) / std_r
                adv = max(-ADV_CLAMP, min(ADV_CLAMP, adv))
                for sr in traj.steps:
                    flat_records.append((sr, adv))

        if not flat_records:
            print(f"step {step+1:3d}/{args.max_steps}  all groups collapsed "
                  f"(equal rewards) — skipping update", flush=True)
            continue

        # ---- Phase 2: batched policy-gradient update ----
        model.train()
        optim.zero_grad()

        total_loss_val = 0.0
        total_kl_val = 0.0
        n_records = len(flat_records)
        MICRO = args.update_micro_batch

        for start in range(0, n_records, MICRO):
            batch = flat_records[start : start + MICRO]
            batch_records = [(sr.prompt_ids, sr.action_ids) for sr, _ in batch]
            batch_advs = [adv for _, adv in batch]

            # Reference logp (no grad)
            if args.beta > 0:
                saved = swap_weights(ref_state)
                with torch.no_grad():
                    ref_logps = compute_logprobs_batched(
                        model, batch_records, device, tokenizer.pad_token_id,
                    )
                swap_weights(saved)
                ref_logps = [r.detach() for r in ref_logps]
            else:
                ref_logps = [None] * len(batch)

            # New logp (with grad)
            new_logps = compute_logprobs_batched(
                model, batch_records, device, tokenizer.pad_token_id,
            )

            # REINFORCE + KL loss
            batch_loss_terms = []
            for new_lp, ref_lp, adv in zip(new_logps, ref_logps, batch_advs):
                if ref_lp is None:
                    ref_lp = new_lp.detach()
                kl_per_tok = new_lp - ref_lp
                pg_per_tok = -adv * new_lp
                loss_per_tok = pg_per_tok + args.beta * kl_per_tok
                batch_loss_terms.append(loss_per_tok.mean())
                total_kl_val += kl_per_tok.mean().item()

            micro_loss = torch.stack(batch_loss_terms).mean()
            scale = len(batch) / n_records
            (micro_loss * scale).backward()
            total_loss_val += micro_loss.item() * len(batch)

        total_loss_val = total_loss_val / max(1, n_records)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optim.step()

        # ---- Log ----
        all_rewards = [r for g in group_stats for r in g[0]]
        avg_r = sum(all_rewards) / max(1, len(all_rewards))
        max_r = max(all_rewards)
        min_r = min(all_rewards)
        avg_loss = total_loss_val
        avg_kl = total_kl_val / max(1, n_records)
        n_traj = sum(len(g) for g in all_groups)
        n_steps_in_traj = sum(len(t.steps) for g in all_groups for t in g)
        avg_tokens = (
            sum(t.submitted_tokens for g in all_groups for t in g)
            / max(1, n_traj)
        )
        avg_raw = (
            sum(t.raw_task_score for g in all_groups for t in g)
            / max(1, n_traj)
        )
        elapsed = time.time() - step_t0
        print(
            f"step {step+1:3d}/{args.max_steps}  "
            f"avg_r={avg_r:+.3f} [{min_r:+.2f}..{max_r:+.2f}]  "
            f"raw={avg_raw:.2f} tokens={avg_tokens:.1f}  "
            f"n_traj={n_traj} n_turns={n_steps_in_traj} "
            f"grp_skip={n_groups_skipped}  "
            f"loss={avg_loss:+.4f} kl={avg_kl:+.4f}  "
            f"{elapsed:.1f}s",
            flush=True,
        )
        metrics.append({
            "step": step + 1,
            "avg_reward": avg_r,
            "min_reward": min_r,
            "max_reward": max_r,
            "avg_raw_task_score": avg_raw,
            "avg_submitted_tokens": avg_tokens,
            "loss": avg_loss,
            "kl": avg_kl,
            "n_trajectories": n_traj,
            "n_turns_total": n_steps_in_traj,
            "n_groups_skipped": n_groups_skipped,
            "elapsed_s": elapsed,
        })

        if args.save_every > 0 and (step + 1) % args.save_every == 0 \
           and (step + 1) < args.max_steps:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  ckpt -> {ckpt.name}", flush=True)

    train_elapsed = time.time() - t_train
    print(f"\n=== training done in {train_elapsed/60:.1f} min ===", flush=True)

    # ---- Save adapter + metrics ----
    final_dir = out_dir / "adapter_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  adapter -> {final_dir}", flush=True)
    print(f"  metrics -> {out_dir / 'train_metrics.json'}", flush=True)

    # ---- Push to hub ----
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(final_dir),
            repo_id=args.push_to_hub,
            repo_type="model",
            path_in_repo="adapter_final",
            commit_message=f"multi-step GRPO adapter ({args.max_steps} steps, "
                           f"turn_limit={args.turn_limit}, "
                           f"thinking={args.enable_thinking})",
        )
        api.upload_file(
            path_or_fileobj=str(out_dir / "train_metrics.json"),
            path_in_repo="metrics/train_metrics_multistep.json",
            repo_id=args.push_to_hub,
            repo_type="model",
            commit_message="multi-step GRPO metrics",
        )
        print(f"[push] uploaded to https://huggingface.co/{args.push_to_hub}",
              flush=True)


if __name__ == "__main__":
    main()
