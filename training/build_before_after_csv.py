"""
Build the demo before/after CSV.

For every task, joins three columns:
  - verbose_prompt    : the hand-written task description (the "before"
                        prompt a human would naturally write)
  - base_prompt       : what the untrained agent emits when asked to
                        compress
  - trained_prompt    : what the trained agent emits when asked to
                        compress

Plus token counts and reward/raw-score deltas for the demo headline:
"trained agent compresses by X% while gaining Y reward."

Inputs:
  - JSONL from `eval_before_after.py --label base    ...`
  - JSONL from `eval_before_after.py --label trained ...`
Output:
  - one CSV with one row per task

The verbose-prompt token count is measured with the SAME tokenizer the
env uses (the target's tokenizer), so it's comparable to the agent's
`submitted_prompt_tokens` numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge base/trained eval JSONLs into demo CSV")
    p.add_argument("--base-jsonl", required=True,
                   help="JSONL from eval_before_after.py --label base")
    p.add_argument("--trained-jsonl", required=True,
                   help="JSONL from eval_before_after.py --label trained")
    p.add_argument("--verbose-profile-csv", default=None,
                   help="CSV from profile_baseline.py — provides "
                        "verbose_accuracy (target's accuracy when given "
                        "the hand-written description as the prompt). "
                        "If omitted, verbose_accuracy is left blank.")
    p.add_argument("--target-model", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="Used to count tokens of the verbose description.")
    p.add_argument("--min-verbose-accuracy", type=float, default=0.0,
                   help="Drop tasks where the verbose-prompt accuracy is "
                        "≤ this value. Default 0.0 (keep everything). Set "
                        "to e.g. 0.0001 to drop only tasks where the "
                        "target genuinely fails regardless of prompt — "
                        "those tasks dilute headline numbers.")
    p.add_argument("--output-csv", default="outputs/before_after_prompts.csv")
    p.add_argument("--push-to-hub", default=None,
                   help="HF model repo id; uploaded as evals/before_after_prompts.csv")
    return p.parse_args()


def load_jsonl(path: Path) -> Dict[str, Dict]:
    """Map task_id -> row. Assumes seeds_per_task=1 (one row per task)."""
    out: Dict[str, Dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row["task_id"]
            # If multiple seeds present, keep the first (duplicates are
            # bit-identical at temp=0).
            out.setdefault(tid, row)
    return out


def load_verbose_profile(path: Path) -> Dict[str, float]:
    """Map task_id -> description_baseline (verbose-prompt accuracy)."""
    out: Dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            tid = r.get("task_id")
            val = r.get("description_baseline")
            if not tid or val in (None, "", "None"):
                continue
            try:
                out[tid] = float(val)
            except ValueError:
                continue
    return out


def main() -> None:
    args = parse_args()

    base_rows = load_jsonl(Path(args.base_jsonl))
    trained_rows = load_jsonl(Path(args.trained_jsonl))
    verbose_acc_map: Dict[str, float] = (
        load_verbose_profile(Path(args.verbose_profile_csv))
        if args.verbose_profile_csv else {}
    )

    # Task bank lookup
    from prompt_golf_env.server.tasks import TASKS
    from prompt_golf_env.server.tasks_v2 import TASKS_V2
    from prompt_golf_env.server.tasks_tough import TASKS_TOUGH
    from prompt_golf_env.server.tasks_policy import TASKS_POLICY
    _ALL_TASKS = {**TASKS, **TASKS_V2, **TASKS_TOUGH, **TASKS_POLICY}

    # Tokenizer for the verbose-prompt token count (same one the env
    # uses, so the numbers are directly comparable).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target_model)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    # Union of task ids present in either file (in case of partial runs)
    all_tids = sorted(set(base_rows) | set(trained_rows))

    # Filter on verbose-prompt accuracy floor.
    # We use the profile CSV's `verbose_accuracy` if available; for tasks
    # missing from the profile, we fall back to the trained eval's
    # raw_task_score (since 0 there + 0 on profile = target genuinely
    # can't do this task).
    if args.min_verbose_accuracy > 0:
        kept = []
        dropped = []
        for tid in all_tids:
            v = verbose_acc_map.get(tid)
            if v is None:
                # Fall back to trained eval — if BOTH base and trained
                # got 0, drop it as untrainable on this target.
                b = base_rows.get(tid, {}).get("raw_task_score", 0) or 0
                t = trained_rows.get(tid, {}).get("raw_task_score", 0) or 0
                if max(b, t) <= 0:
                    dropped.append(tid)
                    continue
                kept.append(tid)
            elif v > args.min_verbose_accuracy:
                kept.append(tid)
            else:
                dropped.append(tid)
        print(f"[csv] filtered: kept {len(kept)} / {len(all_tids)} tasks "
              f"(dropped {len(dropped)} where verbose_accuracy ≤ "
              f"{args.min_verbose_accuracy})", flush=True)
        if dropped:
            print(f"[csv]   dropped: {', '.join(dropped[:10])}"
                  + (f" ...+{len(dropped)-10} more" if len(dropped) > 10 else ""),
                  flush=True)
        all_tids = kept

    rows_out: List[Dict] = []
    for tid in all_tids:
        spec = _ALL_TASKS.get(tid)
        verbose = spec.description if spec else ""
        verbose_tokens = count_tokens(verbose) if verbose else 0

        b = base_rows.get(tid, {})
        t = trained_rows.get(tid, {})

        # Reward / token deltas
        base_reward = b.get("reward")
        trained_reward = t.get("reward")
        reward_delta = (
            None if base_reward is None or trained_reward is None
            else round(trained_reward - base_reward, 4)
        )

        base_tokens = b.get("tokens")
        trained_tokens = t.get("tokens")
        compression_ratio = (
            None if not verbose_tokens or trained_tokens is None
            else round(trained_tokens / verbose_tokens, 3)
        )

        verbose_accuracy = verbose_acc_map.get(tid)
        base_accuracy = b.get("raw_task_score")
        trained_accuracy = t.get("raw_task_score")

        # Pluck up to 3 sample test inputs (no expected outputs — those
        # stay hidden so the user can judge target outputs themselves).
        sample_inputs: List[str] = []
        if spec:
            for inp, _exp in (spec.test_examples or [])[:3]:
                sample_inputs.append(inp)

        # --- HEADLINE deltas vs the VERBOSE human-written prompt ---
        # These are the numbers we lead with in the demo: "did the
        # trained agent beat the human's verbose prompt at the rubric?"
        accuracy_delta_v = (
            None if verbose_accuracy is None or trained_accuracy is None
            else round(trained_accuracy - verbose_accuracy, 4)
        )
        # Reward advantage approximation:
        #   trained_reward includes the env's per-task baseline subtraction,
        #   leakage term, and length cost. To compare against "verbose as
        #   a prompt" without re-running the env, we use the same rubric
        #   weights (LAMBDA_LEN=0.002) and assume leakage=0 (verbose
        #   description doesn't contain held-out test n-grams). The
        #   per-task baseline cancels out since both agent and verbose
        #   would face the same baseline subtraction.
        reward_advantage_v: Optional[float] = None
        if (verbose_accuracy is not None and trained_accuracy is not None
                and verbose_tokens and trained_tokens is not None):
            LAMBDA_LEN = 0.002
            reward_advantage_v = round(
                (trained_accuracy - verbose_accuracy)
                - LAMBDA_LEN * (trained_tokens - verbose_tokens),
                4,
            )

        rows_out.append({
            "task_id": tid,
            "category": spec.category if spec else "",
            "difficulty": spec.difficulty if spec else "",
            "scorer": spec.scorer if spec else "",
            "budget_tokens": spec.budget_tokens if spec else "",
            # --- THE THREE PROMPTS ---
            "verbose_prompt": verbose,
            "verbose_tokens": verbose_tokens,
            "verbose_accuracy": (round(verbose_accuracy, 4)
                                 if verbose_accuracy is not None else None),
            "base_prompt": b.get("agent_prompt", ""),
            "base_tokens": base_tokens,
            "base_accuracy": base_accuracy,
            "trained_prompt": t.get("agent_prompt", ""),
            "trained_tokens": trained_tokens,
            "trained_accuracy": trained_accuracy,
            # --- SAMPLE TEST INPUTS (for UI demo) ---
            "sample_test_inputs": json.dumps(sample_inputs),
            # --- REWARD (rubric-composed, env-scored) ---
            "base_reward": base_reward,
            "trained_reward": trained_reward,
            # --- HEADLINE DELTAS vs VERBOSE (lead numbers) ---
            "compression_ratio_trained_vs_verbose": compression_ratio,
            "accuracy_retention_trained_vs_verbose": (
                None if not verbose_accuracy or trained_accuracy is None
                else round(trained_accuracy / verbose_accuracy, 3)
            ),
            "accuracy_delta_trained_vs_verbose": accuracy_delta_v,
            "reward_advantage_trained_vs_verbose": reward_advantage_v,
            # --- TRAINING-EFFECT delta (vs untrained) — research metric ---
            "reward_delta_trained_minus_base": reward_delta,
        })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows_out[0].keys()) if rows_out else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Summary line
    n = len(rows_out)
    valid = [r for r in rows_out
             if r["base_reward"] is not None and r["trained_reward"] is not None]
    if valid:
        avg_base_rwd = sum(r["base_reward"] for r in valid) / len(valid)
        avg_trained_rwd = sum(r["trained_reward"] for r in valid) / len(valid)

        # Accuracy averages (only over tasks that have all three numbers)
        triplet = [r for r in valid
                   if r["verbose_accuracy"] is not None
                   and r["base_accuracy"] is not None
                   and r["trained_accuracy"] is not None]
        if triplet:
            avg_v = sum(r["verbose_accuracy"] for r in triplet) / len(triplet)
            avg_b = sum(r["base_accuracy"] for r in triplet) / len(triplet)
            avg_t = sum(r["trained_accuracy"] for r in triplet) / len(triplet)
        else:
            avg_v = avg_b = avg_t = None

        avg_compress = sum(r["compression_ratio_trained_vs_verbose"]
                           for r in valid
                           if r["compression_ratio_trained_vs_verbose"] is not None
                           ) / max(1, len(valid))

        print(f"[csv] {n} tasks  ->  {out_path}", flush=True)
        print(f"[csv] avg reward:    base={avg_base_rwd:.3f}  trained={avg_trained_rwd:.3f}  "
              f"delta={avg_trained_rwd - avg_base_rwd:+.3f}", flush=True)
        if avg_v is not None:
            print(f"[csv] avg accuracy:  verbose={avg_v:.3f}  base={avg_b:.3f}  "
                  f"trained={avg_t:.3f}", flush=True)
            retention = avg_t / avg_v if avg_v > 0 else float('nan')
            print(f"[csv] accuracy retention (trained / verbose): {retention:.2%}",
                  flush=True)
            adv_rows = [r for r in valid
                        if r["reward_advantage_trained_vs_verbose"] is not None]
            if adv_rows:
                avg_adv = sum(r["reward_advantage_trained_vs_verbose"]
                              for r in adv_rows) / len(adv_rows)
                n_pos = sum(1 for r in adv_rows
                            if r["reward_advantage_trained_vs_verbose"] > 0)
                print(f"[csv] reward advantage trained vs verbose: "
                      f"mean={avg_adv:+.3f}  "
                      f"trained-beats-verbose-on={n_pos}/{len(adv_rows)} tasks",
                      flush=True)
        print(f"[csv] avg compression (trained / verbose): {avg_compress:.2%}",
              flush=True)
    else:
        print(f"[csv] {n} tasks (no scored rows) -> {out_path}", flush=True)

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True, repo_type="model")
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="evals/before_after_prompts.csv",
            repo_id=args.push_to_hub,
            repo_type="model",
            commit_message=f"before/after demo CSV: {n} tasks",
        )
        print(f"[push] uploaded to https://huggingface.co/{args.push_to_hub}",
              flush=True)


if __name__ == "__main__":
    main()
