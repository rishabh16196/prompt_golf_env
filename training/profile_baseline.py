"""
Profile target-model capability per task.

For every task, runs the target with the verbose hand-written task
description as the prompt against the held-out test examples, scores
with the task's scorer, and records description_baseline per task.

This is the metric for the "is the target capable" check. We
DELIBERATELY skip computing an empty-prompt baseline — on tough tasks
that always returns ~0 and costs as much as the real measurement.

Decision rule for choosing target size:
  - description_baseline > 0.4  -> task solvable, training has headroom
  - description_baseline < 0.2  -> target undersized OR task too hard
  - mostly under 0.2 across cats -> bump target to 4B/8B

Output: CSV with one row per task. Push to hub if --push-to-hub.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-task target-capability profiler")
    p.add_argument("--target-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--target-backend", default="hf",
                   help="hf | mock (mock for local dev only)")
    p.add_argument("--tasks", default="all",
                   help="'all' or comma-separated task ids")
    p.add_argument("--output-csv", default="outputs/baseline_profile.csv")
    p.add_argument("--push-to-hub", default=None,
                   help="HF model repo id; uploaded as profiles/baseline_<target>.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["PROMPT_GOLF_TARGET_MODEL"] = args.target_model
    os.environ["PROMPT_GOLF_TARGET_BACKEND"] = args.target_backend

    # Use the target backend and scorer directly — bypassing env.reset()
    # so we don't pay for empty-prompt baseline on every task.
    from prompt_golf_env.models import MAX_TARGET_OUTPUT_TOKENS
    from prompt_golf_env.server.target_model import get_target_backend
    from prompt_golf_env.server.scorer import score_one
    from prompt_golf_env.server.tasks import TASKS
    from prompt_golf_env.server.tasks_v2 import TASKS_V2
    from prompt_golf_env.server.tasks_tough import TASKS_TOUGH

    _ALL_TASKS = {**TASKS, **TASKS_V2, **TASKS_TOUGH}

    target = get_target_backend()
    print(f"[profile] target backend ready: {target.model_id}", flush=True)

    if args.tasks == "all":
        task_ids = list(_ALL_TASKS.keys())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    print(f"[profile] target={args.target_model}  tasks={len(task_ids)}", flush=True)

    for i, tid in enumerate(task_ids):
        spec = _ALL_TASKS[tid]
        try:
            test_inputs = [x for x, _ in spec.test_examples]
            test_expected = [y for _, y in spec.test_examples]

            # Token count of the verbose description (using target's tokenizer).
            prompt_tokens = target.count_prompt_tokens(spec.description)

            # Run the target with the verbose description as the prompt.
            gens = target.generate_batch(
                prompt=spec.description,
                test_inputs=test_inputs,
                max_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
            )
            # TargetGeneration has fields (input_text, output_text)
            outputs = [g.output_text for g in gens]

            # Score each output against the corresponding expected, average.
            per_example = [
                score_one(spec.scorer, out, exp, task_description=spec.description)
                for out, exp in zip(outputs, test_expected)
            ]
            description_baseline = sum(per_example) / max(1, len(per_example))

            row = {
                "task_id": tid,
                "category": spec.category,
                "difficulty": spec.difficulty,
                "scorer": spec.scorer,
                "description_baseline": round(description_baseline, 3),
                "description_tokens": prompt_tokens,
                "budget_tokens": spec.budget_tokens,
                "n_test_examples": len(test_expected),
            }
            rows.append(row)
            print(
                f"[{i+1:3d}/{len(task_ids)}] {tid:36s} "
                f"desc={description_baseline:.2f}  "
                f"toks={prompt_tokens:4d}  "
                f"scorer={spec.scorer}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i+1:3d}/{len(task_ids)}] {tid}: ERROR  {e}", flush=True)
            rows.append({
                "task_id": tid,
                "category": spec.category,
                "difficulty": spec.difficulty,
                "scorer": spec.scorer,
                "description_baseline": None,
                "description_tokens": None,
                "budget_tokens": spec.budget_tokens,
                "n_test_examples": len(spec.test_examples),
                "error": str(e)[:200],
            })

    # ----- write CSV -----
    cols = [
        "task_id", "category", "difficulty", "scorer",
        "description_baseline", "description_tokens",
        "budget_tokens", "n_test_examples",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    elapsed = time.time() - t0
    print(f"\n[profile] {len(rows)} tasks profiled in {elapsed:.1f}s -> {out_path}",
          flush=True)

    # ----- summary table -----
    valid = [r for r in rows if r["description_baseline"] is not None]
    if valid:
        solvable = [r for r in valid if r["description_baseline"] >= 0.4]
        marginal = [r for r in valid if 0.2 <= r["description_baseline"] < 0.4]
        too_hard = [r for r in valid if r["description_baseline"] < 0.2]
        print("\n=== CAPABILITY BUCKETS ===", flush=True)
        print(f"  solvable (desc >= 0.40): {len(solvable):3d}", flush=True)
        print(f"  marginal (0.20 - 0.40):  {len(marginal):3d}", flush=True)
        print(f"  too hard (desc < 0.20):  {len(too_hard):3d}", flush=True)

        from collections import defaultdict
        by_cat: Dict[str, List[Dict]] = defaultdict(list)
        for r in valid:
            by_cat[r["category"]].append(r)
        print("\n=== BY CATEGORY (avg description_baseline) ===", flush=True)
        for cat in sorted(by_cat):
            items = by_cat[cat]
            avg = sum(it["description_baseline"] for it in items) / len(items)
            print(f"  {cat:24s} desc={avg:.2f}  (n={len(items)})", flush=True)

        print("\n=== HARDEST 10 TASKS (lowest description_baseline) ===",
              flush=True)
        hardest = sorted(valid, key=lambda r: r["description_baseline"])[:10]
        for r in hardest:
            print(f"  {r['task_id']:36s} desc={r['description_baseline']:.2f}  "
                  f"scorer={r['scorer']}", flush=True)

    # ----- push to hub -----
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True, repo_type="model")
        target_slug = args.target_model.replace("/", "_")
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"profiles/baseline_{target_slug}.csv",
            repo_id=args.push_to_hub,
            repo_type="model",
            commit_message=f"baseline profile: {args.target_model} on {len(rows)} tasks",
        )
        print(f"[push] uploaded to https://huggingface.co/{args.push_to_hub}",
              flush=True)


if __name__ == "__main__":
    main()
