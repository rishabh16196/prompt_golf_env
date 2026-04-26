"""
Replay our existing train_metrics.jsonl files into a Trackio dashboard
hosted on a HuggingFace Space.

Each trained-adapter repo on the Hub has a `train_metrics.jsonl` with
per-step metrics (reward, raw_task_score, avg_tokens, loss, ...).
We log every row into a separate Trackio run, configured with that
adapter's hyperparameters and the resulting demo-CSV summary numbers.

After running, the dashboard is live at:
    https://huggingface.co/spaces/<TRACKIO_SPACE>

Run:
    python training/replay_to_trackio.py --space-id rishabh16196/prompt-golf-trackio
"""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
from typing import Dict, List, Optional

import trackio


# Per-adapter metadata: how the run was configured + which hub repo
# holds its train_metrics.jsonl. Add more entries as runs land.
RUNS = [
    {
        "name": "qwen_to_qwen_baseline",
        "group": "single-turn",
        "repo": "rishabh16196/prompt-golf-grpo-1.5b",
        "config": {
            "agent_model": "Qwen/Qwen3-1.7B",
            "target_model": "Qwen/Qwen3-1.7B",
            "judge_model": "Qwen/Qwen3-8B",
            "thinking": False,
            "turn_limit": 1,
            "max_steps": 500,
            "num_generations": 8,
            "lr": 5e-6,
            "beta": 0.04,
            "task_bank_size": 87,
            "story": "same-family control (weak target)",
        },
    },
    {
        "name": "qwen_to_llama_thinkoff_hero",
        "group": "single-turn",
        "repo": "rishabh16196/prompt-golf-qwen-to-llama-nothink",
        "config": {
            "agent_model": "Qwen/Qwen3-1.7B",
            "target_model": "meta-llama/Llama-3.2-3B-Instruct",
            "judge_model": "Qwen/Qwen3-8B",
            "thinking": False,
            "turn_limit": 1,
            "max_steps": 500,
            "num_generations": 8,
            "lr": 5e-6,
            "beta": 0.04,
            "task_bank_size": 90,
            "story": "cross-family hero",
        },
    },
    {
        "name": "qwen_to_llama_thinkon",
        "group": "single-turn",
        "repo": "rishabh16196/prompt-golf-qwen-to-llama",
        "config": {
            "agent_model": "Qwen/Qwen3-1.7B",
            "target_model": "meta-llama/Llama-3.2-3B-Instruct",
            "judge_model": "Qwen/Qwen3-8B",
            "thinking": True,
            "turn_limit": 1,
            "max_steps": 500,
            "num_generations": 8,
            "lr": 5e-6,
            "beta": 0.04,
            "task_bank_size": 90,
            "story": "cross-family thinking-ON A/B variant",
        },
    },
    {
        "name": "qwen_to_llama_multistep",
        "group": "multi-turn",
        "repo": "rishabh16196/prompt-golf-multistep-llama",
        "config": {
            "agent_model": "Qwen/Qwen3-1.7B",
            "target_model": "meta-llama/Llama-3.2-3B-Instruct",
            "judge_model": "Qwen/Qwen3-8B",
            "thinking": False,
            "turn_limit": 3,
            "max_steps": 150,
            "num_generations": 4,
            "lr": 2e-6,
            "beta": 0.04,
            "warmstart_from": "rishabh16196/prompt-golf-qwen-to-llama-nothink",
            "story": "trajectory-level GRPO, warmstarted",
        },
    },
    {
        "name": "llama_to_llama_self",
        "group": "single-turn",
        "repo": "rishabh16196/prompt-golf-llama-self",
        "config": {
            "agent_model": "meta-llama/Llama-3.2-3B-Instruct",
            "target_model": "meta-llama/Llama-3.2-3B-Instruct",
            "judge_model": "Qwen/Qwen3-8B",
            "thinking": False,
            "turn_limit": 1,
            "max_steps": 500,
            "num_generations": 8,
            "lr": 5e-6,
            "beta": 0.04,
            "task_bank_size": 90,
            "story": "self-improvement: Llama writes prompts for Llama",
        },
    },
]


def fetch_jsonl(repo: str, path: str = "train_metrics.jsonl") -> Optional[List[Dict]]:
    """Pull a JSONL file from a Hub model repo. Returns None if missing."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
    except Exception as e:
        print(f"  [skip] {repo}/{path}: {e}", flush=True)
        return None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def replay_run(run_meta: Dict, project: str, space_id: str) -> bool:
    """Push one run's per-step metrics into Trackio. Returns True if logged."""
    rows = fetch_jsonl(run_meta["repo"])
    if not rows:
        return False

    print(f"  [{run_meta['name']}] {len(rows)} steps from {run_meta['repo']}",
          flush=True)
    run = trackio.init(
        project=project,
        name=run_meta["name"],
        group=run_meta.get("group"),
        space_id=space_id,
        config=run_meta.get("config", {}),
        resume="never",
    )
    for row in rows:
        # Prefer explicit step; fall back to position
        step = row.get("step")
        # Strip the step from the metric dict so it isn't re-logged as a metric
        metrics = {k: v for k, v in row.items() if k != "step" and v is not None}
        # Coerce to scalar where possible
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)):
                clean[k] = v
            elif isinstance(v, str):
                # Skip free-form strings to keep dashboard charts clean
                continue
        if clean:
            trackio.log(clean, step=step)
    trackio.finish()
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space-id", default="rishabh16196/prompt-golf-trackio",
                   help="HF Space to host the Trackio dashboard.")
    p.add_argument("--project", default="prompt-golf",
                   help="Project name within the Trackio dashboard.")
    p.add_argument("--only", default=None,
                   help="Comma-separated run names to replay (default: all).")
    args = p.parse_args()

    target_runs = RUNS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        target_runs = [r for r in RUNS if r["name"] in wanted]
    print(f"Replaying {len(target_runs)} runs to "
          f"https://huggingface.co/spaces/{args.space_id}", flush=True)

    n_logged = 0
    for r in target_runs:
        if replay_run(r, project=args.project, space_id=args.space_id):
            n_logged += 1

    print(f"\nDone. {n_logged}/{len(target_runs)} runs replayed.", flush=True)
    print(f"Dashboard: https://huggingface.co/spaces/{args.space_id}", flush=True)


if __name__ == "__main__":
    main()
