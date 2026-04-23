"""
Read train_metrics.jsonl from a training run and produce the demo plots:

  - reward_curve.png    : mean reward per GRPO step (the headline)
  - length_curve.png    : mean prompt tokens per GRPO step (the compression story)
  - breakdown.png       : 2x2 grid of reward, tokens, raw_task_score, length_factor

Usage:
    python training/make_plots.py --metrics outputs/grpo/train_metrics.jsonl --out-dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def smooth(values: List[float], k: int = 10) -> List[float]:
    if not values or k <= 1:
        return values
    out = []
    window: List[float] = []
    for v in values:
        window.append(v)
        if len(window) > k:
            window.pop(0)
        out.append(sum(window) / len(window))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--smooth-k", type=int, default=10)
    p.add_argument("--title-suffix", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    rows = load(Path(args.metrics))
    if not rows:
        raise SystemExit(f"no rows in {args.metrics}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in rows]
    rewards = [r.get("avg_reward") or r.get("reward") or 0.0 for r in rows]
    tokens = [r.get("avg_tokens") or 0.0 for r in rows]
    raws = [r.get("avg_raw_score") or 0.0 for r in rows]
    lfs = [r.get("avg_length_factor") or 0.0 for r in rows]

    # --- reward curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rewards, alpha=0.4, label="per-step")
    ax.plot(steps, smooth(rewards, args.smooth_k), linewidth=2, label=f"smoothed (k={args.smooth_k})")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("mean reward (group-averaged)")
    ax.set_title(f"Prompt Golf — Reward per step{args.title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=150)
    plt.close(fig)

    # --- length curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, tokens, alpha=0.4, label="per-step")
    ax.plot(steps, smooth(tokens, args.smooth_k), linewidth=2, label=f"smoothed (k={args.smooth_k})")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("mean prompt tokens (target tokenizer)")
    ax.set_title(f"Prompt Golf — Compression emerging{args.title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "length_curve.png", dpi=150)
    plt.close(fig)

    # --- 2x2 breakdown ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    series = [
        ("reward", rewards, "mean reward"),
        ("tokens", tokens, "mean prompt tokens"),
        ("raw_task_score", raws, "raw task score"),
        ("length_factor", lfs, "length factor"),
    ]
    for ax, (name, vals, ylabel) in zip(axes.flat, series):
        ax.plot(steps, vals, alpha=0.4)
        ax.plot(steps, smooth(vals, args.smooth_k), linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Prompt Golf — training breakdown{args.title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "breakdown.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {out_dir}/")
    for f in ("reward_curve.png", "length_curve.png", "breakdown.png"):
        print(f"  {out_dir / f}")


if __name__ == "__main__":
    main()
