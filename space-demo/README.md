---
title: Prompt Golf Demo
emoji: ⛳
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: t4-small
python_version: "3.11"
short_description: "RL-compressed prompts for Llama 3.2 (Qwen agent)"
tags:
  - prompt-engineering
  - rl
  - grpo
  - prompt-compression
  - openenv
---

# Prompt Golf — Compression Demo

An interactive demo of a Qwen3-1.7B agent (with LoRA adapter, trained via TRL GRPO on the [Prompt Golf environment](https://huggingface.co/spaces/rishabh16196/prompt_golf_env)) that writes short prompts to steer a frozen Llama-3.2-3B-Instruct target.

## How to use

1. Pick a task from the dropdown (sorted by reward gain — top entries show the biggest training wins).
2. Three prompts populate side-by-side:
   - **Verbose**: the human-written task description
   - **Untrained**: what raw Qwen3-1.7B writes when asked to compress
   - **Trained**: what the GRPO-tuned Qwen3-1.7B + LoRA writes
3. Type a test input and click **Run target with all three prompts** — the demo runs Llama-3.2-3B with each prompt prepended (in one batched forward pass) and shows the three outputs side by side.
4. Optionally click **Regenerate prompts live** to load the agent and have it produce fresh untrained / trained prompts on the fly.

## Headline numbers (90-task bank)

| Stage | Mean accuracy | Mean tokens |
|---|---|---|
| Verbose human prompt | 0.65 | ~63 |
| Untrained Qwen3-1.7B | 0.48 | ~38 |
| Trained Qwen3-1.7B + LoRA | 0.52 | ~35 |

→ **80% accuracy retention at 55% of the verbose token count.** Peak compression: **37× on long-context policy tasks** (e.g. 737-token MSN ad-creative policy → 20-token classifier prompt).

## Hardware

This Space is configured for **T4-small** ($0.40/hr). Llama-3.2-3B in bf16 fits comfortably; the agent (Qwen3-1.7B + LoRA) loads lazily on the first "Regenerate" click.

## Links

- Environment: https://huggingface.co/spaces/rishabh16196/prompt_golf_env
- Trained adapter: https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink
- Demo CSV (90 tasks × all 3 prompt columns): https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv
- Blog post: https://huggingface.co/spaces/rishabh16196/prompt_golf_env/blob/main/BLOG_POST.md
- Training notebook: https://huggingface.co/spaces/rishabh16196/prompt_golf_env/blob/main/notebooks/prompt_golf_train_minimal.ipynb

## Configuration (Space env vars)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` (required, secret) | — | Auth for downloading gated Llama-3.2 |
| `DEMO_TARGET_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Frozen target |
| `DEMO_AGENT_MODEL` | `Qwen/Qwen3-1.7B` | Agent base for live regen |
| `DEMO_AGENT_ADAPTER` | `rishabh16196/prompt-golf-qwen-to-llama-nothink` | Trained LoRA |
| `DEMO_CSV_URL` | hub URL above | Source of precomputed prompts |
