# Prompt Golf — Local Demo UI

A single-file Gradio app that loads the target model locally and shows
**verbose vs untrained vs trained** prompts side by side. Pick a task,
the app fills in the three prompts from the demo CSV; type a test
input; hit "Run target" and watch the target generate with all three
prompts in one batched forward pass.

## Run

```bash
# from repo root
pip install gradio transformers torch  # if not already in your env
python ui/demo_app.py
```

Then open `http://localhost:7860`.

## Configuration (env vars)

| Var | Default | What |
|---|---|---|
| `DEMO_TARGET_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Target model loaded for live generation. |
| `DEMO_CSV` | `outputs/qwen_to_qwen_demo.csv` | Demo CSV produced by `training/build_before_after_csv.py`. Auto-fetches the published one from HF Hub if not local. |
| `HF_TOKEN` | — | Required to download Llama-3.2 (gated). |

Examples:

```bash
# Use the same-family Qwen demo CSV with Qwen3-1.7B target
DEMO_TARGET_MODEL=Qwen/Qwen3-1.7B python ui/demo_app.py

# Once the cross-family Qwen->Llama CSV is built, point at it:
DEMO_CSV=outputs/qwen_to_llama_demo.csv python ui/demo_app.py
```

## What you see

```
[ task dropdown — sorted by reward gain ]

┌────────── VERBOSE ─────────┐ ┌──── UNTRAINED (base) ────┐ ┌──── TRAINED ────┐
│ human-written, 200-500 tok │ │ raw Qwen3 agent          │ │ Qwen3 + LoRA    │
│ tokens │ accuracy          │ │ tokens │ accuracy        │ │ tokens │ acc.   │
└────────────────────────────┘ └──────────────────────────┘ └─────────────────┘

[ test input — type your own ]
[ Run target with all three prompts ]

┌─── target output: VERBOSE ──┐ ┌── target output: UNTRAINED ──┐ ┌── target: TRAINED ──┐

batched in 1.4s | verbose: 250 tok | untrained: 35 tok | trained: 12 tok
```

## Performance notes

- **Batched generation**: all three prompts go through one
  `model.generate()` call with left-padding, so the 3-prompt round trip
  costs about the same as one inference (≈ 1-2 sec on a 4090 / L40S
  at bf16, ~3-5 sec on M-series Macs at MPS).
- Greedy decoding (`temperature=0`) for reproducibility.
- No vLLM by design — vLLM's throughput wins kick in with concurrent
  users; for a single-presenter demo, vanilla `transformers` keeps
  setup minimal and works on Mac.

## Hardware

| Setup | Llama-3.2-3B | Qwen3-1.7B |
|---|---|---|
| Mac M-series (MPS, 16 GB unified) | tight, works | comfortable |
| Mac M-series (32 GB+) | comfortable | comfortable |
| RTX 4090 / L40S (24 GB+) | comfortable | comfortable |
| T4 (16 GB) | tight; switch to Qwen target instead | comfortable |
