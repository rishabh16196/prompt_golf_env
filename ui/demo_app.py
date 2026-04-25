"""
Prompt Golf — interactive demo UI (Gradio).

Loads:
  - the frozen TARGET model (Llama-3.2-3B-Instruct or Qwen3-1.7B)
  - the demo CSV with verbose/base/trained prompts per task

For each task you pick, shows the verbose hand-written prompt next to
the trained agent's compressed prompt, then runs the target with both
on a test input and renders the outputs side by side. Punchline:
"verbose 250 tokens, trained 12 tokens, both produce the correct
answer."

Run:
    pip install gradio
    python ui/demo_app.py

By default loads on GPU (cuda or mps). Edit DEFAULTS below to switch
target model or demo CSV path.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Repo root on path so we can resolve fixture paths relative to it.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Defaults — edit these to point at your trained CSV / target model.
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Target model the prompts will be SCORED against.
    "target_model": os.environ.get(
        "DEMO_TARGET_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
    ),
    # CSV produced by training/build_before_after_csv.py
    "demo_csv": os.environ.get(
        "DEMO_CSV", str(_REPO_ROOT / "outputs" / "qwen_to_qwen_demo.csv")
    ),
    # If the CSV isn't local, you can pull it from the hub:
    "fallback_csv_url": (
        "https://huggingface.co/rishabh16196/prompt-golf-grpo-1.5b/"
        "resolve/main/evals/qwen_to_qwen_demo.csv"
    ),
    "max_new_tokens": 64,
    "temperature": 0.0,
}


# ---------------------------------------------------------------------------
# Load demo CSV
# ---------------------------------------------------------------------------

def _ensure_csv(path: str, url: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    print(f"[demo] {path} not found locally — fetching from hub...", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r, open(p, "wb") as f:
        f.write(r.read())
    print(f"[demo] downloaded -> {p}", flush=True)
    return str(p)


def load_demo_rows() -> List[Dict]:
    csv_path = _ensure_csv(DEFAULTS["demo_csv"], DEFAULTS["fallback_csv_url"])
    rows = list(csv.DictReader(open(csv_path)))
    # Sort by reward delta (desc) so the most interesting tasks bubble up
    def _delta(r: Dict) -> float:
        try:
            return float(r.get("reward_delta_trained_minus_base") or 0)
        except ValueError:
            return 0.0
    rows.sort(key=_delta, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Target model singleton
# ---------------------------------------------------------------------------

_TOK = None
_MODEL = None
_DEVICE = None


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_target() -> None:
    global _TOK, _MODEL, _DEVICE
    if _MODEL is not None:
        return
    _DEVICE = _device()
    name = DEFAULTS["target_model"]
    print(f"[demo] loading target {name} on {_DEVICE}...", flush=True)
    t0 = time.time()
    _TOK = AutoTokenizer.from_pretrained(name)
    # Left-padding is required for decoder-only batched generation —
    # otherwise the right-padded sequences get attention drift on
    # generated continuations.
    _TOK.padding_side = "left"
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
    dtype = torch.bfloat16 if _DEVICE in ("cuda", "mps") else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto" if _DEVICE == "cuda" else None,
    )
    if _DEVICE != "cuda":
        _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()
    print(f"[demo] loaded in {time.time()-t0:.1f}s on {_DEVICE} ({dtype})",
          flush=True)


@torch.inference_mode()
def run_target_batch(prompts: List[str], test_input: str) -> List[str]:
    """Run the target on (prompt[i] + test_input) for all i, in one batched
    forward pass. Returns one decoded output per prompt (in order).

    Empty prompts in the list are returned as empty strings without
    incurring inference cost.
    """
    load_target()
    # Build the full prompts; track which positions are non-empty.
    full_texts = []
    keep_idx = []
    for i, p in enumerate(prompts):
        if p and p.strip():
            full_texts.append(f"{p}\n\n{test_input}".strip())
            keep_idx.append(i)
    if not full_texts:
        return ["" for _ in prompts]

    enc = _TOK(full_texts, return_tensors="pt", padding=True,
               truncation=True, max_length=4096).to(_DEVICE)
    out = _MODEL.generate(
        **enc,
        max_new_tokens=DEFAULTS["max_new_tokens"],
        do_sample=False,
        temperature=1.0,
        pad_token_id=_TOK.pad_token_id,
    )
    # With left-padding, all sequences in the batch share the same
    # prompt-end column = enc["input_ids"].shape[1]. The new tokens
    # are everything beyond that.
    in_len = enc["input_ids"].shape[1]
    decoded = []
    for i in range(out.shape[0]):
        new_ids = out[i][in_len:]
        decoded.append(_TOK.decode(new_ids, skip_special_tokens=True).strip())

    # Re-thread results back into the original ordering (with blanks for
    # the prompts we skipped).
    results = ["" for _ in prompts]
    for j, idx in enumerate(keep_idx):
        results[idx] = decoded[j]
    return results


# Back-compat shim for any caller still using the old name
@torch.inference_mode()
def run_target(prompt: str, test_input: str) -> str:
    return run_target_batch([prompt], test_input)[0]


def count_tokens(text: str) -> int:
    load_target()
    return len(_TOK.encode(text or "", add_special_tokens=False))


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

ROWS: List[Dict] = []


def task_choices() -> List[str]:
    # "task_id (compress 8x, +0.34 reward)"
    out = []
    for r in ROWS:
        try:
            cr = float(r.get("compression_ratio_trained_vs_verbose") or 0)
            rd = float(r.get("reward_delta_trained_minus_base") or 0)
            tag = f"  [{int(round(1/cr))}x compress, Δr={rd:+.2f}]" if cr else ""
        except (ValueError, ZeroDivisionError):
            tag = ""
        out.append(f"{r['task_id']}{tag}")
    return out


def _row_for_label(label: str) -> Optional[Dict]:
    if not label:
        return None
    tid = label.split()[0]
    for r in ROWS:
        if r["task_id"] == tid:
            return r
    return None


def select_task(label: str):
    r = _row_for_label(label) or {}
    return (
        r.get("verbose_prompt", ""),
        r.get("base_prompt", ""),
        r.get("trained_prompt", ""),
        r.get("category", ""),
        r.get("scorer", ""),
        r.get("verbose_tokens", "?"),
        r.get("base_tokens", "?"),
        r.get("trained_tokens", "?"),
        r.get("verbose_accuracy", "?"),
        r.get("base_accuracy", "?"),
        r.get("trained_accuracy", "?"),
        "",  # test_input — start blank so user types their own
    )


def generate_three(
    verbose_prompt: str, base_prompt: str, trained_prompt: str,
    test_input: str,
):
    """Run target with verbose / untrained-base / trained in ONE batched
    forward pass; return three outputs + a metrics line.
    """
    if not test_input.strip():
        empty = "(enter a test input above)"
        return empty, empty, empty, ""
    t0 = time.time()
    outs = run_target_batch(
        [verbose_prompt, base_prompt, trained_prompt],
        test_input,
    )
    elapsed = time.time() - t0
    metrics = (
        f"batched in {elapsed:.1f}s  |  "
        f"verbose: {count_tokens(verbose_prompt)} tok  |  "
        f"untrained: {count_tokens(base_prompt)} tok  |  "
        f"trained: {count_tokens(trained_prompt)} tok"
    )
    return outs[0], outs[1], outs[2], metrics


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    global ROWS
    ROWS = load_demo_rows()
    initial = task_choices()[0] if ROWS else ""

    with gr.Blocks(
        title="Prompt Golf — Compression Demo",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            f"# Prompt Golf — Compression Demo\n"
            f"Compressed prompts from a Qwen3-1.7B agent, scored against "
            f"**`{DEFAULTS['target_model']}`** as the target.  "
            f"Tasks ordered by reward gain (top = biggest improvement)."
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=task_choices(),
                value=initial,
                label="Task",
                scale=4,
            )
            cat = gr.Textbox(label="category", interactive=False, scale=1)
            scorer = gr.Textbox(label="scorer", interactive=False, scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Verbose (human-written)")
                verbose_box = gr.Textbox(
                    label="prompt", lines=8, interactive=True,
                )
                with gr.Row():
                    v_tok = gr.Textbox(label="tokens", interactive=False)
                    v_acc = gr.Textbox(label="accuracy", interactive=False)
            with gr.Column():
                gr.Markdown("### Untrained agent (base)")
                base_box = gr.Textbox(
                    label="prompt", lines=8, interactive=True,
                )
                with gr.Row():
                    b_tok = gr.Textbox(label="tokens", interactive=False)
                    b_acc = gr.Textbox(label="accuracy", interactive=False)
            with gr.Column():
                gr.Markdown("### Trained agent (compressed)")
                trained_box = gr.Textbox(
                    label="prompt", lines=8, interactive=True,
                )
                with gr.Row():
                    t_tok = gr.Textbox(label="tokens", interactive=False)
                    t_acc = gr.Textbox(label="accuracy", interactive=False)

        gr.Markdown("### Test input — edit to try your own")
        test_input = gr.Textbox(
            label="input",
            lines=3,
            placeholder=("Type or paste a test input that the prompt above "
                         "should be applied to."),
        )

        run_btn = gr.Button("Run target with all three prompts", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Target output — VERBOSE")
                out_v = gr.Textbox(label="output", lines=4, interactive=False)
            with gr.Column():
                gr.Markdown("### Target output — UNTRAINED")
                out_b = gr.Textbox(label="output", lines=4, interactive=False)
            with gr.Column():
                gr.Markdown("### Target output — TRAINED")
                out_t = gr.Textbox(label="output", lines=4, interactive=False)

        metrics = gr.Textbox(label="metrics", interactive=False)

        gr.Markdown(
            "---\n"
            "**Notes**\n\n"
            "- Verbose prompts are the hand-written task descriptions; "
            "trained prompts are what a Qwen3-1.7B agent produced after "
            "GRPO training on this task bank.\n"
            "- Token counts use the target's tokenizer.\n"
            "- The target is run greedy (`temperature=0`) for reproducibility."
        )

        # Wire events
        select_outputs = [
            verbose_box, base_box, trained_box, cat, scorer,
            v_tok, b_tok, t_tok, v_acc, b_acc, t_acc, test_input,
        ]
        task_dd.change(select_task, inputs=[task_dd], outputs=select_outputs)
        run_btn.click(
            generate_three,
            inputs=[verbose_box, base_box, trained_box, test_input],
            outputs=[out_v, out_b, out_t, metrics],
        )
        # Trigger initial population
        app.load(select_task, inputs=[task_dd], outputs=select_outputs)

    return app


def main() -> None:
    print(f"[demo] target = {DEFAULTS['target_model']}", flush=True)
    print(f"[demo] csv    = {DEFAULTS['demo_csv']}", flush=True)
    # Pre-load target so the first generation isn't a cold start
    load_target()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
