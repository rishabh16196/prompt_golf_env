"""
Prompt Golf — Hugging Face Spaces demo (Gradio).

Loads:
  - Llama-3.2-3B-Instruct as the frozen TARGET model
  - Qwen3-1.7B + LoRA adapter as the trained AGENT (lazy, on first
    "Regenerate live" click)
  - The demo CSV (verbose / untrained / trained prompts × 90 tasks)
    fetched from the trained-adapter repo on first launch

For each task selected: shows the three prompts side-by-side, and
runs the target on a user-provided test input with all three in one
batched forward pass — so the demo's punch is "watch the same model
produce the same answer with a 12-token prompt that the human had to
write 250 tokens for."

Designed for a HuggingFace Space with GPU (T4 / A10G / L4 / L40S).
HF_TOKEN must be configured as a Space secret (Llama-3.2 is gated).
"""

from __future__ import annotations

import csv
import io
import os
import re
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Defaults — override via Space secrets / env vars if needed
# ---------------------------------------------------------------------------

DEFAULTS = {
    "target_model": os.environ.get(
        "DEMO_TARGET_MODEL", "meta-llama/Llama-3.2-3B-Instruct"
    ),
    "agent_model": os.environ.get(
        "DEMO_AGENT_MODEL", "Qwen/Qwen3-1.7B"
    ),
    "agent_adapter": os.environ.get(
        "DEMO_AGENT_ADAPTER",
        "rishabh16196/prompt-golf-qwen-to-llama-nothink",
    ),
    "demo_csv_url": os.environ.get(
        "DEMO_CSV_URL",
        "https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/"
        "resolve/main/evals/qwen_to_llama_demo.csv",
    ),
    "max_new_tokens": 64,
    "agent_max_new_tokens": 256,
    "enable_thinking": False,  # matches the trained adapter's training config
}


# ---------------------------------------------------------------------------
# Demo CSV loader
# ---------------------------------------------------------------------------

def load_demo_rows() -> List[Dict]:
    url = DEFAULTS["demo_csv_url"]
    print(f"[demo] fetching CSV from {url}", flush=True)
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    n_total = len(rows)

    def _f(r: Dict, k: str) -> float:
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    # Filter out tasks that are dead on this target — both the human
    # verbose prompt AND the trained agent's prompt score 0. Those are
    # tasks where the target genuinely can't do the task regardless of
    # prompt, and they just clutter the demo UI dropdown.
    def _alive(r: Dict) -> bool:
        return (_f(r, "verbose_accuracy") > 0
                or _f(r, "trained_accuracy") > 0
                or _f(r, "trained_reward") > 0)

    rows = [r for r in rows if _alive(r)]

    # Sort by trained reward (desc) — most interesting tasks first
    rows.sort(key=lambda r: _f(r, "trained_reward"), reverse=True)
    print(f"[demo] loaded {len(rows)}/{n_total} rows "
          f"(filtered out tasks dead on this target)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Target / agent singletons
# ---------------------------------------------------------------------------

_TOK = None
_MODEL = None
_DEVICE = None
_AGENT_TOK = None
_AGENT_BASE = None
_AGENT_TRAINED = None


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
    _TOK.padding_side = "left"
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
    dtype = torch.bfloat16 if _DEVICE in ("cuda", "mps") else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype,
        device_map="auto" if _DEVICE == "cuda" else None,
    )
    if _DEVICE != "cuda":
        _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()
    print(f"[demo] target loaded in {time.time()-t0:.1f}s ({dtype})",
          flush=True)


def _build_target_chat(prompt: str, test_input: str) -> str:
    """Apply the target's chat template: prompt as system, test_input as user.

    Llama-3.2-3B-Instruct (and any chat-tuned target) needs this — feeding
    raw `prompt\\n\\ntest_input` makes it ramble in completion mode.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": test_input},
    ]
    if getattr(_TOK, "chat_template", None):
        return _TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    # Fallback for non-chat tokenizers
    return f"{prompt}\n\n{test_input}\n\nAssistant:"


@torch.inference_mode()
def run_target_batch(prompts: List[str], test_input: str) -> List[str]:
    load_target()
    full_texts = []
    keep_idx = []
    for i, p in enumerate(prompts):
        if p and p.strip():
            full_texts.append(_build_target_chat(p, test_input))
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
    in_len = enc["input_ids"].shape[1]
    decoded = []
    for i in range(out.shape[0]):
        new_ids = out[i][in_len:]
        decoded.append(_TOK.decode(new_ids, skip_special_tokens=True).strip())

    results = ["" for _ in prompts]
    for j, idx in enumerate(keep_idx):
        results[idx] = decoded[j]
    return results


def count_tokens(text: str) -> int:
    load_target()
    return len(_TOK.encode(text or "", add_special_tokens=False))


# ---------------------------------------------------------------------------
# Agent (lazy) — for the "Regenerate live" button
# ---------------------------------------------------------------------------

# Inlined utilities (copied from training/train_grpo.py so the Space stays
# self-contained — no need to install the full env package).
SYSTEM_PROMPT = textwrap.dedent("""
    You are a prompt engineer. Your job: write a system prompt that makes a
    separate, frozen target LLM solve the task on HIDDEN test inputs.

    Rules:
      - Output ONLY your prompt, wrapped in <prompt>...</prompt>.
      - Keep it SHORT. Shorter prompts score higher.
      - DO NOT copy train examples verbatim into your prompt — a leakage
        detector scales the reward toward zero if you do.
      - Use imperative voice. Anchor the output format tightly.
""").strip()

PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_prompt(text: str) -> str:
    text = text or ""
    stripped = THINK_BLOCK_RE.sub("", text).strip()
    m = PROMPT_TAG_RE.search(stripped)
    if m and m.group(1).strip():
        return m.group(1).strip()
    for line in stripped.split("\n"):
        line = line.strip()
        if line and not line.lower().startswith(("<think>", "</think>")):
            return line
    return "Follow the instruction. Output only the answer."


def build_user_message(task_id: str, category: str, description: str,
                       budget: int, target_model_id: str) -> str:
    return textwrap.dedent(f"""
        TASK: {task_id}  (category: {category})
        DESCRIPTION: {description}
        TOKEN BUDGET: {budget}
        TARGET: {target_model_id}
        BASELINE (empty prompt) SCORE: 0.00

        Visible train examples (do not copy verbose):
        (none)

        Write your prompt inside <prompt>...</prompt>.
    """).strip()


def load_agents() -> bool:
    global _AGENT_TOK, _AGENT_BASE, _AGENT_TRAINED
    if _AGENT_TRAINED is not None:
        return True
    if not DEFAULTS.get("agent_adapter"):
        return False
    name = DEFAULTS["agent_model"]
    adapter = DEFAULTS["agent_adapter"]
    print(f"[demo] loading agent {name} + adapter {adapter}...", flush=True)
    t0 = time.time()
    _AGENT_TOK = AutoTokenizer.from_pretrained(name)
    _AGENT_TOK.padding_side = "left"
    if _AGENT_TOK.pad_token is None:
        _AGENT_TOK.pad_token = _AGENT_TOK.eos_token
    dev = _device()
    dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
    _AGENT_BASE = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype,
        device_map="auto" if dev == "cuda" else None,
    )
    if dev != "cuda":
        _AGENT_BASE = _AGENT_BASE.to(dev)
    _AGENT_BASE.eval()

    from peft import PeftModel
    base_for_adapter = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype,
        device_map="auto" if dev == "cuda" else None,
    )
    if dev != "cuda":
        base_for_adapter = base_for_adapter.to(dev)
    _AGENT_TRAINED = PeftModel.from_pretrained(base_for_adapter, adapter)
    _AGENT_TRAINED.eval()
    print(f"[demo] agents loaded in {time.time()-t0:.1f}s", flush=True)
    return True


@torch.inference_mode()
def _agent_generate(model, tok, chat_str: str, max_new_tokens: int) -> str:
    enc = tok(chat_str, return_tensors="pt").to(_device())
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def regenerate_live(task_id: str, category: str, verbose_prompt: str,
                    budget_str: str):
    if not task_id:
        return "", "", "(no task selected)"
    if not load_agents():
        return "", "", ("agent loading disabled — set DEMO_AGENT_ADAPTER "
                        "to enable live regeneration")
    try:
        budget = int(budget_str)
    except (ValueError, TypeError):
        budget = 60

    user_msg = build_user_message(
        task_id=task_id, category=category,
        description=verbose_prompt, budget=budget,
        target_model_id=DEFAULTS["target_model"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        chat_str = _AGENT_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=DEFAULTS["enable_thinking"],
        )
    except TypeError:
        chat_str = _AGENT_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    t0 = time.time()
    raw_base = _agent_generate(
        _AGENT_BASE, _AGENT_TOK, chat_str,
        max_new_tokens=DEFAULTS["agent_max_new_tokens"],
    )
    t1 = time.time()
    raw_trained = _agent_generate(
        _AGENT_TRAINED, _AGENT_TOK, chat_str,
        max_new_tokens=DEFAULTS["agent_max_new_tokens"],
    )
    t2 = time.time()

    base_p = extract_prompt(raw_base)
    trained_p = extract_prompt(raw_trained)
    msg = (
        f"agents regenerated in {t2-t0:.1f}s "
        f"(base {t1-t0:.1f}s, trained {t2-t1:.1f}s)  |  "
        f"base: {count_tokens(base_p)} tok  |  "
        f"trained: {count_tokens(trained_p)} tok"
    )
    return base_p, trained_p, msg


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

ROWS: List[Dict] = []


def task_choices() -> List[str]:
    out = []
    for r in ROWS:
        try:
            cr = float(r.get("compression_ratio_trained_vs_verbose") or 0)
            rd = float(r.get("reward_delta_trained_minus_base") or 0)
            tag = (f"  [{int(round(1/cr))}× compress, Δr={rd:+.2f}]"
                   if cr else "")
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


def _parse_sample_inputs(row: Dict) -> List[str]:
    raw = row.get("sample_test_inputs", "") or "[]"
    try:
        import json as _json
        items = _json.loads(raw)
        return [str(x) for x in items if isinstance(x, (str, int, float))]
    except Exception:
        return []


def select_task(label: str):
    r = _row_for_label(label) or {}
    samples = _parse_sample_inputs(r)
    # Prefill the test_input box with the first sample (if any) so the
    # demo is one-click-runnable; the dropdown lets the user pick others.
    initial_input = samples[0] if samples else ""
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
        r.get("budget_tokens", "?"),
        r.get("task_id", ""),
        initial_input,
        gr.Dropdown(choices=samples, value=initial_input or None, interactive=True),
    )


def use_sample_input(sample: str) -> str:
    """Copy the chosen sample into the editable test_input box."""
    return sample or ""


def generate_three(verbose_prompt: str, base_prompt: str, trained_prompt: str,
                   test_input: str):
    if not test_input.strip():
        empty = "(enter a test input above)"
        return empty, empty, empty, ""
    t0 = time.time()
    outs = run_target_batch(
        [verbose_prompt, base_prompt, trained_prompt], test_input,
    )
    elapsed = time.time() - t0
    metrics = (
        f"batched in {elapsed:.1f}s  |  "
        f"verbose: {count_tokens(verbose_prompt)} tok  |  "
        f"untrained: {count_tokens(base_prompt)} tok  |  "
        f"trained: {count_tokens(trained_prompt)} tok"
    )
    return outs[0], outs[1], outs[2], metrics


def compress_and_run(description: str, budget_str: str, test_input: str):
    """Custom-task tab: take a free-form task description + test input,
    have the trained agent emit a compressed prompt, then run the target
    with both the user's verbose description AND the compressed prompt
    so the user can see the side-by-side."""
    description = (description or "").strip()
    test_input = (test_input or "").strip()
    if not description:
        return "", "", "", "", "", "(describe your task above)"
    if not load_agents():
        return "", "", "", "", "", ("agent loading disabled — set "
                                    "DEMO_AGENT_ADAPTER to enable this tab")
    try:
        budget = int(budget_str)
    except (ValueError, TypeError):
        budget = 60

    user_msg = build_user_message(
        task_id="custom_task", category="custom",
        description=description, budget=budget,
        target_model_id=DEFAULTS["target_model"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        chat_str = _AGENT_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=DEFAULTS["enable_thinking"],
        )
    except TypeError:
        chat_str = _AGENT_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    t0 = time.time()
    raw = _agent_generate(
        _AGENT_TRAINED, _AGENT_TOK, chat_str,
        max_new_tokens=DEFAULTS["agent_max_new_tokens"],
    )
    t1 = time.time()
    trained_prompt = extract_prompt(raw)
    trained_tok = count_tokens(trained_prompt)
    verbose_tok = count_tokens(description)

    # Build the exact chat-templated strings the target actually sees,
    # so the user can read what we send to Llama. Empty test_input
    # still produces a valid string (just an empty user turn).
    load_target()
    verbose_chat = _build_target_chat(description, test_input or "")
    trained_chat = _build_target_chat(trained_prompt, test_input or "")

    if test_input:
        # One batched forward pass with both prompts.
        outs = run_target_batch([description, trained_prompt], test_input)
        verbose_output, trained_output = outs[0], outs[1]
        t2 = time.time()
        msg = (
            f"agent: {t1-t0:.1f}s  |  target: {t2-t1:.1f}s  |  "
            f"verbose: {verbose_tok} tok  →  trained: {trained_tok} tok"
        )
    else:
        verbose_output = trained_output = "(enter a test input to run the target)"
        msg = (f"agent: {t1-t0:.1f}s  |  "
               f"verbose: {verbose_tok} tok  →  trained: {trained_tok} tok")

    return (trained_prompt, str(trained_tok), str(verbose_tok),
            verbose_output, trained_output,
            verbose_chat, trained_chat, msg)


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
            f"Compressed prompts from a Qwen3-1.7B agent (trained via GRPO), "
            f"scored against **`{DEFAULTS['target_model']}`** as the target."
        )

        with gr.Tabs():
            with gr.TabItem("Browse trained-vs-untrained"):
                gr.Markdown(
                    "Tasks ordered by reward gain (top = biggest "
                    "improvement). Three columns: **verbose** (human-"
                    "written), **untrained** (raw Qwen3), and **trained** "
                    "(after RL fine-tuning). Pick a task, type a test "
                    "input, watch the target produce outputs side by side."
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

                # Hidden state for live regen
                _task_id_state = gr.Textbox(visible=False)
                _budget_state = gr.Textbox(visible=False)

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
                with gr.Row():
                    sample_dd = gr.Dropdown(
                        choices=[],
                        label="Sample test inputs from this task (click to load)",
                        interactive=True,
                        allow_custom_value=False,
                        scale=2,
                    )
                test_input = gr.Textbox(
                    label="input",
                    lines=3,
                    placeholder=("Type or paste a test input, or pick a sample "
                                 "from the dropdown above. The three prompts will "
                                 "each be prepended to it before the target "
                                 "generates."),
                )

                with gr.Row():
                    regen_btn = gr.Button(
                        "Regenerate prompts live (loads agent + LoRA)",
                        variant="secondary",
                    )
                    run_btn = gr.Button(
                        "Run target with all three prompts", variant="primary"
                    )
                regen_status = gr.Textbox(label="agent status", interactive=False)

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

            with gr.TabItem("Try a new task"):
                gr.Markdown(
                    "Describe a brand-new task, set a token budget, and "
                    "(optionally) a test input. The trained agent will "
                    "compress your description into a short system prompt, "
                    "then the target runs both **your verbose description** "
                    "AND **the compressed prompt** on your input — so you "
                    "can see the side-by-side. First click loads the agent "
                    "+ LoRA (~6 GB)."
                )
                custom_desc = gr.Textbox(
                    label="Describe your task (used as the verbose prompt)",
                    lines=4,
                    placeholder=("e.g. Classify the input email as urgent, "
                                 "normal, or spam. Output one word."),
                )
                with gr.Row():
                    custom_budget = gr.Textbox(
                        label="Token budget", value="60", scale=1,
                    )
                    custom_input = gr.Textbox(
                        label="Test input (optional)", lines=2, scale=4,
                        placeholder="Leave blank to just see the prompt.",
                    )
                custom_btn = gr.Button(
                    "Compress with trained agent + run target",
                    variant="primary",
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Verbose (your description)")
                        custom_verbose_tok = gr.Textbox(
                            label="tokens", interactive=False,
                        )
                        custom_verbose_out = gr.Textbox(
                            label="target output", lines=6, interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("### Trained agent (compressed)")
                        custom_prompt_out = gr.Textbox(
                            label="prompt", lines=4, interactive=False,
                        )
                        custom_tok = gr.Textbox(
                            label="tokens", interactive=False,
                        )
                        custom_target_out = gr.Textbox(
                            label="target output", lines=6, interactive=False,
                        )
                custom_status = gr.Textbox(label="status", interactive=False)

                with gr.Accordion(
                    "🔍 Exact chat-templated string sent to the target "
                    "(the full Llama API call)",
                    open=False,
                ):
                    gr.Markdown(
                        "Each prompt becomes a `system` message and the "
                        "test input a `user` message; we apply the target "
                        "tokenizer's chat template (`apply_chat_template`) "
                        "with `add_generation_prompt=True`. Below is the "
                        "exact text fed to Llama for each side."
                    )
                    with gr.Row():
                        custom_verbose_chat = gr.Textbox(
                            label="Verbose call",
                            lines=10, interactive=False,
                            show_copy_button=True,
                        )
                        custom_trained_chat = gr.Textbox(
                            label="Trained call",
                            lines=10, interactive=False,
                            show_copy_button=True,
                        )

        gr.Markdown(
            "---\n"
            "**About**: this is the demo artifact for "
            "[`prompt_golf_env`](https://huggingface.co/spaces/rishabh16196/prompt_golf_env), "
            "an OpenEnv environment where the agent's *action* is a prompt "
            "and the *reward* is how well that prompt steers a frozen target "
            "LLM. The trained adapter shown here was fine-tuned with GRPO on "
            "a 90-task bank.\n"
            "- 📝 [Blog post](https://huggingface.co/spaces/rishabh16196/prompt_golf_env/blob/main/BLOG_POST.md)\n"
            "- 📊 [Demo CSV](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink/blob/main/evals/qwen_to_llama_demo.csv)\n"
            "- 🤖 [Trained adapter](https://huggingface.co/rishabh16196/prompt-golf-qwen-to-llama-nothink)"
        )

        # Wire events
        select_outputs = [
            verbose_box, base_box, trained_box, cat, scorer,
            v_tok, b_tok, t_tok, v_acc, b_acc, t_acc,
            _budget_state, _task_id_state, test_input, sample_dd,
        ]
        task_dd.change(select_task, inputs=[task_dd], outputs=select_outputs)
        sample_dd.change(use_sample_input, inputs=[sample_dd], outputs=[test_input])
        regen_btn.click(
            regenerate_live,
            inputs=[_task_id_state, cat, verbose_box, _budget_state],
            outputs=[base_box, trained_box, regen_status],
        )
        run_btn.click(
            generate_three,
            inputs=[verbose_box, base_box, trained_box, test_input],
            outputs=[out_v, out_b, out_t, metrics],
        )
        custom_btn.click(
            compress_and_run,
            inputs=[custom_desc, custom_budget, custom_input],
            outputs=[custom_prompt_out, custom_tok, custom_verbose_tok,
                     custom_verbose_out, custom_target_out,
                     custom_verbose_chat, custom_trained_chat,
                     custom_status],
        )
        app.load(select_task, inputs=[task_dd], outputs=select_outputs)

    return app


def main() -> None:
    print(f"[demo] target = {DEFAULTS['target_model']}", flush=True)
    print(f"[demo] adapter = {DEFAULTS['agent_adapter']}", flush=True)
    print(f"[demo] csv url = {DEFAULTS['demo_csv_url']}", flush=True)
    load_target()
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
