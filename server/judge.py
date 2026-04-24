# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM Judge backend for Prompt Golf.

Loads a larger frozen model (default: Qwen/Qwen3-8B) once at server startup
and uses it to score generated outputs on criteria that can't be captured
by structural/regex scorers — persona fidelity, reasoning quality, tone,
etc.

Quantized to 8-bit via bitsandbytes so a ~8B model occupies ~8 GB VRAM
alongside the target (2B bf16 ~4 GB) and the agent stack (~10 GB) on a
single L40S. CPU fallback also provided for smoke tests.

Select via env var PROMPT_GOLF_JUDGE_BACKEND=mock|hf (default "hf").
Select the model via PROMPT_GOLF_JUDGE_MODEL=Qwen/Qwen3-8B (default).
Disable quantization via PROMPT_GOLF_JUDGE_NO_QUANT=1.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Protocol


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-8B"

# Judge prompt template. The judge must output a single float in [0, 1]
# on the first line; any further explanation is ignored.
JUDGE_PROMPT_TEMPLATE = """\
You are a strict grader. Given a task and a candidate output, return a single \
floating-point score in the range [0.0, 1.0] on the FIRST line of your \
response with NO other characters on that line.

Scoring convention:
  1.0 = fully correct / fully satisfies the criterion
  0.5 = partially correct
  0.0 = incorrect or off-task

TASK DESCRIPTION:
{task_description}

EVALUATION CRITERION:
{criterion}

CANDIDATE OUTPUT:
{output}

{expected_block}First line: only the score (a number between 0 and 1).\
"""


class JudgeBackend(Protocol):
    model_id: str

    def score(self, task_description: str, output: str, criterion: str,
              expected: Optional[str] = None) -> float:
        ...


# ---------------------------------------------------------------------------
# Mock backend (deterministic pattern-based, CPU-only)
# ---------------------------------------------------------------------------

class MockJudgeBackend:
    """Rule-based fake judge for local development / CI."""

    model_id = "mock-judge"

    _KEYWORD_MAP = {
        "shakespearean": ("thou", "thy", "hath", "art", "doth", "ere"),
        "pirate": ("arr", "matey", "ye", "ahoy", "booty", "plunder"),
        "refusal": ("cannot", "won't", "unable", "decline", "against"),
        "question": ("?",),
        "terminal": ("$ ", ">>> ", "/bin/", "ls ", "cat "),
        "stepwise": ("step 1", "step 2", "first", "therefore", "so"),
    }

    def score(self, task_description: str, output: str, criterion: str,
              expected: Optional[str] = None) -> float:
        crit = criterion.lower()
        out_lc = output.lower()

        # Pick the keyword group that matches the criterion
        for key, keywords in self._KEYWORD_MAP.items():
            if key in crit:
                hits = sum(1 for kw in keywords if kw in out_lc)
                return min(1.0, hits / max(1, len(keywords) // 2))

        # Generic similarity to expected, if provided
        if expected:
            exp_lc = expected.lower()
            tokens_exp = set(re.findall(r"\w+", exp_lc))
            tokens_out = set(re.findall(r"\w+", out_lc))
            if not tokens_exp:
                return 0.0
            hits = len(tokens_exp & tokens_out)
            return min(1.0, hits / len(tokens_exp))

        return 0.5  # neutral fallback


# ---------------------------------------------------------------------------
# HF backend (Qwen3-8B, 8-bit via bitsandbytes)
# ---------------------------------------------------------------------------

class HFJudgeBackend:
    """Transformers-based judge, 8-bit quantized by default.

    Loaded lazily on first use. Uses greedy decoding for determinism.
    """

    def __init__(self, model_id: str = DEFAULT_JUDGE_MODEL, load_in_8bit: bool = True):
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tok = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"

        load_kwargs = {}
        if self.load_in_8bit and torch.cuda.is_available():
            # bitsandbytes 8-bit: ~half the bf16 footprint, negligible
            # quality loss at ~8B scale.
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                # bnb missing; fall back to bf16
                load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def score(self, task_description: str, output: str, criterion: str,
              expected: Optional[str] = None) -> float:
        self._ensure_loaded()
        import torch

        expected_block = ""
        if expected:
            expected_block = f"EXPECTED OUTPUT (reference):\n{expected}\n\n"

        user_msg = JUDGE_PROMPT_TEMPLATE.format(
            task_description=task_description,
            criterion=criterion,
            output=output,
            expected_block=expected_block,
        )

        # Prefer chat template if present (Qwen3 has one).
        if getattr(self._tok, "chat_template", None):
            messages = [
                {"role": "system", "content": "You grade outputs numerically. Output only the score."},
                {"role": "user", "content": user_msg},
            ]
            prompt_text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = user_msg

        enc = self._tok(prompt_text, return_tensors="pt", truncation=True, max_length=3072).to(self._device)

        with torch.inference_mode():
            gen = self._model.generate(
                **enc,
                max_new_tokens=16,       # only need the number
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tok.pad_token_id,
            )
        new_tokens = gen[0][enc["input_ids"].shape[1]:]
        raw = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(text: str) -> float:
        first_line = text.strip().split("\n", 1)[0].strip()
        # Try to parse a float from the first line
        m = re.search(r"-?\d+\.?\d*", first_line)
        if not m:
            return 0.0
        try:
            v = float(m.group(0))
        except ValueError:
            return 0.0
        return float(max(0.0, min(1.0, v)))


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_SINGLETON: Optional[JudgeBackend] = None


def get_judge_backend() -> JudgeBackend:
    """Process-global judge, loaded on first call."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON

    backend_kind = os.environ.get("PROMPT_GOLF_JUDGE_BACKEND", "hf").lower().strip()
    if backend_kind == "mock":
        _SINGLETON = MockJudgeBackend()
    else:
        model_id = os.environ.get("PROMPT_GOLF_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        load_in_8bit = os.environ.get("PROMPT_GOLF_JUDGE_NO_QUANT", "") == ""
        _SINGLETON = HFJudgeBackend(model_id=model_id, load_in_8bit=load_in_8bit)
    return _SINGLETON
