# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Frozen-target LLM wrapper.

The target is loaded once at server startup and held in memory for the
lifetime of the process. Every episode reuses the same frozen weights —
the *agent* is what learns. Deterministic greedy decoding (temperature=0)
keeps the reward signal stable for GRPO.

Two backends:
  - "hf": load a HuggingFace causal LM via transformers. Default.
  - "mock": a rule-based stand-in for CPU boxes / CI. Lets the env boot and
    pass smoke tests without a GPU.

Select via env var PROMPT_GOLF_TARGET_BACKEND=mock|hf (default "hf").
Select the model via PROMPT_GOLF_TARGET_MODEL=Qwen/Qwen2.5-0.5B-Instruct
(or any HF causal LM id).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class TargetGeneration:
    """One target generation."""
    input_text: str
    output_text: str


class TargetBackend(Protocol):
    model_id: str

    def count_prompt_tokens(self, text: str) -> int:
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        ...

    def generate_batch(
        self,
        prompt: str,
        test_inputs: List[str],
        max_output_tokens: int,
    ) -> List[TargetGeneration]:
        ...


# ---------------------------------------------------------------------------
# Mock backend — pattern-based, deterministic, CPU-only
# ---------------------------------------------------------------------------

class MockTargetBackend:
    """Rule-based fake target for local development.

    Behavior:
      - If the prompt mentions "Answer in one word" and the input contains
        obvious sentiment cues, emit the cue word.
      - If the prompt mentions "JSON" and a pattern "name=X" / "phone=Y"
        appears, emit a JSON object.
      - For arithmetic, try to extract the last number; otherwise echo the
        first word of the input.
      - Otherwise, echo the input's first 10 chars.

    Designed to be *weakly* responsive to prompts — a smart prompt that
    steers the mock toward the right behavior gets a higher score, but
    prompts can't just leak the answer because the mock doesn't blindly
    copy the prompt.
    """

    model_id = "mock"

    # A loose token proxy: splits on whitespace + punctuation.
    _TOKEN_RE = re.compile(r"\w+|[^\w\s]")

    def count_prompt_tokens(self, text: str) -> int:
        return len(self._TOKEN_RE.findall(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        toks = self._TOKEN_RE.findall(text)
        if len(toks) <= max_tokens:
            return text
        # Re-join with single spaces; loses original whitespace but fine for mock.
        return " ".join(toks[:max_tokens])

    def generate_batch(
        self,
        prompt: str,
        test_inputs: List[str],
        max_output_tokens: int,
    ) -> List[TargetGeneration]:
        outs: List[TargetGeneration] = []
        prompt_lc = prompt.lower()

        wants_one_word = "one word" in prompt_lc or "single word" in prompt_lc
        wants_json = "json" in prompt_lc
        wants_upper = "uppercase" in prompt_lc or "all caps" in prompt_lc
        wants_number = "number" in prompt_lc or "digit" in prompt_lc

        for x in test_inputs:
            x_lc = x.lower()
            out: str

            if wants_json and ("name" in x_lc or "phone" in x_lc):
                name_m = re.search(r"name[:=]\s*(\w+)", x, re.IGNORECASE)
                phone_m = re.search(r"(\+?\d[\d\-\s]{6,})", x)
                name = name_m.group(1) if name_m else "unknown"
                phone = phone_m.group(1).strip() if phone_m else "none"
                out = f'{{"name": "{name}", "phone": "{phone}"}}'
            elif wants_one_word:
                if any(w in x_lc for w in ("love", "great", "amazing", "excellent", "happy")):
                    out = "positive"
                elif any(w in x_lc for w in ("hate", "terrible", "awful", "worst", "angry")):
                    out = "negative"
                else:
                    out = "neutral"
            elif wants_number:
                nums = re.findall(r"-?\d+\.?\d*", x)
                out = nums[-1] if nums else "0"
            elif wants_upper:
                out = x.upper()
            else:
                out = x[:40]

            # Crude output cap to respect max_output_tokens.
            toks = self._TOKEN_RE.findall(out)
            if len(toks) > max_output_tokens:
                out = " ".join(toks[:max_output_tokens])

            outs.append(TargetGeneration(input_text=x, output_text=out))

        return outs


# ---------------------------------------------------------------------------
# HF causal-LM backend
# ---------------------------------------------------------------------------

class HFTargetBackend:
    """Frozen HF causal LM target.

    Loaded lazily on first use. Uses greedy decoding and applies the
    tokenizer's chat template if available so that any instruct model can
    be plugged in.
    """

    def __init__(self, model_id: str = DEFAULT_HF_MODEL):
        self.model_id = model_id
        self._model = None
        self._tok = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Lazy import so the mock backend can be used on machines that
        # don't have torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # Decoder-only generation requires left-padding so batched inputs of
        # varying length all end at the same position before the generation
        # continues. Right-padding silently corrupts outputs on shorter inputs.
        self._tok.padding_side = "left"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def count_prompt_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tok.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        self._ensure_loaded()
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tok.decode(ids[:max_tokens], skip_special_tokens=True)

    def _format_one(self, system_prompt: str, user_input: str) -> str:
        # If the tokenizer advertises a chat template, use it.
        if getattr(self._tok, "chat_template", None):
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ]
            return self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback plain concatenation.
        return f"{system_prompt}\n\nInput: {user_input}\nOutput:"

    def generate_batch(
        self,
        prompt: str,
        test_inputs: List[str],
        max_output_tokens: int,
    ) -> List[TargetGeneration]:
        self._ensure_loaded()
        import torch

        formatted = [self._format_one(prompt, x) for x in test_inputs]
        enc = self._tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._device)

        with torch.inference_mode():
            gen = self._model.generate(
                **enc,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tok.pad_token_id,
            )

        outs: List[TargetGeneration] = []
        for i, row in enumerate(gen):
            input_len = enc["input_ids"][i].shape[0]
            new_tokens = row[input_len:]
            text = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
            outs.append(TargetGeneration(input_text=test_inputs[i], output_text=text))
        return outs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SINGLETON: Optional[TargetBackend] = None


def get_target_backend() -> TargetBackend:
    """Return a process-global target backend, loading it on first call."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON

    backend_kind = os.environ.get("PROMPT_GOLF_TARGET_BACKEND", "hf").lower().strip()
    if backend_kind == "mock":
        _SINGLETON = MockTargetBackend()
    else:
        model_id = os.environ.get("PROMPT_GOLF_TARGET_MODEL", DEFAULT_HF_MODEL)
        _SINGLETON = HFTargetBackend(model_id=model_id)
    return _SINGLETON
