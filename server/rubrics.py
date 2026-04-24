# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward rubric for Prompt Golf.

Episodes are single-step: the agent's one action (a prompt) is scored, the
episode terminates, and the reward is a composition of four components:

  1. raw_task_score     — target LLM's accuracy on held-out test inputs
                          when prompted with the submitted prompt, in [0, 1].
  2. length_factor      — 1.0 while the prompt is within budget; decays
                          exponentially as it exceeds the budget.
  3. leakage_penalty    — 1.0 when the prompt contains no held-out test-input
                          n-grams; scales toward 0 when the agent tries to
                          paste answers into its prompt.
  4. baseline_bonus     — extra credit (weight 0.3) for beating the
                          target's zero-shot score on this task with any
                          meaningful prompt.

Final reward:
    base        = raw_task_score * length_factor * leakage_penalty
    bonus       = max(0, raw_task_score - baseline_zero_shot_score) * length_factor
    reward      = clip(base + 0.3 * bonus, 0.0, 1.3)

We return a dict with all four components so that training code can log
them separately and compose rubrics if desired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def length_factor(tokens: int, budget: int, decay_k: int = 20) -> float:
    """Length multiplier that rewards short prompts AND penalizes overshoot.

    - tokens == 0           -> 1.30 (max short-prompt bonus)
    - tokens == budget      -> 1.00 (neutral)
    - tokens > budget       -> exp(-(tokens - budget) / decay_k) (decays fast)

    The >1.0 region inside budget is what makes "shorter is better" a real
    gradient signal; otherwise truncation alone removes the incentive to
    compress once you fit.
    """
    if budget <= 0:
        budget = 1
    if tokens <= budget:
        # Linear from 1.30 at 0 tokens -> 1.00 at budget.
        return 1.0 + 0.30 * (1.0 - tokens / budget)
    over = tokens - budget
    import math
    return float(math.exp(-over / max(1, decay_k)))


def ngram_overlap(prompt: str, held_out_inputs: List[str], n: int = 4) -> float:
    """Fraction of 4-grams in held-out inputs that appear in the prompt.

    Returns 0.0 when the prompt carries no leakage, up to 1.0 when every
    4-gram from every held-out input is present in the prompt. This is
    what the leakage_penalty multiplier is built from.
    """
    prompt_norm = _normalize_for_ngrams(prompt)
    prompt_grams = set(_ngrams(prompt_norm.split(), n))
    if not prompt_grams:
        return 0.0

    total = 0
    hits = 0
    for x in held_out_inputs:
        x_norm = _normalize_for_ngrams(x)
        for gram in _ngrams(x_norm.split(), n):
            total += 1
            if gram in prompt_grams:
                hits += 1
    if total == 0:
        return 0.0
    return hits / total


def leakage_penalty(prompt: str, held_out_inputs: List[str]) -> float:
    """Convert n-gram overlap to a multiplier in [0, 1].

    1.0 == no overlap; 0.0 == perfect leak. Scales quadratically so small
    accidental overlaps aren't harshly punished but systematic copying is.
    """
    overlap = ngram_overlap(prompt, held_out_inputs, n=4)
    penalty = max(0.0, 1.0 - overlap * overlap)  # 0 leak=>1, full leak=>0
    return penalty


def _normalize_for_ngrams(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Top-level rubric
# ---------------------------------------------------------------------------

@dataclass
class RubricResult:
    reward: float
    raw_task_score: float
    length_factor: float
    leakage_penalty: float
    gain_over_baseline: float
    baseline_bonus_component: float
    submitted_tokens: int
    prompt_budget: int


class PromptGolfRubric:
    """Pure-python rubric for Prompt Golf.

    ADDITIVE formulation (v2):
        reward = success_score
               - LAMBDA_LEN * tokens
               - LAMBDA_LEAK * leakage_overlap

    where success_score = raw_task_score - BASELINE_SUBTRACT * baseline.

    Tuning rationale:
      - LAMBDA_LEN = 0.005 → with baseline tokens ~50 and raw_score ~0.25,
        the untrained baseline reward sits near 0.0 (0.25 - 0.25*0.5 - 0.005*50 = 0.0),
        giving smooth gradients in both directions.
      - LAMBDA_LEAK = 1.0 → a fully-leaked prompt (all 4-grams present)
        loses the whole raw_score contribution.
      - BASELINE_SUBTRACT = 0.5 → partially normalize against the target's
        zero-shot ability, so easy-for-target tasks don't saturate reward.

    Old fields kept on RubricResult (length_factor / leakage_penalty) for
    backward-compat logging; they're now derived rather than multiplicative.
    """

    LAMBDA_LEN: float = 0.002            # softer than v2.0 (was 0.005)
    LAMBDA_LEAK: float = 1.0
    BASELINE_SUBTRACT: float = 0.5
    MIN_TOKENS_FLOOR: int = 5            # prompts below this get a flat penalty
    MIN_TOKENS_PENALTY: float = 0.25     # ← large enough to overcome length_cost savings

    # Keep old clip boundaries so downstream plots don't break
    REWARD_CLIP_LOW: float = -0.5
    REWARD_CLIP_HIGH: float = 1.3

    def grade(
        self,
        *,
        raw_task_score: float,
        baseline_zero_shot_score: float,
        submitted_tokens: int,
        prompt_budget: int,
        prompt_text: str,
        held_out_inputs: List[str],
    ) -> RubricResult:
        overlap = ngram_overlap(prompt_text, held_out_inputs, n=4)
        # Quadratic leak penalty so small accidental overlap ≈ free,
        # systematic copying hammers.
        leak_cost = self.LAMBDA_LEAK * (overlap ** 2)

        # Length cost: linear for reasonable-length prompts; hard floor
        # below MIN_TOKENS_FLOOR to prevent degenerate policy collapse
        # to 1-token outputs on tasks where the target can't be steered.
        tokens = max(0, submitted_tokens)
        length_cost = self.LAMBDA_LEN * float(tokens)
        if tokens < self.MIN_TOKENS_FLOOR:
            # Flat penalty shrinks linearly from MIN_TOKENS_PENALTY at 0 tokens
            # to 0 at MIN_TOKENS_FLOOR tokens. Guarantees a >1-token prompt
            # beats a 1-token prompt at equal raw_score.
            short_penalty = self.MIN_TOKENS_PENALTY * (
                1.0 - tokens / max(1, self.MIN_TOKENS_FLOOR)
            )
            length_cost += short_penalty

        success = raw_task_score - self.BASELINE_SUBTRACT * baseline_zero_shot_score
        gain = raw_task_score - baseline_zero_shot_score

        reward = success - length_cost - leak_cost
        reward = float(max(self.REWARD_CLIP_LOW, min(self.REWARD_CLIP_HIGH, reward)))

        # Derived legacy fields (for log continuity with v1 metrics jsonl)
        lf_legacy = length_factor(submitted_tokens, prompt_budget)
        lp_legacy = 1.0 - overlap * overlap   # 1.0 == clean, 0.0 == leaked

        return RubricResult(
            reward=reward,
            raw_task_score=float(raw_task_score),
            length_factor=float(lf_legacy),
            leakage_penalty=float(lp_legacy),
            gain_over_baseline=float(gain),
            baseline_bonus_component=float(length_cost),  # repurposed: log length_cost
            submitted_tokens=int(submitted_tokens),
            prompt_budget=int(prompt_budget),
        )


def grade_details_dict(result: RubricResult, task_id: str, passed_threshold: float = 0.5) -> Dict[str, Any]:
    """Shape the rubric result into the metadata dict the observation exposes."""
    return {
        "task": task_id,
        "reward": round(result.reward, 4),
        "raw_task_score": round(result.raw_task_score, 4),
        "length_factor": round(result.length_factor, 4),
        "leakage_penalty": round(result.leakage_penalty, 4),
        "gain_over_baseline": round(result.gain_over_baseline, 4),
        "baseline_bonus_component": round(result.baseline_bonus_component, 4),
        "submitted_tokens": result.submitted_tokens,
        "prompt_budget": result.prompt_budget,
        "passed": result.reward >= passed_threshold,
    }
