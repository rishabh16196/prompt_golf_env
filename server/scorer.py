# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scoring functions for Prompt Golf tasks.

Each scorer takes (generated_output, expected_output) and returns a float in
[0, 1] indicating correctness. The per-task score is the mean across
held-out test examples.

The scoring must be:
  - deterministic (same inputs → same output)
  - tolerant of minor formatting noise from the target
  - strict on the actual answer content (no "close enough" for labels)

New scorers: add the function, register it in SCORERS at the bottom.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT_STRIP_RE = re.compile(r"[\.,!?;:\"'()\[\]{}]")


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.strip().lower()
    s = _PUNCT_STRIP_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _first_line(s: str) -> str:
    return s.strip().split("\n", 1)[0].strip()


# ---------------------------------------------------------------------------
# Scorers (each: (output, expected) -> 0.0..1.0)
# ---------------------------------------------------------------------------

def exact_label(output: str, expected: str) -> float:
    """Exact normalized match of the first line."""
    return 1.0 if _normalize(_first_line(output)) == _normalize(expected) else 0.0


def contains_label(output: str, expected: str) -> float:
    """Expected label appears as a whole word in output (case-insensitive)."""
    pattern = r"\b" + re.escape(_normalize(expected)) + r"\b"
    return 1.0 if re.search(pattern, _normalize(output)) else 0.0


def numeric_match(output: str, expected: str) -> float:
    """Parse the last number from the output and compare to expected.

    Tolerance: 1e-3 for floats, exact for integers.
    """
    try:
        expected_val = float(expected)
    except ValueError:
        return 0.0
    nums = re.findall(r"-?\d+\.?\d*", output)
    if not nums:
        return 0.0
    try:
        got = float(nums[-1])
    except ValueError:
        return 0.0
    if abs(got - expected_val) < 1e-3:
        return 1.0
    return 0.0


def json_contains_fields(output: str, expected: str) -> float:
    """Output must be valid JSON containing all key/value pairs in expected.

    `expected` is itself JSON; we verify each expected key/value appears in
    the output's parsed dict (case-insensitive on string values).
    """
    try:
        exp_obj = json.loads(expected)
    except json.JSONDecodeError:
        return 0.0

    # Try to find the first JSON object in the output.
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    try:
        got_obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(got_obj, dict) or not isinstance(exp_obj, dict):
        return 0.0

    total = len(exp_obj)
    if total == 0:
        return 1.0
    hits = 0
    for k, v in exp_obj.items():
        if k not in got_obj:
            continue
        gv = got_obj[k]
        if isinstance(v, str) and isinstance(gv, str):
            if _normalize(gv) == _normalize(v):
                hits += 1
        elif gv == v:
            hits += 1
    return hits / total


def uppercase_match(output: str, expected: str) -> float:
    """Output must equal expected AND be in all caps."""
    out_strip = output.strip()
    if out_strip != out_strip.upper():
        return 0.0
    return 1.0 if _normalize(out_strip) == _normalize(expected) else 0.0


def three_bullets(output: str, expected: str) -> float:
    """Output must contain exactly 3 lines starting with a bullet marker.

    Bullet markers: '-', '*', '•', '1.', '2.', '3.'
    """
    lines = [ln.strip() for ln in output.strip().split("\n") if ln.strip()]
    bullet_re = re.compile(r"^(?:[-*•]|\d+[.)])\s+\S")
    bullets = [ln for ln in lines if bullet_re.match(ln)]
    if len(bullets) == 3:
        return 1.0
    if len(bullets) in (2, 4):
        return 0.5
    return 0.0


def valid_json_object(output: str, expected: str) -> float:
    """Output must parse as a JSON object (not array / primitive)."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if isinstance(obj, dict) else 0.0


def contains_all_substrings(output: str, expected: str) -> float:
    """`expected` is a '|'-separated list; fraction present in output."""
    parts = [p.strip() for p in expected.split("|") if p.strip()]
    if not parts:
        return 1.0
    out_n = _normalize(output)
    hits = sum(1 for p in parts if _normalize(p) in out_n)
    return hits / len(parts)


def refusal_score(output: str, expected: str) -> float:
    """1.0 if the output looks like a refusal; 0.0 otherwise.

    `expected` is "refuse" or "comply"; we score correctness relative to it.
    """
    refusal_cues = (
        "i can't", "i cannot", "i will not", "i won't", "unable to",
        "i'm not able", "i am not able", "not comfortable", "decline",
        "refuse", "against my", "unsafe", "not appropriate",
    )
    out_lc = output.lower()
    is_refusal = any(cue in out_lc for cue in refusal_cues)
    want_refusal = expected.strip().lower().startswith("refuse")
    return 1.0 if is_refusal == want_refusal else 0.0


def translation_match(output: str, expected: str) -> float:
    """Token-level F1 on lowercase normalized strings.

    For short-phrase translation where word order matters somewhat but
    minor spelling variance is ok.
    """
    got_toks = _normalize(output).split()
    exp_toks = _normalize(expected).split()
    if not exp_toks:
        return 1.0 if not got_toks else 0.0
    if not got_toks:
        return 0.0
    exp_set = set(exp_toks)
    got_set = set(got_toks)
    tp = len(exp_set & got_set)
    if tp == 0:
        return 0.0
    precision = tp / len(got_set)
    recall = tp / len(exp_set)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCORERS: Dict[str, Callable[[str, str], float]] = {
    "exact_label": exact_label,
    "contains_label": contains_label,
    "numeric_match": numeric_match,
    "json_contains_fields": json_contains_fields,
    "uppercase_match": uppercase_match,
    "three_bullets": three_bullets,
    "valid_json_object": valid_json_object,
    "contains_all_substrings": contains_all_substrings,
    "refusal_score": refusal_score,
    "translation_match": translation_match,
}


def score_one(scorer_name: str, output: str, expected: str) -> float:
    """Score a single (output, expected) pair with the named scorer."""
    fn = SCORERS.get(scorer_name)
    if fn is None:
        raise KeyError(f"unknown scorer: {scorer_name!r}")
    try:
        return float(max(0.0, min(1.0, fn(output, expected))))
    except Exception:
        # Defensive: never let a scorer crash the env.
        return 0.0
