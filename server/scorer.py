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

# ---------------------------------------------------------------------------
# V2 Structural scorers (for tasks with hard, non-obvious minimum prompts)
# ---------------------------------------------------------------------------

def stepwise_math(output: str, expected: str) -> float:
    """Output must show numbered/marked reasoning steps AND match numeric answer.

    `expected` encoded as "N|<answer>" where N = min expected steps.
    Example: "2|42" → need >=2 steps and final number 42.
    """
    if "|" not in expected:
        return 0.0
    n_str, ans = expected.split("|", 1)
    try:
        n_req = int(n_str)
    except ValueError:
        return 0.0
    # Count step markers on their own lines: "1.", "Step 1", "First,", etc.
    step_re = re.compile(r"(?im)^\s*(?:step\s*\d+|\d+[.)]|first|second|then|next|finally)\b")
    n_steps = len(step_re.findall(output))
    # Numeric answer check
    ans_ok = numeric_match(output, ans) > 0
    # Partial credit: both needed for full score
    if n_steps >= n_req and ans_ok:
        return 1.0
    if n_steps >= n_req:
        return 0.4  # has structure but wrong answer
    if ans_ok:
        return 0.5  # right answer but no shown work
    return 0.0


def acrostic_match(output: str, expected: str) -> float:
    """First letter of each non-empty line must spell the expected word.

    `expected` is the target word, case-insensitive.
    """
    target = expected.strip().lower()
    if not target:
        return 0.0
    lines = [ln.strip() for ln in output.strip().split("\n") if ln.strip()]
    if len(lines) != len(target):
        # Partial: exact-length bonus, otherwise scaled
        correct = 0
        for i, ch in enumerate(target):
            if i < len(lines) and lines[i][:1].lower() == ch:
                correct += 1
        return correct / (len(target) * 2)  # capped at 0.5 when length wrong
    hits = sum(
        1 for i, ch in enumerate(target)
        if lines[i] and lines[i][:1].lower() == ch
    )
    return hits / len(target)


def avoid_letter(output: str, expected: str) -> float:
    """Output must NOT contain the specified letter (case-insensitive).

    `expected` is the forbidden letter(s). 1.0 if absent, scales down by count.
    Also requires non-trivial length (> 3 words) to guard against empty output.
    """
    forbidden = set(expected.strip().lower())
    if not forbidden:
        return 0.0
    words = output.split()
    if len(words) < 3:
        return 0.0
    out_lc = output.lower()
    violations = sum(1 for ch in out_lc if ch in forbidden)
    if violations == 0:
        return 1.0
    # Exponential decay
    import math
    return float(max(0.0, math.exp(-violations / 3.0)))


def valid_yaml_depth(output: str, expected: str) -> float:
    """Output must parse as YAML AND reach the requested nesting depth.

    `expected` = min nesting depth (int as string). Depth counted as max
    indent level. Parses best-effort (no PyYAML dep).
    """
    try:
        depth_req = int(expected.strip())
    except ValueError:
        return 0.0
    # Parse via naive indent counting — no PyYAML to avoid another dep
    max_depth = 0
    for line in output.split("\n"):
        if not line.strip() or line.strip().startswith("#"):
            continue
        # Count leading spaces (YAML uses spaces, 2-per-level canonical)
        indent = len(line) - len(line.lstrip(" "))
        level = indent // 2
        if level > max_depth:
            max_depth = level
    # Must also have a colon somewhere (key: value shape)
    if ":" not in output:
        return 0.0
    if max_depth >= depth_req:
        return 1.0
    return max_depth / max(1, depth_req)


def json_key_order(output: str, expected: str) -> float:
    """Output JSON object must have keys in the order given.

    `expected` = comma-separated key names in required order.
    """
    want_order = [k.strip() for k in expected.split(",") if k.strip()]
    if not want_order:
        return 0.0
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    # Walk top-level keys in insertion order (requires regex since stdlib
    # json loses order info — actually py3.7+ preserves it but in dict form).
    raw = match.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    got_order = list(obj.keys())
    # Compare prefix of got to required
    if len(got_order) < len(want_order):
        return 0.0
    correct = sum(1 for i, k in enumerate(want_order) if got_order[i] == k)
    return correct / len(want_order)


def ends_question(output: str, expected: str) -> float:
    """Output must end with a question mark and look like an actual question.

    `expected` is unused (pass "?").
    """
    text = output.strip()
    if not text.endswith("?"):
        return 0.0
    # Require at least one interrogative word to avoid "OK?"
    qwords = ("what", "why", "how", "when", "where", "who", "which", "could", "would", "should", "do", "does", "is", "are", "can")
    toks = set(re.findall(r"\w+", text.lower()))
    return 1.0 if any(w in toks for w in qwords) else 0.5


def word_count_exact(output: str, expected: str) -> float:
    """Output word count must exactly match expected integer.

    `expected` = "N" or "N|<min_length_chars>" to also enforce substance.
    Partial credit for ±1 word. Punctuation-only tokens don't count.
    """
    if "|" in expected:
        n_str, min_chars_str = expected.split("|", 1)
        try:
            min_chars = int(min_chars_str)
        except ValueError:
            min_chars = 0
    else:
        n_str = expected
        min_chars = 0
    try:
        n = int(n_str.strip())
    except ValueError:
        return 0.0
    words = re.findall(r"[A-Za-z0-9]+", output)
    got = len(words)
    if len(output.strip()) < min_chars:
        return 0.0
    if got == n:
        return 1.0
    if abs(got - n) == 1:
        return 0.6
    if abs(got - n) == 2:
        return 0.2
    return 0.0


def terminal_output_pattern(output: str, expected: str) -> float:
    """Output must look like terminal output: starts with prompt symbol OR
    pure command output (no prose).

    `expected` encodes an optional substring that the output must contain
    (e.g., the filename or command name). Pass "" for pattern-only.
    """
    text = output.strip()
    if not text:
        return 0.0
    prose_indicators = ("the ", "here is", "here's", "as a terminal", "sure")
    text_lc = text.lower()
    if any(ind in text_lc[:60] for ind in prose_indicators):
        return 0.0
    # Looks like terminal output if starts with $ / # / > or directly with
    # command output (e.g., ls result, cat result)
    starts_ok = bool(re.match(r"^[\$#>]|^[a-z0-9_./-]+\s", text))
    substr_ok = expected.strip() == "" or expected.strip().lower() in text_lc
    if starts_ok and substr_ok:
        return 1.0
    if starts_ok:
        return 0.6
    if substr_ok:
        return 0.3
    return 0.0


def selective_translate(output: str, expected: str) -> float:
    """Nouns translated to French, rest kept in English.

    `expected` is a '|'-separated list of required French noun translations.
    Partial credit for each noun that shows up.
    """
    required = [w.strip().lower() for w in expected.split("|") if w.strip()]
    if not required:
        return 0.0
    out_lc = output.lower()
    hits = sum(1 for w in required if w in out_lc)
    return hits / len(required)


# ---------------------------------------------------------------------------
# LLM-judge scorers (delegated to server/judge.py)
# ---------------------------------------------------------------------------

def judge_criteria(output: str, expected: str, task_description: str = "") -> float:
    """Generic judge scorer. `expected` is the evaluation criterion text.

    The judge is lazy-loaded singleton from judge.py.
    """
    # Import lazily to avoid loading judge on env construction
    try:
        from .judge import get_judge_backend
    except ImportError:
        from server.judge import get_judge_backend
    judge = get_judge_backend()
    return judge.score(
        task_description=task_description,
        output=output,
        criterion=expected,
    )


def judge_vs_expected(output: str, expected: str, task_description: str = "") -> float:
    """Judge compares output to a reference expected answer (free-form).

    For tasks where structural scoring is infeasible but we have an
    approximate gold (e.g., persona rewrites, style transfers). The
    `expected` here is the ideal reference; judge returns a similarity
    / quality score.
    """
    try:
        from .judge import get_judge_backend
    except ImportError:
        from server.judge import get_judge_backend
    judge = get_judge_backend()
    return judge.score(
        task_description=task_description,
        output=output,
        criterion=(
            "Compare the candidate output to the expected reference and "
            "score its quality and faithfulness (1.0 = perfect, 0.0 = bad)."
        ),
        expected=expected,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCORERS: Dict[str, Callable[..., float]] = {
    # v1 structural
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
    # v2 structural
    "stepwise_math": stepwise_math,
    "acrostic_match": acrostic_match,
    "avoid_letter": avoid_letter,
    "valid_yaml_depth": valid_yaml_depth,
    "json_key_order": json_key_order,
    "ends_question": ends_question,
    "word_count_exact": word_count_exact,
    "terminal_output_pattern": terminal_output_pattern,
    "selective_translate": selective_translate,
    # v2 judge-based
    "judge_criteria": judge_criteria,
    "judge_vs_expected": judge_vs_expected,
}

# Scorers that need task_description as additional context
_NEEDS_TASK_DESC = {"judge_criteria", "judge_vs_expected"}


def score_one(scorer_name: str, output: str, expected: str,
              task_description: str = "") -> float:
    """Score a single (output, expected) pair with the named scorer.

    Some scorers (judge_*) accept an extra `task_description` kwarg. The
    call site can pass it unconditionally; structural scorers ignore it.
    """
    fn = SCORERS.get(scorer_name)
    if fn is None:
        raise KeyError(f"unknown scorer: {scorer_name!r}")
    try:
        if scorer_name in _NEEDS_TASK_DESC:
            raw = fn(output, expected, task_description=task_description)
        else:
            raw = fn(output, expected)
        return float(max(0.0, min(1.0, raw)))
    except Exception:
        # Defensive: never let a scorer crash the env.
        return 0.0
