# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Prompt Golf environment.

Each episode = one task. The agent submits a *prompt string* as its action.
The server runs a frozen target LLM against held-out test inputs using that
prompt, scores the outputs, and returns reward = task_score * length_factor.

The agent never sees the held-out test inputs — only a small number of
"train" examples that illustrate the task. This is what prevents
answer-leakage reward hacking.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Task categories and registry-level constants
# ---------------------------------------------------------------------------

TASK_CATEGORIES: List[str] = [
    "classification",   # sentiment, topic, toxicity
    "extraction",       # NER, JSON key-value
    "format",           # "exactly 3 bullets", "uppercase only"
    "arithmetic",       # word problems, percentages
    "translation",      # short-phrase translation
    "style",            # formal<->casual, active<->passive
    "reasoning",        # short multi-step problems
    "refusal",          # get target to decline unsafe requests
]

# The full task bank lives in server/tasks.py; TASK_NAMES is the index.
# Keeping it here means clients can enumerate tasks without importing the
# heavy server module.
TASK_NAMES: List[str] = [
    # classification (5)
    "sentiment_basic",
    "sentiment_nuanced",
    "topic_news",
    "toxicity_detect",
    "intent_support",
    # extraction (3)
    "ner_people",
    "json_contact",
    "number_extract",
    # format (3)
    "format_three_bullets",
    "format_uppercase",
    "format_json_object",
    # arithmetic (2)
    "arith_word",
    "arith_percent",
    # translation (2)
    "translate_greetings",
    "translate_numbers",
    # style (2)
    "style_formal",
    "style_concise",
    # reasoning (2)
    "reason_compare",
    "reason_order",
    # refusal (1)
    "refuse_unsafe",
]

# Default prompt budget in tokens (counted against the target's tokenizer).
# Episodes can override via a task-specific budget.
DEFAULT_PROMPT_BUDGET: int = 120

# Maximum tokens the target is allowed to generate per test input.
MAX_TARGET_OUTPUT_TOKENS: int = 48

# Number of held-out test examples scored per episode. Kept small to keep
# target inference under the per-step time budget.
TEST_EXAMPLES_PER_EPISODE: int = 6

# Number of visible train examples shown to the agent in the observation.
TRAIN_EXAMPLES_VISIBLE: int = 3

# --- Multi-turn ---
# When turn_limit > 1, the test pool is split:
#   - first MULTITURN_FEEDBACK_EXAMPLES are shown to the agent between
#     turns (target outputs revealed) so it can refine its prompt
#   - the remaining MULTITURN_SCORING_EXAMPLES score ONLY the final turn
# This prevents the agent from overfitting its prompt to outputs it will
# also be scored on. Single-turn (default) skips the split and scores on
# the full TEST_EXAMPLES_PER_EPISODE slice, preserving v2 behavior.
MULTITURN_FEEDBACK_EXAMPLES: int = 2
MULTITURN_SCORING_EXAMPLES: int = 4
DEFAULT_TURN_LIMIT: int = 1


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class GolfAction(Action):
    """The agent's action is the prompt text itself.

    The env will prepend this prompt to each held-out test input, pass the
    concatenation to the frozen target LLM, and score the resulting outputs.
    """

    prompt: str = Field(
        ...,
        description=(
            "Prompt text that will be prepended to each held-out test input "
            "before being sent to the frozen target model. Reward rises with "
            "task success and falls with prompt length. Over-budget prompts "
            "are truncated and incur an explicit penalty."
        ),
        max_length=4000,
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class GolfObservation(Observation):
    """Observation returned to the agent.

    On reset: contains task metadata, a small set of *train* I/O pairs, and
    the token budget. `reward` is 0 and `done` is False.

    On step (the scored step): contains the same task metadata plus the
    per-component score breakdown in `metadata`, with `reward` and
    `done=True`.
    """

    # -- Task framing --
    task_id: str = Field(default="", description="Stable task identifier (see TASK_NAMES).")
    task_category: str = Field(default="", description="Category name from TASK_CATEGORIES.")
    task_description: str = Field(
        default="",
        description=(
            "Natural-language instruction describing what the target LLM "
            "must produce on each test input. The agent should design a "
            "prompt that makes the target follow this instruction."
        ),
    )
    target_model_id: str = Field(
        default="",
        description="HF model id (or 'mock') of the frozen target the prompt will be scored against.",
    )

    # -- Budget + visible examples --
    prompt_budget_tokens: int = Field(
        default=DEFAULT_PROMPT_BUDGET,
        description=(
            "Soft cap on prompt length in target tokens. Prompts within "
            "budget earn full length credit; prompts over budget are "
            "truncated and docked by length_penalty."
        ),
    )
    max_target_output_tokens: int = Field(
        default=MAX_TARGET_OUTPUT_TOKENS,
        description="Max tokens the target will generate per test input.",
    )
    num_test_examples: int = Field(
        default=TEST_EXAMPLES_PER_EPISODE,
        description="How many held-out examples the prompt is scored on.",
    )
    train_examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Small number of visible (input, expected_output) pairs the "
            "agent can study to infer the task. These are NOT the scoring "
            "set — the scoring set is hidden."
        ),
    )
    scorer_name: str = Field(
        default="",
        description="Which scorer will be used (see server/scorer.py).",
    )

    # -- Baseline reference --
    baseline_zero_shot_score: float = Field(
        default=0.0,
        description=(
            "Target model's task score on the held-out set with an empty "
            "prompt. The agent must beat this to earn positive reward on "
            "the task component."
        ),
    )

    # -- Populated after step(), empty on reset --
    submitted_prompt_tokens: Optional[int] = Field(
        default=None,
        description="Token count of the submitted prompt (after truncation).",
    )
    raw_task_score: Optional[float] = Field(
        default=None,
        description="Target LLM's accuracy on held-out set with the submitted prompt, 0.0-1.0.",
    )
    length_factor: Optional[float] = Field(
        default=None,
        description="Length multiplier applied to the task score, 0.0-1.0.",
    )
    leakage_penalty: Optional[float] = Field(
        default=None,
        description="Anti-gaming penalty for n-gram overlap with held-out test inputs, 0.0-1.0 (1.0 = no leak).",
    )
    gain_over_baseline: Optional[float] = Field(
        default=None,
        description="raw_task_score minus baseline_zero_shot_score.",
    )
    grade_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full breakdown of reward components (only set on terminal observation).",
    )
    sample_generations: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Up to 2 (input, target_output, expected) triples from the "
            "held-out set, for debugging / demo. Only populated at step."
        ),
    )

    # --- Multi-turn fields (single-turn episodes leave these at defaults) ---
    turn_number: int = Field(
        default=1,
        description=(
            "1-indexed current turn within the episode. Always 1 for "
            "single-turn (turn_limit=1) episodes."
        ),
    )
    turn_limit: int = Field(
        default=DEFAULT_TURN_LIMIT,
        description=(
            "Total turns the agent has in this episode. Set via "
            "reset(turn_limit=N). When turn_number==turn_limit, the "
            "next step() will be terminal and scored on the held-out "
            "scoring slice."
        ),
    )
    prior_attempts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "History of attempts in this episode (only populated on "
            "non-terminal observations during multi-turn). Each entry: "
            "{prompt, tokens, feedback_score, sample_generations}. The "
            "agent uses these to refine its prompt for the next turn."
        ),
    )
