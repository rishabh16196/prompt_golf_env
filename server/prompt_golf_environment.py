# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Prompt Golf Environment Implementation.

Episode shape (single step):
  - reset(task="..."): env picks a task, computes baseline_zero_shot score
                       once, returns a GolfObservation with task_description,
                       3 visible train examples, prompt budget, etc.
  - step(GolfAction(prompt)):
       * truncate prompt to budget if needed
       * run the frozen target LLM on held-out test inputs with the prompt
       * score each output with the task's scorer
       * compute length factor + leakage penalty + baseline bonus
       * return terminal GolfObservation with reward and component breakdown

Reset caches the baseline score per (target, task) so we pay it once per
process — not every reset.

The task bank is in tasks.py. The scorer fns are in scorer.py. The frozen
target is in target_model.py. The reward rubric is in rubrics.py.
"""

from __future__ import annotations

import random
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DEFAULT_PROMPT_BUDGET,
        DEFAULT_TURN_LIMIT,
        MAX_TARGET_OUTPUT_TOKENS,
        MULTITURN_FEEDBACK_EXAMPLES,
        MULTITURN_SCORING_EXAMPLES,
        TEST_EXAMPLES_PER_EPISODE,
        TRAIN_EXAMPLES_VISIBLE,
        GolfAction,
        GolfObservation,
    )
    from .rubrics import PromptGolfRubric, grade_details_dict
    from .scorer import score_one
    from .target_model import TargetBackend, TargetGeneration, get_target_backend
    from .tasks import TASKS, TaskSpec, get_task, list_task_ids
    from .tasks_v2 import TASKS_V2
    from .tasks_tough import TASKS_TOUGH
except ImportError:
    from models import (
        DEFAULT_PROMPT_BUDGET,
        DEFAULT_TURN_LIMIT,
        MAX_TARGET_OUTPUT_TOKENS,
        MULTITURN_FEEDBACK_EXAMPLES,
        MULTITURN_SCORING_EXAMPLES,
        TEST_EXAMPLES_PER_EPISODE,
        TRAIN_EXAMPLES_VISIBLE,
        GolfAction,
        GolfObservation,
    )
    from server.rubrics import PromptGolfRubric, grade_details_dict
    from server.scorer import score_one
    from server.target_model import TargetBackend, TargetGeneration, get_target_backend
    from server.tasks import TASKS, TaskSpec, get_task, list_task_ids
    from server.tasks_v2 import TASKS_V2
    from server.tasks_tough import TASKS_TOUGH

# Merged v1 + v2 + tough task bank. task_ids don't clash by construction
# (v2 tasks are uniquely named, tough tasks are prefixed `tough_`).
_ALL_TASKS = {**TASKS, **TASKS_V2, **TASKS_TOUGH}


# Baseline zero-shot scores are (target_id, task_id) -> score. Computed on
# demand, cached for the lifetime of the process.
_BASELINE_CACHE: dict[tuple[str, str], float] = {}


class PromptGolfEnvironment(Environment):
    """
    Single-step env. Each episode = one prompt attempt on one task.

    The target LLM is a process-wide singleton. The agent's action is a
    prompt string; the env tokenizes it against the target's tokenizer,
    truncates to budget, runs the target on held-out inputs, scores, and
    terminates.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()

        # Current task
        self._task: Optional[TaskSpec] = None
        # Resampled every reset
        self._train_ex: List[tuple[str, str]] = []
        self._test_ex: List[tuple[str, str]] = []
        # Multi-turn slices (only populated when turn_limit > 1)
        self._feedback_ex: List[tuple[str, str]] = []
        self._scoring_ex: List[tuple[str, str]] = []
        # Cached per-episode baseline (target with empty prompt)
        self._baseline_zero_shot: float = 0.0

        # Multi-turn state (single-turn defaults preserve v2 behavior)
        self._turn_count: int = 0
        self._turn_limit: int = DEFAULT_TURN_LIMIT
        self._prior_attempts: List[dict] = []

        # Reward rubric (stateless per episode)
        self._rubric = PromptGolfRubric()

        # Frozen target — lazy-load to keep import light
        self._target: TargetBackend = get_target_backend()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        turn_limit: int = DEFAULT_TURN_LIMIT,
    ) -> GolfObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._task = self._choose_task(task)

        # Reset multi-turn state
        self._turn_count = 0
        self._turn_limit = max(1, int(turn_limit))
        self._prior_attempts = []

        # Sample visible train examples (stable for this episode)
        train_pool = list(self._task.train_examples)
        self._rng.shuffle(train_pool)
        self._train_ex = train_pool[:TRAIN_EXAMPLES_VISIBLE]

        # Select held-out test examples. For tasks with >TEST_EXAMPLES_PER_EPISODE
        # test cases, pick a random slice so different episodes score on
        # different subsets — discourages overfitting to a fixed set.
        test_pool = list(self._task.test_examples)
        self._rng.shuffle(test_pool)
        self._test_ex = test_pool[:TEST_EXAMPLES_PER_EPISODE]

        # Multi-turn split: feedback slice (revealed between turns) vs
        # scoring slice (only ever scored on the FINAL turn). Single-turn
        # episodes leave both empty and use _test_ex as before.
        if self._turn_limit > 1:
            self._feedback_ex = self._test_ex[:MULTITURN_FEEDBACK_EXAMPLES]
            self._scoring_ex = self._test_ex[
                MULTITURN_FEEDBACK_EXAMPLES:
                MULTITURN_FEEDBACK_EXAMPLES + MULTITURN_SCORING_EXAMPLES
            ]
            # Guarantee a non-empty scoring slice even on tasks with few
            # test examples — fall back to the full slice.
            if not self._scoring_ex:
                self._scoring_ex = list(self._test_ex)
        else:
            self._feedback_ex = []
            self._scoring_ex = []

        # Compute (or reuse) baseline for this task with empty prompt
        cache_key = (self._target.model_id, self._task.task_id)
        if cache_key not in _BASELINE_CACHE:
            _BASELINE_CACHE[cache_key] = self._score_prompt("")
        self._baseline_zero_shot = _BASELINE_CACHE[cache_key]

        return GolfObservation(
            task_id=self._task.task_id,
            task_category=self._task.category,
            task_description=self._task.description,
            target_model_id=self._target.model_id,
            prompt_budget_tokens=self._task.budget_tokens or DEFAULT_PROMPT_BUDGET,
            max_target_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
            num_test_examples=(
                len(self._scoring_ex) if self._turn_limit > 1
                else len(self._test_ex)
            ),
            train_examples=[
                {"input": x, "expected": y} for (x, y) in self._train_ex
            ],
            scorer_name=self._task.scorer,
            baseline_zero_shot_score=round(self._baseline_zero_shot, 4),
            done=False,
            reward=0.0,
            turn_number=1,
            turn_limit=self._turn_limit,
            prior_attempts=[],
            metadata={
                "task_difficulty": self._task.difficulty,
                "task_tags": list(self._task.tags),
            },
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: GolfAction) -> GolfObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._task is None:
            raise RuntimeError("step() called before reset()")

        # Bump turn counter; `is_final_turn` decides scoring slice + done-flag.
        self._turn_count += 1
        is_final_turn = self._turn_count >= self._turn_limit

        # Truncate prompt to the task's budget (in target tokens).
        budget = self._task.budget_tokens or DEFAULT_PROMPT_BUDGET
        truncated_prompt = self._target.truncate_to_tokens(action.prompt, budget)
        submitted_tokens = self._target.count_prompt_tokens(truncated_prompt)

        # Pick the scoring slice for THIS turn:
        # - single-turn (turn_limit=1): score on the full _test_ex (v2 behavior)
        # - multi-turn non-final: score on _feedback_ex (cheap, revealed to agent)
        # - multi-turn final:    score on _scoring_ex (held-out, drives reward)
        if self._turn_limit > 1:
            scoring_slice = self._scoring_ex if is_final_turn else self._feedback_ex
        else:
            scoring_slice = self._test_ex

        raw_task_score, sample_gens = self._score_prompt(
            truncated_prompt, return_samples=True, examples=scoring_slice,
        )

        # ----- Non-final turn in multi-turn: return feedback obs (done=False) -----
        if not is_final_turn:
            self._prior_attempts.append({
                "turn": self._turn_count,
                "prompt": truncated_prompt,
                "tokens": submitted_tokens,
                "feedback_score": round(raw_task_score, 4),
                "sample_generations": sample_gens,
            })
            return GolfObservation(
                task_id=self._task.task_id,
                task_category=self._task.category,
                task_description=self._task.description,
                target_model_id=self._target.model_id,
                prompt_budget_tokens=budget,
                max_target_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
                num_test_examples=len(self._scoring_ex),
                train_examples=[
                    {"input": x, "expected": y} for (x, y) in self._train_ex
                ],
                scorer_name=self._task.scorer,
                baseline_zero_shot_score=round(self._baseline_zero_shot, 4),
                submitted_prompt_tokens=submitted_tokens,
                raw_task_score=round(raw_task_score, 4),  # on feedback slice
                sample_generations=sample_gens,
                done=False,
                reward=0.0,                                # no reward until terminal
                turn_number=self._turn_count + 1,           # next turn
                turn_limit=self._turn_limit,
                prior_attempts=list(self._prior_attempts),
                metadata={
                    "task_difficulty": self._task.difficulty,
                    "task_tags": list(self._task.tags),
                    "is_intermediate_feedback": True,
                },
            )

        # ----- Final (or single-turn): apply rubric, return terminal obs -----
        held_out_inputs = [x for x, _ in scoring_slice]
        result = self._rubric.grade(
            raw_task_score=raw_task_score,
            baseline_zero_shot_score=self._baseline_zero_shot,
            submitted_tokens=submitted_tokens,
            prompt_budget=budget,
            prompt_text=truncated_prompt,
            held_out_inputs=held_out_inputs,
        )
        details = grade_details_dict(result, task_id=self._task.task_id)

        return GolfObservation(
            task_id=self._task.task_id,
            task_category=self._task.category,
            task_description=self._task.description,
            target_model_id=self._target.model_id,
            prompt_budget_tokens=budget,
            max_target_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
            num_test_examples=len(scoring_slice),
            train_examples=[
                {"input": x, "expected": y} for (x, y) in self._train_ex
            ],
            scorer_name=self._task.scorer,
            baseline_zero_shot_score=round(self._baseline_zero_shot, 4),
            submitted_prompt_tokens=submitted_tokens,
            raw_task_score=round(result.raw_task_score, 4),
            length_factor=round(result.length_factor, 4),
            leakage_penalty=round(result.leakage_penalty, 4),
            gain_over_baseline=round(result.gain_over_baseline, 4),
            grade_details=details,
            sample_generations=sample_gens,
            done=True,
            reward=round(result.reward, 4),
            turn_number=self._turn_count,
            turn_limit=self._turn_limit,
            prior_attempts=list(self._prior_attempts),
            metadata={
                "task_difficulty": self._task.difficulty,
                "task_tags": list(self._task.tags),
            },
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _choose_task(self, task: Optional[str]) -> TaskSpec:
        if task is None or task == "random":
            task_id = self._rng.choice(list(_ALL_TASKS.keys()))
        elif task in _ALL_TASKS:
            task_id = task
        else:
            task_id = next(iter(_ALL_TASKS.keys()))
        return _ALL_TASKS[task_id]

    def _score_prompt(
        self,
        prompt: str,
        return_samples: bool = False,
        examples: Optional[List[tuple[str, str]]] = None,
    ) -> float | tuple[float, list]:
        """Run target on test inputs with `prompt`, score each output,
        return mean score. Optionally also return up to 2 sample triples
        for debugging.

        `examples` overrides the default `self._test_ex` slice — used by
        multi-turn step() to score against the feedback or scoring slice
        rather than the full pool.
        """
        assert self._task is not None
        ex_pool = examples if examples is not None else self._test_ex
        test_inputs = [x for x, _ in ex_pool]
        test_expected = [y for _, y in ex_pool]

        generations: List[TargetGeneration] = self._target.generate_batch(
            prompt=prompt,
            test_inputs=test_inputs,
            max_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
        )

        per_item = []
        for gen, expected in zip(generations, test_expected):
            per_item.append(score_one(
                self._task.scorer,
                gen.output_text,
                expected,
                task_description=self._task.description,
            ))
        mean_score = sum(per_item) / max(1, len(per_item))

        if return_samples:
            samples = []
            for gen, expected in list(zip(generations, test_expected))[:2]:
                samples.append(
                    {
                        "input": gen.input_text,
                        "target_output": gen.output_text,
                        "expected": expected,
                    }
                )
            return mean_score, samples
        return mean_score
