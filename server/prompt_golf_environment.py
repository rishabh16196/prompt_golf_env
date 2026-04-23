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
        MAX_TARGET_OUTPUT_TOKENS,
        TEST_EXAMPLES_PER_EPISODE,
        TRAIN_EXAMPLES_VISIBLE,
        GolfAction,
        GolfObservation,
    )
    from .rubrics import PromptGolfRubric, grade_details_dict
    from .scorer import score_one
    from .target_model import TargetBackend, TargetGeneration, get_target_backend
    from .tasks import TASKS, TaskSpec, get_task, list_task_ids
except ImportError:
    from models import (
        DEFAULT_PROMPT_BUDGET,
        MAX_TARGET_OUTPUT_TOKENS,
        TEST_EXAMPLES_PER_EPISODE,
        TRAIN_EXAMPLES_VISIBLE,
        GolfAction,
        GolfObservation,
    )
    from server.rubrics import PromptGolfRubric, grade_details_dict
    from server.scorer import score_one
    from server.target_model import TargetBackend, TargetGeneration, get_target_backend
    from server.tasks import TASKS, TaskSpec, get_task, list_task_ids


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
        # Cached per-episode baseline (target with empty prompt)
        self._baseline_zero_shot: float = 0.0

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
    ) -> GolfObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._task = self._choose_task(task)

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
            num_test_examples=len(self._test_ex),
            train_examples=[
                {"input": x, "expected": y} for (x, y) in self._train_ex
            ],
            scorer_name=self._task.scorer,
            baseline_zero_shot_score=round(self._baseline_zero_shot, 4),
            done=False,
            reward=0.0,
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

        # Truncate prompt to the task's budget (in target tokens).
        budget = self._task.budget_tokens or DEFAULT_PROMPT_BUDGET
        truncated_prompt = self._target.truncate_to_tokens(action.prompt, budget)
        submitted_tokens = self._target.count_prompt_tokens(truncated_prompt)

        # Score the prompt.
        raw_task_score, sample_gens = self._score_prompt(
            truncated_prompt, return_samples=True
        )

        # Apply rubric.
        held_out_inputs = [x for x, _ in self._test_ex]
        result = self._rubric.grade(
            raw_task_score=raw_task_score,
            baseline_zero_shot_score=self._baseline_zero_shot,
            submitted_tokens=submitted_tokens,
            prompt_budget=budget,
            prompt_text=truncated_prompt,
            held_out_inputs=held_out_inputs,
        )
        details = grade_details_dict(result, task_id=self._task.task_id)

        # Build terminal observation. We re-emit the task framing so the
        # agent/trainer has a self-contained record of the episode.
        return GolfObservation(
            task_id=self._task.task_id,
            task_category=self._task.category,
            task_description=self._task.description,
            target_model_id=self._target.model_id,
            prompt_budget_tokens=budget,
            max_target_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
            num_test_examples=len(self._test_ex),
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
            task_id = self._rng.choice(list_task_ids())
        elif task in TASKS:
            task_id = task
        else:
            # Defensive fallback: unknown task_id becomes the first task.
            task_id = list_task_ids()[0]
        return get_task(task_id)

    def _score_prompt(
        self, prompt: str, return_samples: bool = False
    ) -> float | tuple[float, list]:
        """Run target on test inputs with `prompt`, score each output,
        return mean score. Optionally also return up to 2 sample triples
        for debugging.
        """
        assert self._task is not None
        test_inputs = [x for x, _ in self._test_ex]
        test_expected = [y for _, y in self._test_ex]

        generations: List[TargetGeneration] = self._target.generate_batch(
            prompt=prompt,
            test_inputs=test_inputs,
            max_output_tokens=MAX_TARGET_OUTPUT_TOKENS,
        )

        per_item = []
        for gen, expected in zip(generations, test_expected):
            per_item.append(score_one(self._task.scorer, gen.output_text, expected))
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
