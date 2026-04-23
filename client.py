# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt Golf Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DEFAULT_PROMPT_BUDGET,
    MAX_TARGET_OUTPUT_TOKENS,
    TEST_EXAMPLES_PER_EPISODE,
    GolfAction,
    GolfObservation,
)


class PromptGolfEnv(EnvClient[GolfAction, GolfObservation, State]):
    """
    Client for the Prompt Golf Environment.

    Use reset(task="task_name") to select a task (see TASK_NAMES) or
    task="random" to sample one. The observation includes a natural-language
    task description and a few visible train examples. Submit a prompt via
    step(GolfAction(prompt=...)) — the episode terminates on that step and
    the observation's `reward` carries the final reward.

    Example:
        >>> async with PromptGolfEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="sentiment_basic")
        ...     obs = result.observation
        ...     my_prompt = f"{obs.task_description}\\nAnswer with one word."
        ...     result = await env.step(GolfAction(prompt=my_prompt))
        ...     print(f"Reward: {result.reward:.3f}  |  "
        ...           f"Score: {result.observation.raw_task_score:.2f}  |  "
        ...           f"Tokens: {result.observation.submitted_prompt_tokens}")
    """

    def _step_payload(self, action: GolfAction) -> Dict:
        return {"prompt": action.prompt}

    def _parse_result(self, payload: Dict) -> StepResult[GolfObservation]:
        obs_data = payload.get("observation", {})
        observation = GolfObservation(
            task_id=obs_data.get("task_id", ""),
            task_category=obs_data.get("task_category", ""),
            task_description=obs_data.get("task_description", ""),
            target_model_id=obs_data.get("target_model_id", ""),
            prompt_budget_tokens=obs_data.get("prompt_budget_tokens", DEFAULT_PROMPT_BUDGET),
            max_target_output_tokens=obs_data.get(
                "max_target_output_tokens", MAX_TARGET_OUTPUT_TOKENS
            ),
            num_test_examples=obs_data.get("num_test_examples", TEST_EXAMPLES_PER_EPISODE),
            train_examples=obs_data.get("train_examples", []),
            scorer_name=obs_data.get("scorer_name", ""),
            baseline_zero_shot_score=obs_data.get("baseline_zero_shot_score", 0.0),
            submitted_prompt_tokens=obs_data.get("submitted_prompt_tokens"),
            raw_task_score=obs_data.get("raw_task_score"),
            length_factor=obs_data.get("length_factor"),
            leakage_penalty=obs_data.get("leakage_penalty"),
            gain_over_baseline=obs_data.get("gain_over_baseline"),
            grade_details=obs_data.get("grade_details"),
            sample_generations=obs_data.get("sample_generations"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
