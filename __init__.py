# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt Golf Environment.

An environment where the agent's action is a *prompt*. A frozen target LLM
receives that prompt on held-out test inputs, and the agent is rewarded for
getting the target to produce correct outputs using as few tokens as possible.
"""

from .client import PromptGolfEnv
from .models import GolfAction, GolfObservation

__all__ = [
    "GolfAction",
    "GolfObservation",
    "PromptGolfEnv",
]
