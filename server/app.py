# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Prompt Golf Environment.

Endpoints:
    - POST /reset: Reset the environment (optionally with task name)
    - POST /step: Submit a prompt, get the reward back
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uv run --project . server
    python -m prompt_golf_env.server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from prompt_golf_env.models import GolfAction, GolfObservation
    from prompt_golf_env.server.prompt_golf_environment import PromptGolfEnvironment
except ImportError:
    try:
        from ..models import GolfAction, GolfObservation
        from .prompt_golf_environment import PromptGolfEnvironment
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import GolfAction, GolfObservation
        from server.prompt_golf_environment import PromptGolfEnvironment


app = create_app(
    PromptGolfEnvironment,
    GolfAction,
    GolfObservation,
    env_name="prompt_golf_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
