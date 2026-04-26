# HF Spaces Dockerfile for Prompt Golf.
# Serves the OpenEnv HTTP contract on port 8000.
# Defaults to the mock target backend so the free-tier CPU Space boots fast
# without downloading any LLM weights. Set PROMPT_GOLF_TARGET_BACKEND=hf
# (and PROMPT_GOLF_TARGET_MODEL=...) at runtime on bigger hardware.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Install the package; the full transformers/torch deps from pyproject.toml
# are skipped here because we default to mock. If you need the HF target
# backend on the Space, override CMD or bake a different image.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "openenv-core[core]>=0.2.2" uvicorn fastapi pydantic gradio && \
    pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    PROMPT_GOLF_TARGET_BACKEND=mock \
    ENABLE_WEB_INTERFACE=true \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
