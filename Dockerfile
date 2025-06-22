FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock src /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev \
    ffmpeg libsm6 libxext6 \
    poppler-utils incron && \
    apt-get clean && \
    rm -rf "/var/lib/apt/lists/*"

RUN useradd -ms /bin/bash rag
USER rag
WORKDIR /app

COPY --from=builder --chown=rag:rag /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["ollama-rag", "--vector-store-path=/home/rag/.knowledge_db"]


