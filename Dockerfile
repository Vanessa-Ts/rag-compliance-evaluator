# Base
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get full-upgrade -y && \
    apt-get install -y --no-install-recommends locales tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app


ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/venv \
    PYTHONPATH="/app/src"

RUN uv venv -p /usr/local/bin/python3 /venv/ && \
    chown -R appuser:appuser /venv

# Builder
FROM base AS builder

COPY src/ ./src/

RUN uv sync --frozen --no-dev --no-cache


# ============================================================
# Dev
# ============================================================
FROM base AS development

COPY src/ ./src/

RUN uv sync --frozen --no-cache --group dev && \
    uv cache clean

USER appuser

CMD ["sleep", "infinity"]


# Test
FROM base AS test

COPY src/ ./src/
COPY tests/ ./tests/

RUN uv sync --frozen --no-cache --group test && \
    uv cache clean

USER appuser

CMD ["pytest", "tests/", \
     "-v", \
     "--cov=app", \
     "--cov-report=term-missing", \
     "--cov-report=xml:coverage.xml", \
     "--junitxml=report.xml"]


# Prod - uncomment when ready to deploy
# FROM base AS production
#
# # Copy pre-built venv from builder (deps cached separately from source)
# COPY --from=builder /venv /venv
# COPY src/ ./src/
#
# RUN uv cache clean
#
# USER appuser
#
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1


# EXPOSE 8000
#
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]