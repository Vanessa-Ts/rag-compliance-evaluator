# Base
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get full-upgrade -y && \
    apt-get install -y --no-install-recommends locales tzdata curl ca-certificates make git && \
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


ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/venv \
    PYTHONPATH="/app/src"


# Deps
# ============================================================
FROM base AS deps

COPY pyproject.toml uv.lock ./

# Stub src so uv sync doesn't complain about missing project
RUN mkdir -p src/app && \
    touch src/app/__init__.py

RUN uv venv -p /usr/local/bin/python3 /venv/ && \
    uv sync --frozen --no-dev --no-install-project && \
    uv cache clean


# Builder (prod venv ready)
# ============================================================
FROM base AS builder

COPY --from=deps /venv /venv
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install the project itself (source only, deps already in venv)
RUN uv sync --frozen --no-dev --no-cache


# Dev
# ============================================================
FROM base AS development

COPY --from=deps /venv /venv
COPY pyproject.toml uv.lock ./

# Stub for the extra group install
RUN mkdir -p src/app && \
    touch src/app/__init__.pyn

# Layer on dev deps (pytest, ruff, mypy, etc. — torch already in venv)
RUN uv sync --frozen --no-install-project --group dev && \
    uv cache clean

COPY src/ ./src/

USER appuser
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/appuser/.local/bin:$PATH"

CMD ["sleep", "infinity"]


# Test
# ============================================================

FROM base AS test

COPY --from=deps /venv /venv
COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=0

RUN mkdir -p src/app && \
    touch src/app/__init__.py

RUN uv sync --frozen --no-cache --group test && \
    uv cache clean

COPY src/ ./src/
COPY tests/ ./tests/

USER appuser

CMD ["pytest", "tests/", \
     "-v", \
     "--cov=app", \
     "--cov-report=term-missing", \
     "--cov-report=xml:coverage.xml", \
     "--junitxml=report.xml"]


# Prod — uncomment when ready to deploy
# ============================================================
# FROM base AS production
#
# COPY --from=builder /venv /venv
# COPY src/ ./src/
#
# USER appuser
#
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1
#
# EXPOSE 8080
#
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]