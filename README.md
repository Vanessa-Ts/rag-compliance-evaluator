# docker-dev-template

A containerized Python development template for building web services with FastAPI, Docker Compose multi-stage builds, and `uv` for dependency management.

Designed as a reusable starting point - production and infrastructure blocks are scaffolded as commented sections, ready to uncomment when needed.


## Stack

| Tool | Purpose |
|---|---|
| FastAPI + Jinja2 | API framework + server-side UI |
| uv | Dependency management |
| Docker + Compose | Multi-stage build, local dev |
| Pydantic Settings | Environment-aware config |
| Black + isort + ruff | Formatting and linting |
| mypy | Static type checking |
| pytest | Testing |



## Quick Start

#### 1. Start dev container
docker compose up --build

#### 2. Inside the container: create env file and sync deps
cp .env.example .env
uv sync

#### 3. Start the web service via launch.json or manually
python src/app/main.py

Open the repo in VS Code and use "Reopen in Container" - the .devcontainer/devcontainer.json handles the rest. Start the web service via the provided launch.json debug configuration.



## Docker Stages

| Stage | Purpose | Used by |
|---|---|---|
| `base` | OS, python, uv, venv skeleton — no source, no deps | All stages |
| `builder` | Installs production deps into `/venv` | `production` (via COPY --from) |
| `development` | Full dev tooling, source mounted at runtime | `docker compose up` |
| `test` | Runs pytest with coverage | CI pipeline |
| `production` | Minimal image, pre-built venv | Uncomment when ready to deploy |



## Scaffolded (Commented)

Ready to enable when your project needs them:

- **Production stage** in `Dockerfile` - uses `COPY --from=builder` for minimal image
- **`app-prod` service** in `docker-compose.yml` - with resource limits and healthchecks
- **Postgres service** with healthcheck and persistent volume
- **Docker networking** for multi-service communication
