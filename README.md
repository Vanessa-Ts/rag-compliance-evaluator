# rag-compliance-evaluator

A FastAPI RAG service for multi-jurisdiction employment-law Q&A, built with a focus on production LLM practices: SSE streaming, Anthropic prompt caching, tool-use structured evaluation, and RAGAS-style context-relevance scoring.

---

## Stack

| Component | Purpose |
|---|---|
| FastAPI + Jinja2 | REST API + server-side UI |
| ChromaDB | Persistent vector store |
| sentence-transformers | Offline embedding (`all-MiniLM-L6-v2`) |
| LangChain | LLM abstraction (Ollama / OpenAI) |
| Anthropic SDK | Direct SDK for prompt caching + tool-use eval |
| uv | Dependency management |
| Docker + Compose | Multi-stage build, dev container |
| Pydantic Settings | Environment-aware config |
| pytest + mypy | Tests + static types |

---

## Quick Start

```bash
# 1. Start dev container (source bind-mounted — edits take effect immediately)
docker compose up --build

# 2. Inside container: create env file and sync deps
cp .env.example .env
uv sync

# 3. Start the service (hot-reload on port 8080)
python src/app/main.py

# 4. Pull the local LLM (Ollama, persisted in a Docker volume)
docker compose exec ollama ollama pull llama3.2:3b
```

Open in VS Code with **Reopen in Container** and use the launch configs in `.vscode/launch.json` — "Run: uvicorn (reload)" for dev, "Debug: uvicorn (no reload)" for breakpoints.

---

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Env var | Default model | Notes |
|---|---|---|---|
| `ollama` (default) | `OLLAMA_BASE_URL` | `llama3.2:3b` | Free, local |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | Prompt caching enabled by default |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | Via LangChain |

Override the model with `GENERATION_MODEL`. Disable prompt caching with `ENABLE_PROMPT_CACHING=false`.

---

## API

### RAG

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Chunk and embed `data/corpus/*.md` into ChromaDB |
| `POST` | `/query` | Answer a question; returns JSON with answer, citations, latency |
| `POST` | `/query/stream` | Same but streams tokens via SSE, then emits a `done` event |

**SSE format for `/query/stream`:**
```
event: token
data: {"token": "California"}

event: done
data: {"answer": "...", "citations": [...], "latency_ms": 87.3, "provider": "anthropic", "model": "claude-sonnet-4-6"}
```

### Evaluation

| Method | Path | Description |
|---|---|---|
| `POST` | `/evaluate` | Start a background eval job (202 + `job_id`) |
| `GET` | `/evaluate/stream/{job_id}` | SSE stream of per-item progress |
| `GET` | `/evaluate/last` | Most recent completed eval result |

---

## Evaluation Metrics

Each golden item is scored on:

| Metric | Description |
|---|---|
| `precision_at_k` | Fraction of top-k retrieved docs in the expected set |
| `hit_at_k` | Boolean: at least one expected doc in top-k |
| `faithfulness_score` | LLM judge: is the answer grounded in the context? |
| `faithfulness_reasoning` | Judge's reasoning string (Anthropic tool-use path only) |
| `context_relevance_score` | RAGAS-style: how relevant are the retrieved passages to the question? |
| `latency_ms` | End-to-end query latency |

Summary fields: `retrieval_precision_at_k`, `hit_rate_at_k`, `mean_faithfulness`, `mean_context_relevance`, `mean_latency_ms`, `p95_latency_ms`.

When `LLM_PROVIDER=anthropic`, judges use Anthropic tool use for structured output (no JSON parsing). Other providers use a JSON-prompt fallback.

---

## Prompt Caching (Anthropic)

When `LLM_PROVIDER=anthropic` (and `ENABLE_PROMPT_CACHING=true`, the default), the service uses the Anthropic SDK directly to attach `cache_control: {"type": "ephemeral"}` to the system prompt and retrieved context block. Cache hits reduce input-token cost on repeated or similar queries and appear as `cache_read_input_tokens` in the Anthropic dashboard.

---

## Corpus

`data/corpus/` holds 7 markdown employment-law documents; see `data/corpus/README.md` for the `doc_id` registry. Golden evaluation pairs live in `data/golden/qa.yaml`. On startup the lifespan hook warms the embedding model and auto-ingests the corpus if ChromaDB is empty.

---

## Development Commands

```bash
# Run all tests with coverage
source .venv/bin/activate && pytest

# Type check
mypy src/

# Lint / format
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

---

## Docker Stages

| Stage | Purpose |
|---|---|
| `base` | Python 3.11-slim + uv + venv skeleton |
| `builder` | Production deps only |
| `development` | Full dev tooling, source bind-mounted, `sleep infinity` |
| `test` | Runs pytest with coverage on container start |

CI (`.github/workflows/ci.yml`) builds the `test` stage and collects `coverage.xml` and `report.xml`.

### Scaffolded (commented out, ready to enable)

- `production` stage in `Dockerfile` — minimal image via `COPY --from=builder`
- `app-prod` service in `docker-compose.yml` — resource limits, healthcheck, restart policy
- `postgres` service — persistent volume and healthcheck
- Docker networking block
