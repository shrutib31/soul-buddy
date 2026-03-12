# SoulBuddy Backend

FastAPI backend for the SoulBuddy application. Handles chat, conversation history, user context, and LLM response generation via a LangGraph pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Setup](#local-setup)
4. [Docker](#docker)
5. [Database Initialisation](#database-initialisation)
6. [Configuration Reference](#configuration-reference)
7. [Running the Server](#running-the-server)
8. [API Endpoints](#api-endpoints)
9. [Project Structure](#project-structure)
10. [Adding New APIs](#adding-new-apis)
11. [Testing](#testing)
12. [Code Quality](#code-quality)
13. [Logging](#logging)

---

## Architecture Overview

```
HTTP Request
  → server.py          (FastAPI lifespan, middleware, router registration)
  → api/chat.py        (endpoint handler, Pydantic validation, auth)
  → LangGraph pipeline (graph/graph_builder.py)
       conv_id_handler
         → load_user_context
             → store_message  (parallel, fire-and-forget)
             → classification_node → response_generator → store_bot_response → render
  → HTTP Response
```

### Key Subsystems

| Subsystem | Location | Notes |
|-----------|----------|-------|
| FastAPI app & startup | `server.py` | Lifespan manages all connections |
| LangGraph pipeline | `graph/graph_builder.py` | Compiled once at startup |
| Graph state | `graph/state.py` | `ConversationState` Pydantic model |
| Classification | `graph/nodes/agentic_nodes/classification_node.py` | Rule-based (no ML model) — greeting / crisis / out-of-scope / intent / situation / severity |
| Response templates | `graph/nodes/agentic_nodes/response_templates.py` | Pre-built responses for greeting, crisis, out-of-scope — bypasses LLM |
| ORM models (data DB) | `orm/models.py` | Conversations, turns, summaries |
| ORM models (auth DB) | `orm/auth_models.py` | Users, personality profiles |
| DB sessions | `config/sqlalchemy_db.py` | `SQLAlchemyDataDB` / `SQLAlchemyAuthDB` singletons |
| Redis cache | `services/cache_service.py` | Cache-aside, fail-silent |
| Encryption | `services/key_manager.py` | AES-256-GCM via GCP KMS (opt-in) |
| Settings | `config/settings.py` | Single `settings` object — import everywhere |

### Classification Pipeline

The classification node is **fully rule-based** (regex + keyword patterns). No ML model or GPU required.

| Check | Output | Routed to |
|-------|--------|-----------|
| Greeting detected | `is_greeting = True` | Greeting template (no LLM call) |
| Out-of-scope request | `is_out_of_scope = True` | Out-of-scope template (no LLM call) |
| Crisis detected | `is_crisis_detected = True` | Crisis template + hotline numbers (no LLM call) |
| Everything else | `intent` / `situation` / `severity` | LLM response generator |

**Out-of-scope detection** flags explicit bot requests for off-domain tasks (cooking, coding, legal/financial advice, travel, entertainment). Personal narratives that *mention* off-domain topics ("I was coding all night and I'm stressed") are intentionally **not** flagged — they're valid wellness context.

### Two Databases

- **Data DB** (`soulbuddy`) — conversations, conversation turns, summaries, domain config
- **Auth DB** (`souloxy-db`) — user identity, personality profiles, detailed user profiles

---

## Prerequisites

- Python 3.11+
- PostgreSQL (two databases — see [Database Initialisation](#database-initialisation))
- Redis (optional — server degrades gracefully without it)
- Docker + Docker Compose (for containerised deployments)
- [UV](https://github.com/astral-sh/uv) (recommended) or pip

---

## Local Setup

### Option 1: Automated (recommended)

```bash
chmod +x setup.sh
./setup.sh
```

The script detects UV and uses it if available, otherwise falls back to pip.

### Option 2: UV (manual)

```bash
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
mkdir -p logs
```

### Option 3: pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs
```

### Installing UV (optional but recommended)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS with Homebrew
brew install uv
```

---

## Docker

Two compose files live at the **repo root** (one level above `sb-backend/`):

| File | Purpose |
|------|---------|
| `docker-compose-sb-backend.yml` | Local development — bind-mounts source for hot-reload, Redis exposed on host |
| `docker-compose-sb-backend-staging.yml` | Staging — baked image, Redis internal only, memory limit enforced |

### Local development

```bash
# From repo root
docker compose -f docker-compose-sb-backend.yml up --build

# Stop
docker compose -f docker-compose-sb-backend.yml down
```

Uses env file: `sb-backend/.env.docker.local`

### Staging

```bash
# From repo root
docker compose -f docker-compose-sb-backend-staging.yml up --build -d

# Stop
docker compose -f docker-compose-sb-backend-staging.yml down
```

Uses env file: `sb-backend/.env.docker` (default) or override with `ENV_FILE=./sb-backend/.env.docker.local.staging docker compose …`

### Memory limit

The staging compose enforces a container memory cap (default 512 MB) via `mem_limit`. Override in your env file:

```
BACKEND_MEM_LIMIT=768m
BACKEND_MEMSWAP_LIMIT=768m
```

### Startup order

Both compose files use `depends_on: condition: service_healthy` — the backend waits for Redis to pass its `redis-cli ping` healthcheck before starting.

### CI / Image publishing

Every push to `main` that touches `sb-backend/**` triggers `.github/workflows/docker-publish.yml`, which builds and pushes to GHCR:

```
ghcr.io/<owner>/<repo>/sb-backend:latest
ghcr.io/<owner>/<repo>/sb-backend:main
ghcr.io/<owner>/<repo>/sb-backend:<short-sha>
```

---

## Database Initialisation

### 1. Create the databases

```bash
psql -U postgres
CREATE DATABASE soulbuddy;
CREATE DATABASE "souloxy-db";
\q
```

### 2. Initialise schema and seed config data

```bash
python scripts/init_db.py
```

Creates all tables and seeds required config rows. Runs automatically inside Docker via `entrypoint.sh` (non-fatal if it fails).

### 3. Drop all tables (reset)

```bash
python scripts/cleanup_db.py
```

---

## Configuration Reference

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|----------|-------------|
| `DATA_DB_URL` | Primary data DB — `postgresql+asyncpg://user:pass@host:port/soulbuddy` |
| `AUTH_DB_URL` | Auth DB — `postgresql+asyncpg://user:pass@host:port/souloxy-db` |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |

### LLM Provider (at least one required)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_FLAG` | `false` | Set `true` to use OpenAI |
| `OPENAI_API_KEY` | — | Required when `OPENAI_FLAG=true` |
| `OLLAMA_FLAG` | `false` | Set `true` to use Ollama |
| `OLLAMA_BASE_URL` | `http://72.60.99.35:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name |
| `COMPARE_RESULTS` | `false` | Call both providers and pick the best response |

### Redis Cache (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_MAX_CONNECTIONS` | `20` | Connection pool size |
| `REDIS_TTL_PROFILE` | `7200` | Profile cache TTL in seconds (2 h) |
| `REDIS_TTL_CONFIG` | `86400` | Config cache TTL in seconds (24 h) |
| `REDIS_TTL_CONVERSATION` | `1800` | Conversation history TTL in seconds (30 min) |

Redis is non-fatal — if unreachable, all reads fall back to the database transparently.

### Docker Resource Limits (staging compose only)

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_MEM_LIMIT` | `512m` | Container memory cap |
| `BACKEND_MEMSWAP_LIMIT` | `512m` | Swap cap — set equal to `BACKEND_MEM_LIMIT` to disable swap |

### Encryption (opt-in, disabled by default)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCRYPTION_ENABLED` | `false` | Set `true` to enable AES-256-GCM message encryption |
| `GCP_PROJECT_ID` | — | GCP project ID |
| `GCP_KMS_LOCATION` | — | KMS key location (e.g. `global`) |
| `GCP_KMS_KEYRING` | — | KMS keyring name |
| `GCP_KMS_KEY` | — | KMS key name |
| `SERVICE_ACCOUNT_KEY` | `./config/serviceAccountKey.json` | Path to GCP service account JSON |

When `ENCRYPTION_ENABLED=false` (the default), no GCP credentials are needed and messages are stored as plaintext. When enabled, every stored message is AES-256-GCM encrypted using a per-conversation key derived from a GCP KMS master key, and is transparently decrypted on retrieval.

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `LOG_DIR` | `logs` | Directory for log files |
| `LOGGING_CONFIG_PATH` | `logging.yaml` | Path to logging YAML config |

---

## Running the Server

```bash
source venv/bin/activate

# Development — with auto-reload
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Production (matches entrypoint.sh used in Docker)
uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --limit-concurrency 20 \
    --limit-max-requests 500 \
    --no-access-log
```

> **Why `--workers 1`?** The app is memory-constrained (512 MB target). Multiple workers multiply RAM usage. A single async worker with `--limit-concurrency 20` handles concurrent I/O workloads efficiently without the overhead.

**Startup sequence** (steps 1–4 are fatal; 5–6 are non-fatal):

1. Logging
2. Data DB — asyncpg pool
3. Auth DB — asyncpg pool
4. SQLAlchemy engines
5. Redis (fail-silent, retries reconnect in background)
6. Supabase

---

## API Endpoints

### Health

```
GET /health
```

Returns status of all connected services.

### Chat

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/chat` | Optional | Chat (incognito or cognito) |
| `POST` | `/api/v1/chat/stream` | Optional | Streaming chat (SSE) |

**Incognito** (`is_incognito: true`, default): no `Authorization` header needed; conversation is not persisted.

**Cognito** (`is_incognito: false`): requires `Authorization: Bearer <supabase-token>`; conversation is persisted and encrypted if `ENCRYPTION_ENABLED=true`.

Request body:
```json
{
  "message": "I need support",
  "is_incognito": true,
  "domain": "student",
  "sb_conv_id": "optional-existing-uuid"
}
```

### Conversation History

All history endpoints require `Authorization: Bearer <supabase-token>`. Encrypted messages are transparently decrypted before returning.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/chat/conversations/messages` | All conversations + messages for the authenticated user |
| `GET` | `/api/v1/chat/conversations/{conversation_id}/messages` | All messages for a specific conversation |

### Classification

```
POST /api/v1/classify
```

Classifies a message without going through the full chat pipeline.

---

## Project Structure

```
sb-backend/
├── server.py                    # FastAPI app, lifespan, router registration
├── entrypoint.sh                # Docker entrypoint: init_db → start server
│
├── api/                         # HTTP layer — thin handlers, auth, validation
│   ├── chat.py                  # Chat + conversation history endpoints
│   ├── classify.py              # Classification endpoint
│   └── supabase_auth.py         # Auth dependencies (optional + required token)
│
├── graph/                       # LangGraph pipeline
│   ├── graph_builder.py         # Graph definition and compilation
│   ├── state.py                 # ConversationState Pydantic model
│   ├── streaming.py             # SSE streaming helper
│   └── nodes/
│       ├── function_nodes/      # Deterministic nodes (DB I/O, rendering)
│       │   ├── conv_id_handler.py
│       │   ├── load_user_context.py
│       │   ├── store_message.py       # Encrypts before storing (if enabled)
│       │   ├── store_bot_response.py  # Encrypts before storing (if enabled)
│       │   ├── get_messages.py        # Decrypts on retrieval
│       │   └── render.py
│       └── agentic_nodes/       # Classification + LLM response nodes
│           ├── classification_node.py   # Rule-based: greeting/crisis/out-of-scope/intent/situation/severity
│           ├── response_templates.py    # Pre-built responses (no LLM) for greeting/crisis/out-of-scope
│           ├── response_generator.py    # LLM call (Ollama / OpenAI)
│           └── response_evaluator.py   # Picks best response when COMPARE_RESULTS=true
│
├── orm/                         # SQLAlchemy ORM models
│   ├── models.py                # Data DB models (conversations, turns, summaries)
│   ├── auth_models.py           # Auth DB models (users, profiles)
│   └── encryption_audit_log.py  # Encryption audit log model
│
├── services/                    # Shared business logic
│   ├── cache_service.py         # Redis cache-aside singleton (fail-silent)
│   └── key_manager.py           # AES-256-GCM encryption via GCP KMS
│
├── config/                      # Configuration singletons
│   ├── settings.py              # Central settings object — import this everywhere
│   ├── sqlalchemy_db.py         # SQLAlchemy engine singletons
│   ├── database.py              # asyncpg pool (data DB)
│   ├── auth_database.py         # asyncpg pool (auth DB)
│   ├── redis.py                 # Redis connection + reconnect loop
│   └── supabase.py              # Supabase client
│
├── scripts/                     # Admin scripts
│   ├── init_db.py               # Create tables + seed data
│   └── cleanup_db.py            # Drop all tables
│
├── tests/                       # Unit tests — 300 tests, no live DB/LLM/GCP needed
│   ├── api/
│   ├── config/
│   ├── graph/
│   └── services/
│
├── conftest.py                  # Root pytest config (stubs google.cloud.kms in CI)
├── .env.example                 # Template — copy to .env
├── requirements.txt             # Production dependencies
└── pyproject.toml               # Project metadata + dev dependencies
```

---

## Adding New APIs

1. Create `api/<your_module>.py` with a FastAPI router:

```python
from fastapi import APIRouter
router = APIRouter(prefix="/your-prefix")

@router.get("/something")
async def get_something():
    return {"hello": "world"}
```

2. Register it in `server.py`:

```python
from api.your_module import router as your_router
app.include_router(your_router, prefix="/api/v1", tags=["YourTag"])
```

---

## Testing

Tests run without any live DB, Redis, LLM, or GCP connection — all external dependencies are mocked.

```bash
# Run all unit tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/api/test_chat.py -v

# Exclude integration tests (default in CI)
pytest -m "not integration"
```

CI runs on every push via GitHub Actions (`.github/workflows/backend-tests.yml`).

---

## Code Quality

```bash
# Format with Black
black .

# Lint with Ruff
ruff check .

# Type check with MyPy
mypy .
```

---

## Logging

Logs go to both the console and timestamped files under `logs/` (e.g. `soulbuddy_2026-03-08_10-30-00.log`).

| Variable | Example |
|----------|---------|
| `LOG_LEVEL` | `debug` / `info` / `warning` / `error` |
| `LOG_DIR` | `logs` |
| `LOGGING_CONFIG_PATH` | `config-files/logging.yaml` |

---

## License

MIT
