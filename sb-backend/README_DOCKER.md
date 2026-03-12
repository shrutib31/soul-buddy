# SoulBuddy Backend — Docker Guide

This guide covers building and running the SoulBuddy backend with Docker for both local development and staging.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Files](#environment-files)
3. [Local Development](#local-development)
4. [Staging](#staging)
5. [Enabling Encryption](#enabling-encryption)
6. [Useful Commands](#useful-commands)
7. [Troubleshooting](#troubleshooting)

> **CLI override variables** are documented inside [Environment Files](#cli-override-variables).

---

## Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) (includes Docker Compose)
- A copy of the appropriate env file (see [Environment Files](#environment-files))

---

## Environment Files

Never commit real credentials. The compose files load env vars from a file on your host:

| Compose file | Default env file | Typical use |
|---|---|---|
| `docker-compose-sb-backend.yml` | `sb-backend/.env.docker.local` | Local dev |
| `docker-compose-sb-backend-staging.yml` | `sb-backend/.env.docker` | Staging |
| `docker-compose-sb-backend-staging.yml` | `sb-backend/.env.docker.local.staging` | Staging with local overrides — pass via `ENV_FILE` |

Create your file from the template:

```bash
cp sb-backend/.env.example sb-backend/.env.docker.local   # dev
cp sb-backend/.env.example sb-backend/.env.docker          # staging
```

Then fill in real values. Minimum required:

```env
DATA_DB_URL=postgresql+asyncpg://user:pass@host.docker.internal:5432/soulbuddy
AUTH_DB_URL=postgresql+asyncpg://user:pass@host.docker.internal:5432/souloxy-db
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
OPENAI_FLAG=true
OPENAI_API_KEY=sk-...
# Dev compose (docker-compose-sb-backend.yml):
REDIS_URL=redis://soulbuddy-redis:6379
# Staging compose (docker-compose-sb-backend-staging.yml):
# REDIS_URL=redis://redis:6379
```

> **DB host inside Docker**: use `host.docker.internal` to reach PostgreSQL running on your laptop.
> **Redis inside Docker**: use the compose service name — `soulbuddy-redis` for dev, `redis` for staging. Never use `localhost`.

### CLI override variables

These can be set in your shell before any `docker compose` command to override defaults without editing the env file:

| Variable | Default | Description |
|---|---|---|
| `ENV_FILE` | `./sb-backend/.env.docker.local` | Path to the env file loaded into `sb-backend` |
| `PORT` | `8000` | Port uvicorn binds to **inside** the container |
| `BACKEND_PORT` | `8000` | Host port mapped to the container's `PORT` |
| `BACKEND_MEM_LIMIT` | `512m` | Container memory cap (staging compose only) |
| `BACKEND_MEMSWAP_LIMIT` | `512m` | Swap cap — set equal to `BACKEND_MEM_LIMIT` to disable swap (staging only) |

```bash
# Use a different env file
ENV_FILE=./sb-backend/.env.staging docker compose -f docker-compose-sb-backend.yml up

# Run backend on a different host port (host 9000 → container 8000)
BACKEND_PORT=9000 docker compose -f docker-compose-sb-backend.yml up --build

# Change both internal and host port
PORT=8080 BACKEND_PORT=8080 docker compose -f docker-compose-sb-backend.yml up --build

# Combine multiple overrides, run detached
ENV_FILE=./sb-backend/.env.staging BACKEND_PORT=9000 PORT=9000 \
  docker compose -f docker-compose-sb-backend.yml up --build -d
```

---

## Local Development

The dev compose file (`docker-compose-sb-backend.yml`) starts:
- `sb-backend` — the FastAPI server with a bind-mount for hot-reload
- `soulbuddy-redis` — Redis 7 (port 6379 exposed to host for local inspection)

### Start all services

```bash
# Foreground (logs printed to terminal)
docker compose -f docker-compose-sb-backend.yml up --build

# Detached / background
docker compose -f docker-compose-sb-backend.yml up --build -d
```

### Start only one service

```bash
docker compose -f docker-compose-sb-backend.yml up sb-backend
docker compose -f docker-compose-sb-backend.yml up soulbuddy-redis
```

### Restart a service (picks up env changes, no rebuild)

```bash
docker compose -f docker-compose-sb-backend.yml restart sb-backend
docker compose -f docker-compose-sb-backend.yml restart soulbuddy-redis
```

### Stop (keeps containers and volumes)

```bash
docker compose -f docker-compose-sb-backend.yml stop
```

### Stop and remove containers

```bash
docker compose -f docker-compose-sb-backend.yml down
```

### Startup order

Both compose files use `depends_on: condition: service_healthy` so the backend waits for Redis to pass its `redis-cli ping` healthcheck before starting. The server still degrades gracefully if Redis goes down after startup.

### How the bind-mount works

The entire `./sb-backend` directory is mounted to `/app` inside the container. Code changes on your host are instantly visible — no rebuild needed. The entrypoint runs `init_db.py` on each start, then launches the server.

---

## Staging

The staging compose file (`docker-compose-sb-backend-staging.yml`) starts:
- `sb-backend` — the FastAPI server using the **built image** (no bind-mount)
- `redis` — Redis 7 (not exposed to the host, internal network only)

**Key differences from dev:**
- No bind-mount — code is baked into the image at build time
- Memory limit enforced: `mem_limit` and `memswap_limit` default to `512m` (configurable via `BACKEND_MEM_LIMIT` / `BACKEND_MEMSWAP_LIMIT` in your env file)
- Redis not exposed to the host — internal compose network only
- Startup order: backend waits for Redis to pass its `redis-cli ping` healthcheck before starting

### Start all services

```bash
# Foreground
docker compose -f docker-compose-sb-backend-staging.yml up --build

# Detached / background
docker compose -f docker-compose-sb-backend-staging.yml up --build -d
```

### Restart a service

```bash
docker compose -f docker-compose-sb-backend-staging.yml restart sb-backend
```

### Stop (keeps containers)

```bash
docker compose -f docker-compose-sb-backend-staging.yml stop
```

### Stop and remove containers

```bash
docker compose -f docker-compose-sb-backend-staging.yml down
```

### Override memory limit

To run staging with more (or less) memory than the 512 MB default, set the vars in your env file or inline:

```bash
BACKEND_MEM_LIMIT=768m BACKEND_MEMSWAP_LIMIT=768m \
  docker compose -f docker-compose-sb-backend-staging.yml up --build -d
```

Or set them in `sb-backend/.env.docker` / `sb-backend/.env.docker.local.staging`:

```env
BACKEND_MEM_LIMIT=768m
BACKEND_MEMSWAP_LIMIT=768m
```

---

## Enabling Encryption

Encryption is **disabled by default** (`ENCRYPTION_ENABLED=false`). When enabled, conversation messages are encrypted at rest using AES-256-GCM with keys derived from GCP KMS.

### Local dev (encryption enabled)

The bind-mount already exposes the entire `sb-backend/` directory, so you just need to:

1. Place your GCP service account key at `sb-backend/config/serviceAccountKey.json`
2. Add to `sb-backend/.env.docker.local`:

```env
ENCRYPTION_ENABLED=true
GCP_PROJECT_ID=your-project-id
GCP_KMS_LOCATION=global
GCP_KMS_KEYRING=your-keyring
GCP_KMS_KEY=your-key
SERVICE_ACCOUNT_KEY=./config/serviceAccountKey.json
```

3. Start as normal:

```bash
docker compose -f docker-compose-sb-backend.yml up --build
```

### Staging (encryption enabled)

No bind-mount in staging, so mount the key file explicitly. Add to `docker-compose-sb-backend-staging.yml` under `sb-backend`:

```yaml
volumes:
  - /path/on/host/serviceAccountKey.json:/run/secrets/gcp-key.json:ro
```

And in `sb-backend/.env.docker`:

```env
ENCRYPTION_ENABLED=true
GCP_PROJECT_ID=your-project-id
GCP_KMS_LOCATION=global
GCP_KMS_KEYRING=your-keyring
GCP_KMS_KEY=your-key
SERVICE_ACCOUNT_KEY=/run/secrets/gcp-key.json
```

> The service account key is excluded from the Docker image by `.dockerignore` — never bake credentials into an image.

---

## Useful Commands

```bash
# View live logs (follow mode) — via docker CLI
docker logs -f sb-backend
docker logs -f soulbuddy-redis

# View last N lines of logs
docker logs --tail 100 sb-backend

# View live logs via compose (all services at once)
docker compose -f docker-compose-sb-backend.yml logs -f

# View logs for a single service via compose
docker compose -f docker-compose-sb-backend.yml logs -f sb-backend
docker compose -f docker-compose-sb-backend.yml logs -f soulbuddy-redis

# List running containers and their status
docker compose -f docker-compose-sb-backend.yml ps

# Open a shell inside the running backend container
docker exec -it sb-backend sh

# Rebuild a single service image (after dependency changes, no --no-cache)
docker compose -f docker-compose-sb-backend.yml build sb-backend

# Rebuild without cache (after changing requirements.txt / pyproject.toml)
docker compose -f docker-compose-sb-backend.yml build --no-cache

# Check Redis directly (dev only — Redis port 6379 is exposed to host)
redis-cli ping
redis-cli keys "*"
redis-cli flushall   # clear all cache entries

# Health check
curl http://localhost:8000/health

# Restart only the backend (e.g. to pick up env changes or a code fix)
docker compose -f docker-compose-sb-backend.yml restart sb-backend
```

> **Log files inside the container**: Application logs are written to `/app/logs` (controlled by `LOG_DIR=/app/logs` in the env file). Access them via `docker exec -it sb-backend sh` then `ls /app/logs/`.

---

## Troubleshooting

**Port already in use**
```bash
lsof -i :8000   # find the process using port 8000
lsof -i :6379   # find the process using port 6379 (Redis)
```

**Cannot connect to PostgreSQL**
- Use `host.docker.internal` (not `localhost`) in your DB URL when the DB runs on your laptop.
- If the DB runs in another container on the same compose network, use the container/service name.

**Redis connection errors in logs**
The server starts fine without Redis — it retries in the background. You'll see a warning:
```
⚠️  Redis unavailable — running without cache (DB fallback active)
```
Redis will reconnect automatically once it comes back up.

**Encryption: `Failed to retrieve master key from GCP KMS`**
- Check that `SERVICE_ACCOUNT_KEY` points to a file that exists inside the container.
- Verify the service account has the `cloudkms.cryptoKeyVersions.useToEncrypt` IAM role.
- If you don't need encryption, leave `ENCRYPTION_ENABLED=false` (the default).

**Dependencies changed**
After editing `requirements.txt` or `pyproject.toml`, rebuild the image:
```bash
docker compose -f docker-compose-sb-backend.yml build --no-cache
```
