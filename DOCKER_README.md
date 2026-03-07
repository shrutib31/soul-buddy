# SoulBuddy Backend — Docker Compose Guide

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)
- PostgreSQL running locally or accessible from the host machine
- A valid `.env.docker.local` file at `sb-backend/.env.docker.local`

---

## Compose File

```
docker-compose-sb-backend.yml
```

Starts two services:

| Service | Container | Description |
|---|---|---|
| `soulbuddy-redis` | `soulbuddy-redis` | Redis 7 cache (LRU, 64 MB) |
| `sb-backend` | `sb-backend` | FastAPI backend (Python 3.11) |

---

## Environment Variables (CLI overrides)

These variables can be set in your shell before running any `docker compose` command to override defaults.

| Variable | Default | Description |
|---|---|---|
| `ENV_FILE` | `./sb-backend/.env.docker.local` | Path to the env file loaded into `sb-backend` |
| `PORT` | `8000` | Port uvicorn binds to **inside** the container |
| `BACKEND_PORT` | `8000` | Host port mapped to the container's `PORT` |

---

## Commands

### Start all services (build images on first run)

```bash
docker compose -f docker-compose-sb-backend.yml up --build
```

### Start all services in the background (detached)

```bash
docker compose -f docker-compose-sb-backend.yml up --build -d
```

### Start only Redis

```bash
docker compose -f docker-compose-sb-backend.yml up soulbuddy-redis
```

### Start only the backend

```bash
docker compose -f docker-compose-sb-backend.yml up sb-backend
```

### Stop all services

```bash
docker compose -f docker-compose-sb-backend.yml down
```

### Stop one service

```bash
docker compose -f docker-compose-sb-backend.yml stop sb-backend
docker compose -f docker-compose-sb-backend.yml stop soulbuddy-redis
```

### Rebuild the backend image (after dependency changes)

```bash
docker compose -f docker-compose-sb-backend.yml build sb-backend
```

### View logs

```bash
# All services
docker compose -f docker-compose-sb-backend.yml logs -f

# Backend only
docker compose -f docker-compose-sb-backend.yml logs -f sb-backend

# Redis only
docker compose -f docker-compose-sb-backend.yml logs -f soulbuddy-redis
```

---

## Overriding Variables at Runtime

Set variables inline before the command, or export them in your shell first.

### Use a different env file

```bash
ENV_FILE=./sb-backend/.env.staging \
  docker compose -f docker-compose-sb-backend.yml up --build
```

### Run backend on a different host port (e.g. 9000 → 8000)

```bash
BACKEND_PORT=9000 \
  docker compose -f docker-compose-sb-backend.yml up --build
```

### Change the internal port and host port together

```bash
PORT=8080 BACKEND_PORT=8080 \
  docker compose -f docker-compose-sb-backend.yml up --build
```

### Combine multiple overrides

```bash
ENV_FILE=./sb-backend/.env.staging \
BACKEND_PORT=9000 \
PORT=9000 \
  docker compose -f docker-compose-sb-backend.yml up --build -d
```

---

## Notes

- **Hot reload**: `./sb-backend` is bind-mounted to `/app` inside the container. Code changes are live without rebuilding the image. Dependency changes (`requirements.txt`) still require a rebuild (`--build`).
- **Redis connection**: The backend connects to Redis via the compose service name `soulbuddy-redis` (`REDIS_URL=redis://soulbuddy-redis:6379`). Do not use `localhost` here.
- **PostgreSQL connection**: The backend connects to Postgres on the host machine via `host.docker.internal` (set in `.env.docker.local`). Ensure Postgres is running and accessible on the configured port.
- **Redis fail-silent**: If Redis is unavailable, cache operations degrade gracefully to direct DB reads — the backend will still start and serve requests.
- **Log files**: Application logs are written to `/app/logs` inside the container (`LOG_DIR=/app/logs`).
