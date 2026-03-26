#!/bin/bash
# Quick Docker Compose startup for sb-backend (local dev)

echo "Starting sb-backend with Docker Compose..."
docker compose -f "$COMPOSE_FILE" up --build

set -e

usage() {
  echo "Usage: $0 [deploy|builddeploy|start|rebuild]"
  echo "  deploy      : Start containers (no rebuild, no removal, reuse existing)"
  echo "  builddeploy : Remove all containers, rebuild, and start fresh"
  echo "  start       : Start containers if stopped (no build, no removal)"
  echo "  rebuild     : Only rebuild and replace the 'soulbuddy' backend container, reuse others"
  exit 1
}

if [ $# -eq 0 ]; then
  usage
fi

MODE="$1"

# Go to repo root (one level above sb-backend)
cd "$(dirname "$0")/.."

# Compose file and env file
COMPOSE_FILE="docker-compose-sb-backend.yml"
ENV_FILE="sb-backend/.env.docker.local"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Error: $COMPOSE_FILE not found in repo root."
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Warning: $ENV_FILE not found. Using default env vars."
fi

case "$MODE" in
  builddeploy)
    echo "Removing all containers defined in $COMPOSE_FILE..."
    SERVICE_NAMES=$(docker compose -f "$COMPOSE_FILE" config --services)
    for SVC in $SERVICE_NAMES; do
      CONTAINER_IDS=$(docker ps -aq -f name="${SVC}")
      if [ -n "$CONTAINER_IDS" ]; then
        echo "Removing existing container(s) for service $SVC: $CONTAINER_IDS"
        docker rm -f $CONTAINER_IDS || true
      fi
    done
    echo "Building and starting sb-backend with Docker Compose..."
    docker compose -f "$COMPOSE_FILE" up --build
    ;;
  deploy)
    echo "Starting sb-backend containers (no build, no removal)..."
    docker compose -f "$COMPOSE_FILE" up
    ;;
  start)
    echo "Starting sb-backend containers if stopped (no build, no removal)..."
    docker compose -f "$COMPOSE_FILE" start
    ;;
  rebuild)
    # Only rebuild and replace the 'soulbuddy' backend container, reuse others
    SOULBUDDY_SERVICE="sb-backend"
    echo "Stopping and removing only the $SOULBUDDY_SERVICE container..."
    CONTAINER_ID=$(docker ps -aq -f name="$SOULBUDDY_SERVICE")
    if [ -n "$CONTAINER_ID" ]; then
      docker rm -f $CONTAINER_ID || true
    fi
    echo "Rebuilding $SOULBUDDY_SERVICE service..."
    docker compose -f "$COMPOSE_FILE" build $SOULBUDDY_SERVICE
    echo "Starting $SOULBUDDY_SERVICE and reusing other containers..."
    docker compose -f "$COMPOSE_FILE" up -d $SOULBUDDY_SERVICE
    ;;
  *)
    usage
    ;;
esac
