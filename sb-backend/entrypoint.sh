#!/bin/sh
set -e

# Initialize database schema/migrations (non-fatal — server still starts if this fails)
echo "Initializing database..."
python scripts/init_db.py || echo "WARNING: init_db.py failed — continuing startup"

# Start FastAPI server
echo "Starting server..."
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --limit-concurrency 20 \
    --limit-max-requests 500 \
    --no-access-log
