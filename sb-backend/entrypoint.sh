#!/bin/sh
set -e

# Initialize database schema/migrations (non-fatal — server still starts if this fails)
echo "Initializing database..."
python scripts/init_db.py || echo "WARNING: init_db.py failed — continuing startup"

# Start FastAPI server
echo "Starting server..."
exec python server.py
