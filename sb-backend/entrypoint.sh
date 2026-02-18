#!/bin/sh
set -e

# Initialize database (runs your init_db.py script)
echo "Initializing database..."
python scripts/init_db.py

# Start FastAPI server
echo "Starting server..."
exec python server.py
