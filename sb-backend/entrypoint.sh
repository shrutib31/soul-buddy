#!/bin/sh
set -e

echo "Initializing database..."
python scripts/init_db.py

echo "Starting server..."
exec python server.py
