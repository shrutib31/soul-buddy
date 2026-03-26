#!/bin/bash
# Quick Docker setup and server start script

set -e

# 1. Build the Docker image
echo "Building Docker image..."
docker build -t sb-backend .

# 2. Run the Docker container
echo "Starting the server in Docker..."
docker run --rm -it -p 8000:8000 sb-backend
