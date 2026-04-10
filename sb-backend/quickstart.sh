#!/bin/bash
# Quick setup script for Python venv, dependencies, and server start

set -e

# Name of the venv directory
env_dir=".venv"

# 1. Create venv if it doesn't exist
if [ ! -d "$env_dir" ]; then
  echo "Creating virtual environment in $env_dir..."
  python3 -m venv "$env_dir"
fi

# 2. Activate venv
source "$env_dir/bin/activate"
echo "Virtual environment activated."

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies (requirements.txt must exist)
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found! Please create one."
  exit 1
fi

# 5. Start the server (adjust as needed)
echo "Starting the server..."
python server.py
