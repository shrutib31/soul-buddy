#!/bin/bash

# ============================================================================
# SoulBuddy Backend - Setup Script
# ============================================================================
# This script sets up the development environment for the SoulBuddy backend
# It supports both traditional venv and UV package manager
# ============================================================================

set -e  # Exit on error

echo "=================================="
echo "SoulBuddy Backend Setup"
echo "=================================="
echo ""

# Check if UV is installed
if command -v uv &> /dev/null; then
    echo "✅ UV detected - using UV for faster dependency management"
    USE_UV=true
else
    echo "⚠️  UV not found - falling back to standard venv + pip"
    echo "   Install UV for faster package management: curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

echo ""
echo "Step 1: Creating virtual environment..."

if [ "$USE_UV" = true ]; then
    # Create venv using UV (without managed Python)
    uv venv venv --python python3
    echo "✅ Virtual environment created with UV"
else
    # Create venv using standard Python
    python3 -m venv venv
    echo "✅ Virtual environment created with Python venv"
fi

echo ""
echo "Step 2: Activating virtual environment..."

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "Step 3: Installing dependencies..."

if [ "$USE_UV" = true ]; then
    # Install with UV (using sync for reproducible installs)
    echo "Installing main dependencies with UV..."
    uv pip sync requirements.txt
    echo ""
    echo "Installing development dependencies..."
    uv pip sync requirements.txt requirements-dev.txt
    echo "✅ UV pip sync completed"
else
    # Install with pip
    pip install --upgrade pip
    echo "Installing main dependencies..."
    pip install -r requirements.txt
    echo ""
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

echo ""
echo "✅ All dependencies installed successfully"

echo ""
echo "Step 4: Creating necessary directories..."

# Create logs directory
mkdir -p logs
echo "✅ Created logs/ directory"

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure your .env file with database credentials"
echo ""
echo "3. Run the server:"
echo "   python server.py"
echo "   or"
echo "   uvicorn server:app --reload"
echo ""
