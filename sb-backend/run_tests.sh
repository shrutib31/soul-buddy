#!/usr/bin/env bash
#
# Test Runner Script for SoulBuddy Backend
#
# Usage:
#   ./run_tests.sh              # Run all unit tests (exclude integration)
#   ./run_tests.sh integration  # Run all tests including integration
#   ./run_tests.sh coverage     # Run with coverage report
#   ./run_tests.sh specific tests/graph/nodes/test_intent_detection.py
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}SoulBuddy Backend Test Suite${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Installing...${NC}"
    pip install pytest pytest-asyncio pytest-cov
fi

# Default: run unit tests only
MODE=${1:-unit}

case $MODE in
    integration)
        echo -e "${GREEN}Running ALL tests (including integration)...${NC}"
        echo ""
        pytest tests/ -v
        ;;
    
    coverage)
        echo -e "${GREEN}Running tests with coverage...${NC}"
        echo ""
        pytest tests/ -v --cov=graph --cov=api --cov=config --cov-report=term-missing --cov-report=html
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    specific)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify test file or path${NC}"
            echo "Usage: ./run_tests.sh specific tests/path/to/test_file.py"
            exit 1
        fi
        echo -e "${GREEN}Running specific test: $2${NC}"
        echo ""
        pytest "$2" -v
        ;;
    
    unit|*)
        echo -e "${GREEN}Running UNIT tests (excluding integration)...${NC}"
        echo ""
        pytest tests/ -v -m "not integration"
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}============================================${NC}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
