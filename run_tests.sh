#!/bin/bash

# Test runner script for larsnet project
# This script runs all tests with coverage reporting

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Running larsnet Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running in Docker or local
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}Running inside Docker container${NC}"
    cd /app
else
    echo -e "${YELLOW}Running on local machine${NC}"
    echo -e "${YELLOW}Note: It's recommended to run tests inside Docker for consistency${NC}"
    echo -e "${YELLOW}Use: docker exec -it larsnet-larsnet_env-1 bash /app/run_tests.sh${NC}"
    echo ""
fi

# Default: run all tests with coverage
TEST_PATH="${1:-stems_to_midi}"
COVERAGE_ARGS="--cov=stems_to_midi --cov-report=term-missing --cov-report=html"

echo -e "${BLUE}Running tests...${NC}"
echo ""

# Run pytest with coverage
pytest ${TEST_PATH} ${COVERAGE_ARGS}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Test run complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Coverage report saved to: htmlcov/index.html${NC}"
echo -e "${BLUE}To view: open htmlcov/index.html${NC}"
echo ""
