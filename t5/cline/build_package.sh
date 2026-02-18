#!/bin/bash
#
# build_package.sh - Build the bias-mcp-server package
#
# Usage:
#   ./build_package.sh [--clean]
#
# Options:
#   --clean    Remove existing dist/ and build/ directories before building
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building cline-bias-server package ===${NC}"

# Check for --clean flag
if [[ "$1" == "--clean" ]]; then
    echo -e "${YELLOW}Cleaning previous builds...${NC}"
    rm -rf dist/ build/ *.egg-info
    echo "Clean complete."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python: $PYTHON_VERSION"

# Ensure build tools are installed
echo -e "${YELLOW}Installing build dependencies...${NC}"
pip install --quiet --upgrade build wheel hatchling

# Build the package
echo -e "${YELLOW}Building package...${NC}"
python -m build

# Check the output
if [ -d "dist" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo ""
    echo "Generated files:"
    ls -la dist/
    echo ""
    echo -e "${GREEN}To upload to TestPyPI:${NC}"
    echo "  ./publish_testpypi.sh"
    echo ""
    echo -e "${GREEN}To upload to PyPI:${NC}"
    echo "  ./publish_pypi.sh"
else
    echo -e "${RED}Build failed - no dist directory created${NC}"
    exit 1
fi