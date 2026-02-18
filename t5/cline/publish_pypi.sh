#!/bin/bash
# Publish Cline Bias Server to PyPI

#
# Usage:
#   ./publish_pypi.sh [--test-first]
#
# Options:
#   --test-first    Upload to TestPyPI first, then prompt for PyPI upload
#
# Prerequisites:
#   1. Create an account at https://pypi.org/
#   2. Create an API token at https://pypi.org/manage/account/token/
#   3. Set environment variables:
#      export PYPI_USERNAME=__token__
#      export PYPI_TOKEN=your-api-token-here
#
# Or create ~/.pypirc with:
#   [distutils]
#   index-servers =
#       pypi
#
#   [pypi]
#   username = __token__
#   password = pypi-...
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Publishing bias-mcp-server to PyPI ===${NC}"

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo -e "${RED}Error: dist/ directory not found${NC}"
    echo "Run ./build_package.sh first"
    exit 1
fi

# Check for twine
if ! command -v twine &> /dev/null; then
    echo -e "${YELLOW}Installing twine...${NC}"
    pip install --quiet twine
fi

# If --test-first flag, upload to TestPyPI first
if [[ "$1" == "--test-first" ]]; then
    echo -e "${BLUE}Step 1: Uploading to TestPyPI first...${NC}"
    ./publish_testpypi.sh
    
    echo ""
    echo -e "${YELLOW}TestPyPI upload complete. Verify the package works:${NC}"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bias-mcp-server"
    echo ""
    read -p "Continue to PyPI upload? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Show what we're about to upload
echo -e "${YELLOW}Files to upload:${NC}"
ls -la dist/
echo ""

# Confirm upload
read -p "Upload to PyPI? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check for credentials
if [ -z "$PYPI_TOKEN" ]; then
    # Check for ~/.pypirc
    if [ -f ~/.pypirc ]; then
        echo -e "${YELLOW}Using credentials from ~/.pypirc${NC}"
        twine upload dist/*
    else
        echo -e "${RED}Error: No PyPI credentials found${NC}"
        echo ""
        echo "Set environment variables:"
        echo "  export PYPI_USERNAME=__token__"
        echo "  export PYPI_TOKEN=your-api-token"
        echo ""
        echo "Or create ~/.pypirc with your credentials"
        exit 1
    fi
else
    echo -e "${YELLOW}Using environment variable credentials${NC}"
    twine upload \
        --username "__token__" \
        --password "$PYPI_TOKEN" \
        dist/*
fi

echo ""
echo -e "${GREEN}Successfully published to PyPI!${NC}"
echo ""
echo "Install from PyPI:"
echo "  pip install bias-mcp-server"
echo ""
echo "View package at:"
echo "  https://pypi.org/project/bias-mcp-server/"