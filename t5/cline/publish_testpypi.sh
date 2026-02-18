#!/bin/bash
#
# publish_testpypi.sh - Publish the bias-mcp-server package to TestPyPI
#
# Usage:
#   ./publish_testpypi.sh
#
# Prerequisites:
#   1. Create an account at https://test.pypi.org/
#   2. Create an API token at https://test.pypi.org/manage/account/token/
#   3. Set environment variables:
#      export TEST_PYPI_USERNAME=__token__
#      export TEST_PYPI_TOKEN=your-api-token-here
#
# Or create ~/.pypirc with:
#   [distutils]
#   index-servers =
#       testpypi
#
#   [testpypi]
#   repository = https://test.pypi.org/legacy/
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
NC='\033[0m' # No Color

echo -e "${GREEN}=== Publishing bias-mcp-server to TestPyPI ===${NC}"

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

# Check for credentials
if [ -z "$TEST_PYPI_TOKEN" ]; then
    # Check for ~/.pypirc
    if [ -f ~/.pypirc ]; then
        echo -e "${YELLOW}Using credentials from ~/.pypirc${NC}"
        twine upload --repository testpypi dist/*
    else
        echo -e "${RED}Error: No TestPyPI credentials found${NC}"
        echo ""
        echo "Set environment variables:"
        echo "  export TEST_PYPI_USERNAME=__token__"
        echo "  export TEST_PYPI_TOKEN=your-api-token"
        echo ""
        echo "Or create ~/.pypirc with your credentials"
        exit 1
    fi
else
    echo -e "${YELLOW}Using environment variable credentials${NC}"
    twine upload --repository testpypi \
        --username "__token__" \
        --password "$TEST_PYPI_TOKEN" \
        dist/*
fi

echo ""
echo -e "${GREEN}Successfully published to TestPyPI!${NC}"
echo ""
echo "Install from TestPyPI:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bias-mcp-server"
echo ""
echo "View package at:"
echo "  https://test.pypi.org/project/bias-mcp-server/"