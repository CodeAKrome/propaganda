#!/bin/bash
#
# install_mcp_vscode.sh - Install bias-mcp-server for VSCode (Kilo Code)
#
# This script:
# 1. Installs the bias-mcp-server package via pip
# 2. Configures the MCP server in VSCode settings
#
# Usage:
#   ./install_mcp_vscode.sh [--model-path /path/to/model]
#
# Options:
#   --model-path    Path to LoRA adapter (default: ./bias-detector-output)
#   --base-model    Base T5 model name (default: t5-large)
#   --device        Device selection: auto, mps, cuda, cpu (default: auto)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values - use relative path from project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/bias-detector-output"
BASE_MODEL="t5-large"
DEVICE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH   Path to LoRA adapter (default: ./bias-detector-output)"
            echo "  --base-model NAME   Base T5 model name (default: t5-large)"
            echo "  --device DEVICE     Device: auto, mps, cuda, cpu (default: auto)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Bias MCP Server - VSCode Installation ===${NC}"
echo ""

# Step 1: Install the package
echo -e "${BLUE}Step 1: Installing bias-mcp-server...${NC}"
pip install bias-mcp-server

# Verify installation
if ! command -v bias-mcp-server &> /dev/null; then
    echo -e "${RED}Error: bias-mcp-server command not found after installation${NC}"
    exit 1
fi

echo -e "${GREEN}✓ bias-mcp-server installed successfully${NC}"
echo ""

# Step 2: Get the installation path
INSTALL_PATH=$(which bias-mcp-server)
echo -e "${BLUE}Installation path: ${INSTALL_PATH}${NC}"
echo ""

# Step 3: Resolve model path to absolute path
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$(pwd)/$MODEL_PATH"
fi
echo -e "${BLUE}Model path: ${MODEL_PATH}${NC}"
echo ""

# Step 4: Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model path does not exist: ${MODEL_PATH}${NC}"
    echo -e "${YELLOW}You will need to download or train the model before using the server.${NC}"
    echo ""
fi

# Step 5: Determine VSCode settings path
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SETTINGS_PATH="$HOME/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    SETTINGS_PATH="$HOME/.config/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${BLUE}Step 2: Configuring MCP settings...${NC}"
echo -e "${BLUE}Settings path: ${SETTINGS_PATH}${NC}"
echo ""

# Step 6: Create or update MCP settings
mkdir -p "$(dirname "$SETTINGS_PATH")"

if [ -f "$SETTINGS_PATH" ]; then
    echo -e "${YELLOW}Existing MCP settings found. Creating backup...${NC}"
    cp "$SETTINGS_PATH" "${SETTINGS_PATH}.backup"
    
    # Read existing settings
    SETTINGS=$(cat "$SETTINGS_PATH")
else
    # Create new settings file
    SETTINGS='{"mcpServers": {}}'
fi

# Create the server configuration
# Use Python directly for more reliable execution
PYTHON_PATH=$(which python3)
SERVER_COMMAND="$PYTHON_PATH -m mcp_bias_server.server"

# Build the new server entry
cat > /tmp/bias_server_config.json << EOF
{
  "command": "$PYTHON_PATH",
  "args": ["-m", "mcp_bias_server.server"],
  "env": {
    "BIAS_MODEL_PATH": "$MODEL_PATH",
    "BIAS_BASE_MODEL": "$BASE_MODEL",
    "BIAS_DEVICE": "$DEVICE"
  }
}
EOF

# Use Python to merge JSON (more reliable than jq)
python3 << PYTHON_SCRIPT
import json
import sys

try:
    # Read existing settings
    with open("$SETTINGS_PATH", "r") as f:
        settings = json.load(f)
except FileNotFoundError:
    settings = {"mcpServers": {}}

# Read new server config
with open("/tmp/bias_server_config.json", "r") as f:
    new_server = json.load(f)

# Ensure mcpServers exists
if "mcpServers" not in settings:
    settings["mcpServers"] = {}

# Add/update the bias-analyzer server
settings["mcpServers"]["bias-analyzer"] = new_server

# Write back
with open("$SETTINGS_PATH", "w") as f:
    json.dump(settings, f, indent=2)

print("Settings updated successfully")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MCP settings configured${NC}"
else
    echo -e "${RED}Error updating MCP settings${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Configuration:"
echo "  Server name:    bias-analyzer"
echo "  Model path:     $MODEL_PATH"
echo "  Base model:     $BASE_MODEL"
echo "  Device:         $DEVICE"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Restart VSCode to load the new MCP server"
echo ""
echo "2. The following tools will be available:"
echo "   - analyze_bias: Analyze text for political bias"
echo "   - analyze_batch: Analyze multiple texts"
echo "   - get_model_info: Get model information"
echo ""
echo "3. Test the server by asking the AI:"
echo "   'Use the bias-analyzer to check the political bias of this text: [your text]'"
echo ""
echo -e "${BLUE}Settings file: ${SETTINGS_PATH}${NC}"
