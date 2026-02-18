#!/bin/bash
# Universal installer for bias-mcp-server
# Works with: KiloCode VSCode, KiloCode CLI, Claude CLI

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Bias MCP Server Installer ===${NC}"

# Detect platform
PLATFORM="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
fi

echo "Platform: $PLATFORM"

# Install the package
echo -e "\n${YELLOW}Installing bias-mcp-server from PyPI...${NC}"
pip install --upgrade bias-mcp-server

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 -c "from mcp_bias_server.bias_engine import BiasEngine; print('✓ Package installed successfully')"

# Detect which clients to configure
CONFIGURE_KILOCODE_VSCODE=false
CONFIGURE_KILOCODE_CLI=false
CONFIGURE_CLAUDE_CLI=false

# Check for VSCode with KiloCode extension
KILOCODE_VSCODE_PATH="$HOME/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json"
if [[ "$PLATFORM" == "linux" ]]; then
    KILOCODE_VSCODE_PATH="$HOME/.config/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json"
fi

if [[ -d "$(dirname "$KILOCODE_VSCODE_PATH")" ]] || [[ -f "$KILOCODE_VSCODE_PATH" ]]; then
    CONFIGURE_KILOCODE_VSCODE=true
fi

# Check for KiloCode CLI
if command -v kilocode &> /dev/null; then
    CONFIGURE_KILOCODE_CLI=true
fi

# Check for Claude CLI
if command -v claude &> /dev/null; then
    CONFIGURE_CLAUDE_CLI=true
fi

# MCP server configuration
MCP_CONFIG='{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {}
    }
  }
}'

# Function to merge MCP config
merge_mcp_config() {
    local config_file="$1"
    local temp_file=$(mktemp)
    
    if [[ ! -f "$config_file" ]]; then
        # Create new config
        mkdir -p "$(dirname "$config_file")"
        echo "$MCP_CONFIG" > "$config_file"
        echo -e "${GREEN}✓ Created new MCP config at $config_file${NC}"
        return
    fi
    
    # Check if bias-analyzer already exists
    if grep -q '"bias-analyzer"' "$config_file" 2>/dev/null; then
        echo -e "${YELLOW}⚠ bias-analyzer already configured in $config_file${NC}"
        return
    fi
    
    # Add bias-analyzer to existing config
    python3 << EOF
import json
import sys

config_file = "$config_file"
try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"]["bias-analyzer"] = {
    "command": "python3",
    "args": ["-m", "mcp_bias_server.server"],
    "env": {}
}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Added bias-analyzer to", config_file)
EOF
}

# Configure clients
echo -e "\n${YELLOW}Configuring MCP clients...${NC}"

if [[ "$CONFIGURE_KILOCODE_VSCODE" == "true" ]]; then
    echo -e "\n${GREEN}Configuring KiloCode for VSCode...${NC}"
    merge_mcp_config "$KILOCODE_VSCODE_PATH"
fi

if [[ "$CONFIGURE_KILOCODE_CLI" == "true" ]]; then
    echo -e "\n${GREEN}Configuring KiloCode CLI...${NC}"
    KILOCODE_CLI_CONFIG="$HOME/.kilocode/mcp_settings.json"
    merge_mcp_config "$KILOCODE_CLI_CONFIG"
fi

if [[ "$CONFIGURE_CLAUDE_CLI" == "true" ]]; then
    echo -e "\n${GREEN}Configuring Claude CLI...${NC}"
    CLAUDE_CONFIG="$HOME/.claude/claude_desktop_config.json"
    if [[ "$PLATFORM" == "macos" ]]; then
        CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    fi
    merge_mcp_config "$CLAUDE_CONFIG"
fi

# If no clients detected, show manual config
if [[ "$CONFIGURE_KILOCODE_VSCODE" == "false" && "$CONFIGURE_KILOCODE_CLI" == "false" && "$CONFIGURE_CLAUDE_CLI" == "false" ]]; then
    echo -e "\n${YELLOW}No MCP clients detected. Manual configuration:${NC}"
    echo -e "\nAdd this to your MCP settings file:"
    echo "$MCP_CONFIG"
fi

# Test the server
echo -e "\n${YELLOW}Testing bias analyzer...${NC}"
python3 << 'TESTEOF'
from mcp_bias_server.bias_engine import BiasEngine
import json

engine = BiasEngine()
result = engine.analyze("The president announced new economic policies today.")
print("✓ Test analysis successful:")
print(json.dumps(result.to_dict(), indent=2))
TESTEOF

echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo -e "\nTo use the bias analyzer:"
echo -e "  1. Restart your MCP client (VSCode/Claude)"
echo -e "  2. The 'bias-analyzer' MCP server will be available"
echo -e "  3. Use tools: analyze_bias, analyze_batch, get_model_info"
