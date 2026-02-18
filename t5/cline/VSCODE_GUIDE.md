# Cline Bias MCP Server - VSCode Installation Guide

This guide explains how to use the Cline Bias MCP Server in VSCode with Kilo Code.

## Quick Setup

### Step 1: Install the Package (Already Done!)

The package is already installed and tested:
```bash
pip install -e t5/cline
```

### Step 2: Configure VSCode MCP Settings

Open your MCP settings file:

**macOS:**
```bash
code ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

**Linux:**
```bash
code ~/.config/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

### Step 3: Add the Server Configuration

Add this to your `mcp_settings.json`:

```json
{
  "mcpServers": {
    "cline-bias": {
      "command": "python3",
      "args": ["-m", "cline_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/Users/kyle/hub/propaganda/t5/cline/src/cline_bias_server/bias-detector-output",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

**Important:** Adjust the path if your project is in a different location.

### Step 4: Restart VSCode

```bash
killall "Visual Studio Code"
code
```

Or use Cmd+Q to quit and reopen VSCode.

---

## Testing

After restarting, in the Kilo Code chat, try:

```
Use cline-bias to analyze: "The president announced new economic policies today."
```

Or test individual tools:

```
Call analyze_bias with: "The radical left continues to push their extreme agenda."
```

---

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `BIAS_MODEL_PATH` | Path to LoRA adapter weights | Bundled model |
| `BIAS_BASE_MODEL` | HuggingFace base model | t5-large |
| `BIAS_DEVICE` | Device: auto, mps, cuda, cpu | auto |

### Device Selection

| Device | When to Use |
|--------|-------------|
| `auto` | Let system choose (recommended) |
| `mps` | Apple Silicon (M1/M2/M3/M4) |
| `cuda` | NVIDIA GPU |
| `cpu` | No GPU |

---

## Troubleshooting

### Server Not Appearing

1. Check JSON syntax:
```bash
python3 -c "import json; json.load(open('~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json'))"
```

2. Restart VSCode completely

### Model Not Found

Verify the path exists:
```bash
ls /Users/kyle/hub/propaganda/t5/cline/src/cline_bias_server/bias-detector-output/
```

### Slow First Run

The model loads lazily on first use - this is normal. Subsequent runs will be faster.

---

## Alternative: Use Without MCP

You can also use the bias analyzer directly in Python:

```python
from cline_bias_server import quick_analyze

result = quick_analyze("Your text here")
print(result)
# {'dominant_direction': 'Center', 'dominant_degree': 'High', ...}
```

Or use the CLI:
```bash
bias "Your text here"
```
