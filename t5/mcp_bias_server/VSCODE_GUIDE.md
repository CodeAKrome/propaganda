# Bias MCP Server - VSCode Installation & Testing Guide

This guide provides complete instructions for installing, configuring, and testing the Bias MCP Server in VSCode with Kilo Code.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [VSCode Configuration](#vscode-configuration)
4. [Testing the MCP Server](#testing-the-mcp-server)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **VSCode**: Latest version
- **Kilo Code Extension**: Installed in VSCode
- **Model Weights**: Trained T5-LoRA adapter (optional for Ollama backend)

### Check Python Version

```bash
python3 --version
# Should show 3.10.x or higher
```

### LoRA Adapter Files

The trained model files are located at `t5/bias-detector-output/`:

| File | Description | Size |
|------|-------------|------|
| `adapter_config.json` | LoRA configuration | ~1KB |
| `adapter_model.safetensors` | Trained weights | ~18MB |
| `tokenizer.json` | Tokenizer vocabulary | ~2MB |
| `tokenizer_config.json` | Tokenizer config | ~2KB |
| `device_info.json` | Training device info | ~22B |

**LoRA Configuration:**
- Rank (r): 16
- Alpha: 32
- Target modules: q, v (attention layers)
- Base model: t5-large

### Install Kilo Code Extension

1. Open VSCode
2. Go to Extensions (Cmd+Shift+X)
3. Search for "Kilo Code"
4. Click Install

---

## Installation Methods

### Method 1: Automated Installation (Recommended)

Run the provided installation script:

```bash
# Navigate to the MCP server directory
cd t5/mcp_bias_server

# Run the installation script
./install_mcp_vscode.sh --model-path /path/to/bias-detector-output
```

**Script Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to LoRA adapter | `./bias-detector-output` |
| `--base-model` | Base T5 model name | `t5-large` |
| `--device` | Device selection | `auto` |

**Example with all options:**

```bash
./install_mcp_vscode.sh \
    --model-path ~/models/bias-detector-output \
    --base-model t5-large \
    --device mps
```

### Method 2: Manual Installation

#### Step 1: Install the Package

```bash
# From PyPI (after publishing)
pip install bias-mcp-server

# Or from source
cd t5/mcp_bias_server
pip install -e .
```

#### Step 2: Verify Installation

```bash
# Check if the command is available
which bias-mcp-server

# Or test the Python module
python3 -c "from mcp_bias_server.bias_engine import BiasEngine; print('OK')"
```

#### Step 3: Configure VSCode Settings

Open the MCP settings file:

**macOS:**
```bash
code ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

**Linux:**
```bash
code ~/.config/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

Add the server configuration:

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/path/to/propaganda/t5/bias-detector-output",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

**Important:** Replace `/path/to/propaganda` with the actual path to your project directory.

**Important Notes:**
- Use absolute paths for `BIAS_MODEL_PATH`
- Use `python3` instead of `bias-mcp-server` for better reliability
- The `env` section is required for model configuration

---

## VSCode Configuration

### Configuration File Location

The MCP settings file is located at:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json` |
| Linux | `~/.config/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json` |
| Windows | `%APPDATA%\Code\User\globalStorage\kilocode.kilo-code\settings\mcp_settings.json` |

### Full Configuration Example

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/Users/username/models/bias-detector-output",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    },
    "other-mcp-server": {
      "command": "other-server"
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BIAS_MODEL_PATH` | Path to LoRA adapter weights | `./bias-detector-output` |
| `BIAS_BASE_MODEL` | HuggingFace base model name | `t5-large` |
| `BIAS_DEVICE` | Device: `auto`, `mps`, `cuda`, `cpu` | `auto` |

### Device Selection Guide

| Device | When to Use |
|--------|-------------|
| `auto` | Let the system choose (recommended) |
| `mps` | Apple Silicon (M1/M2/M3) |
| `cuda` | NVIDIA GPU |
| `cpu` | No GPU available |

---

## Testing the MCP Server

### Step 1: Restart VSCode

After configuration, restart VSCode to load the new MCP server:

```bash
# On macOS
killall "Visual Studio Code"
code
```

Or use Cmd+Q to quit, then reopen.

### Step 2: Check MCP Server Status

1. Open the Kilo Code panel (click the Kilo icon in the sidebar)
2. Look for "MCP Servers" section
3. Verify `bias-analyzer` is listed and connected

### Step 3: Test with a Simple Query

In the Kilo Code chat, type:

```
Use the bias-analyzer to check the political bias of this text: 
"The president announced new economic policies today."
```

Expected response:
```json
{
  "success": true,
  "result": {
    "dir": {"L": 0.25, "C": 0.50, "R": 0.25},
    "deg": {"L": 0.60, "M": 0.30, "H": 0.10},
    "reason": "The article presents policy announcements in a neutral tone..."
  }
}
```

### Step 4: Test Individual Tools

#### Test `analyze_bias`

```
Call the analyze_bias tool with this text: 
"The radical left continues to push their extreme agenda."
```

#### Test `analyze_batch`

```
Use analyze_batch to analyze these texts:
1. "Markets rallied on positive economic news."
2. "The administration's policies are destroying our country."
```

#### Test `get_model_info`

```
Call get_model_info to see the loaded model details.
```

### Step 5: Command Line Testing

Test the server directly from the terminal:

```bash
# Set environment variables
export BIAS_MODEL_PATH=/path/to/bias-detector-output
export BIAS_DEVICE=auto

# Run the server
python3 -m mcp_bias_server.server

# The server will start and wait for MCP commands on stdin
```

### Step 6: Python Testing

Create a test script:

```python
#!/usr/bin/env python3
"""Test the BiasEngine directly."""

import os
os.environ["BIAS_MODEL_PATH"] = "/path/to/bias-detector-output"

from mcp_bias_server.bias_engine import BiasEngine

# Create engine
engine = BiasEngine()

# Test analysis
result = engine.analyze("The president announced new policies today.")
print(f"Direction: {result.direction}")
print(f"Degree: {result.degree}")
print(f"Reason: {result.reason}")
```

Run it:
```bash
python3 test_bias.py
```

---

## Troubleshooting

### Server Not Appearing in Kilo Code

**Symptoms:** The `bias-analyzer` server doesn't appear in the MCP Servers list.

**Solutions:**

1. **Check configuration file location:**
   ```bash
   # macOS
   ls ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
   ```

2. **Verify JSON syntax:**
   ```bash
   python3 -c "import json; json.load(open('path/to/mcp_settings.json'))"
   ```

3. **Restart VSCode completely:**
   ```bash
   killall "Visual Studio Code"
   code
   ```

### Model Not Found Error

**Symptoms:** `Model path does not exist` or `Adapter weights not found`

**Solutions:**

1. **Check the path:**
   ```bash
   ls -la $BIAS_MODEL_PATH
   ```

2. **Use absolute path:**
   ```json
   {
     "env": {
       "BIAS_MODEL_PATH": "/Users/username/full/path/to/bias-detector-output"
     }
   }
   ```

3. **Verify model files exist:**
   ```bash
   ls $BIAS_MODEL_PATH/adapter_config.json
   ls $BIAS_MODEL_PATH/adapter_model.safetensors
   ```

### Import Errors

**Symptoms:** `ModuleNotFoundError: No module named 'mcp_bias_server'`

**Solutions:**

1. **Install the package:**
   ```bash
   pip install bias-mcp-server
   ```

2. **Or install from source:**
   ```bash
   cd t5/mcp_bias_server
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import mcp_bias_server; print('OK')"
   ```

### Device Errors

**Symptoms:** `MPS device not available` or `CUDA out of memory`

**Solutions:**

1. **Force CPU mode:**
   ```json
   {
     "env": {
       "BIAS_DEVICE": "cpu"
     }
   }
   ```

2. **Check MPS availability (macOS):**
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **Check CUDA availability:**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### Server Crashes on Startup

**Symptoms:** Server appears briefly then disappears.

**Solutions:**

1. **Check server logs:**
   - Open VSCode Output panel (View > Output)
   - Select "MCP" from the dropdown

2. **Test server manually:**
   ```bash
   export BIAS_MODEL_PATH=/path/to/model
   python3 -m mcp_bias_server.server
   ```

3. **Check Python version:**
   ```bash
   python3 --version  # Should be 3.10+
   ```

### Slow First Inference

**Symptoms:** First analysis takes a long time.

**Explanation:** The model loads lazily on first inference. This is normal.

**Solutions:**

1. **Accept the delay** (one-time only)
2. **Pre-load the model:**
   ```python
   from mcp_bias_server.bias_engine import BiasEngine
   engine = BiasEngine(lazy_load=False)
   ```

### Tool Not Found

**Symptoms:** `Unknown tool: analyze_bias`

**Solutions:**

1. **Verify server is running:**
   - Check Kilo Code MCP panel
   - Look for green status indicator

2. **Restart the server:**
   - Click "Restart" next to the server in Kilo Code
   - Or restart VSCode

3. **Check server implementation:**
   ```bash
   python3 -c "from mcp_bias_server.server import server; print(server.list_tools())"
   ```

---

## Advanced Configuration

### Using with Ollama Backend

If you don't have the trained T5 model, use Ollama:

1. **Install Ollama:**
   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3.2
   ```

2. **Run the demo:**
   ```bash
   python3 demo_ollama_bias.py
   ```

### Multiple Model Configurations

Configure multiple bias analyzers with different models:

```json
{
  "mcpServers": {
    "bias-analyzer-t5": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/models/t5-bias-output"
      }
    },
    "bias-analyzer-alt": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/models/alt-bias-output"
      }
    }
  }
}
```

### Logging Configuration

Enable debug logging:

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/path/to/model",
        "BIAS_DEBUG": "1"
      }
    }
  }
}
```

---

## Quick Reference

### Installation Commands

```bash
# Install package
pip install bias-mcp-server

# Automated VSCode setup
./install_mcp_vscode.sh --model-path /path/to/model

# Manual verification
python3 -c "from mcp_bias_server.bias_engine import BiasEngine; print('OK')"
```

### Configuration Template

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/ABSOLUTE/PATH/TO/model",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

### Test Queries

```
# Simple test
Use bias-analyzer to analyze: "The stock market rose today."

# Batch test
Use analyze_batch for these: ["Text 1", "Text 2"]

# Model info
Call get_model_info
```

---

## Support

For issues not covered in this guide:

1. Check the [DOCUMENTATION.md](DOCUMENTATION.md) for full API reference
2. Check the [README.md](README.md) for quick start guide
3. Open an issue on GitHub with:
   - VSCode version
   - Python version
   - Error messages
   - Configuration (redact sensitive paths)