# T5 Bias Detection - Standalone Installation Guide

Install the bias detector in a completely new directory, independent of the source git tree.

---

## Overview

This guide installs the bias detector as a standalone package with:
- Model weights copied to a dedicated location
- Package installed from PyPI (or local wheel)
- No dependency on source git repository

---

## Step 1: Create Installation Directory

```bash
# Create a new directory anywhere (example: ~/bias-detector)
mkdir -p ~/bias-detector
cd ~/bias-detector
```

---

## Step 2: Copy Model Weights

The trained LoRA adapter weights must be copied from the source:

```bash
# Create model directory
mkdir -p models/bias-detector-output

# Copy from source (adjust path to your source location)
cp /Users/kyle/hub/propaganda/t5/bias-detector-output/* models/bias-detector-output/

# Verify files
ls -la models/bias-detector-output/
# Should show:
#   adapter_config.json
#   adapter_model.safetensors (~18MB)
#   tokenizer.json
#   tokenizer_config.json
```

---

## Step 3: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip
```

---

## Step 4: Install the Package

### Option A: Install from PyPI (After Publishing)

```bash
pip install bias-mcp-server
```

### Option B: Install from Local Source (Build Wheel)

From the source repository:
```bash
# In source directory
cd /Users/kyle/hub/propaganda/t5/mcp_bias_server
pip install build
python -m build
# Creates dist/bias_mcp_server-0.1.2-py3-none-any.whl
```

Copy wheel to new location:
```bash
# Copy wheel to installation directory
cp dist/bias_mcp_server-0.1.2-py3-none-any.whl ~/bias-detector/
```

Install in new location:
```bash
cd ~/bias-detector
pip install bias_mcp_server-0.1.2-py3-none-any.whl
```

### Option C: Install Directly from Git

```bash
pip install git+https://github.com/yourusername/bias-mcp-server.git
```

---

## Step 5: Create Test Script

Create `test_bias.py`:

```python
#!/usr/bin/env python3
"""Test bias detection installation."""

import os
import sys

# Set model path (relative to this script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "bias-detector-output")
os.environ["BIAS_MODEL_PATH"] = model_path

from mcp_bias_server.bias_engine import BiasEngine

def main():
    print("=" * 60)
    print("BIAS DETECTOR - INSTALLATION TEST")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print()
    
    # Check model files
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required:
        path = os.path.join(model_path, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[OK] {f} ({size:,} bytes)")
        else:
            print(f"[FAIL] Missing: {f}")
            sys.exit(1)
    
    print()
    print("Loading model (first load takes 10-30 seconds)...")
    
    # Test analysis
    engine = BiasEngine()
    result = engine.analyze("The president announced new economic policies today.")
    
    print("[OK] Analysis complete")
    print()
    print(f"Direction: L={result.direction['L']:.2f} C={result.direction['C']:.2f} R={result.direction['R']:.2f}")
    print(f"Degree:    L={result.degree['L']:.2f} M={result.degree['M']:.2f} H={result.degree['H']:.2f}")
    print()
    print("=" * 60)
    print("INSTALLATION SUCCESSFUL!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Step 6: Create stdin Analyzer

Create `bias_stdin.py`:

```python
#!/usr/bin/env python3
"""Analyze text from stdin, output JSON to stdout."""

import os
import sys
import json

# Set model path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["BIAS_MODEL_PATH"] = os.path.join(script_dir, "models", "bias-detector-output")

from mcp_bias_server.bias_engine import BiasEngine

# Read from stdin
text = sys.stdin.read().strip()
if not text:
    print("Error: No input", file=sys.stderr)
    sys.exit(1)

# Analyze
engine = BiasEngine()
result = engine.analyze(text)

# Output JSON
print(json.dumps({
    "direction": result.direction,
    "degree": result.degree,
    "reason": result.reason
}, indent=2))
```

---

## Step 7: Run Tests

```bash
# Activate venv
source venv/bin/activate

# Run test script
python test_bias.py

# Test stdin analyzer
echo "The president announced new policies." | python bias_stdin.py

# Analyze a file
cat some_article.txt | python bias_stdin.py
```

---

## Directory Structure After Installation

```
~/bias-detector/
|-- venv/                          # Virtual environment
|-- models/
|   `-- bias-detector-output/      # Trained model weights
|       |-- adapter_config.json
|       |-- adapter_model.safetensors
|       |-- tokenizer.json
|       `-- tokenizer_config.json
|-- test_bias.py                   # Installation test
|-- bias_stdin.py                  # stdin analyzer
`-- bias_mcp_server-0.1.2-py3-none-any.whl  # (if installed from wheel)
```

---

## Environment Variables (Optional)

For permanent configuration, add to `~/.bashrc` or `~/.zshrc`:

```bash
export BIAS_MODEL_PATH=~/bias-detector/models/bias-detector-output
export BIAS_DEVICE=auto  # or: mps, cuda, cpu
```

---

## MCP Server Configuration (VSCode)

Edit MCP settings:

```bash
code ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

Add:

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "/Users/YOURUSERNAME/bias-detector/venv/bin/python",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/Users/YOURUSERNAME/bias-detector/models/bias-detector-output"
      }
    }
  }
}
```

**Important:** Use absolute paths and the venv's Python interpreter.

---

## Quick Commands

| Task | Command |
|------|---------|
| Activate | `source ~/bias-detector/venv/bin/activate` |
| Test | `python ~/bias-detector/test_bias.py` |
| Analyze | `echo "text" \| python ~/bias-detector/bias_stdin.py` |
| Analyze file | `cat file.txt \| python ~/bias-detector/bias_stdin.py` |

---

## Troubleshooting

### Import Error
```bash
pip uninstall bias-mcp-server
pip install bias-mcp-server  # or reinstall wheel
```

### Model Not Found
```bash
ls ~/bias-detector/models/bias-detector-output/adapter_model.safetensors
```

### Wrong Python
```bash
which python  # Should point to venv
# If not:
source ~/bias-detector/venv/bin/activate
```

---

## Summary

1. Create new directory
2. Copy model weights from source
3. Create and activate venv
4. Install package (PyPI, wheel, or git)
5. Create test scripts
6. Run tests
7. Configure MCP (optional)