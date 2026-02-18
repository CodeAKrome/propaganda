# T5 Bias Detection - Fresh Install & Test Guide

Complete guide for installing and testing the T5 bias detection system from scratch.

---

## Prerequisites

- Python 3.10+ 
- Virtual environment created
- Git (for cloning if needed)

---

## Step 1: Create & Activate Virtual Environment

```bash
# Create venv (if not already created)
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate   # Windows

# Verify Python version
python --version  # Should be 3.10+
```

---

## Step 2: Install the Package

### Method A: Install from Source (Recommended for Development)

```bash
cd t5/mcp_bias_server
pip install -e .
```

### Method B: Install from PyPI (After Publishing)

```bash
pip install bias-mcp-server
```

### Method C: Install with Dev Dependencies

```bash
cd t5/mcp_bias_server
pip install -e ".[dev]"
```

---

## Step 3: Verify Installation

```bash
# Test import
python -c "from mcp_bias_server.bias_engine import BiasEngine; print('Import: OK')"

# Check command availability
which bias-mcp-server
# or
bias-mcp-server --help
```

---

## Step 4: Set Environment Variables

```bash
# Required: Path to trained model weights
export BIAS_MODEL_PATH=/Users/kyle/hub/propaganda/t5/bias-detector-output

# Optional: Base model (default: t5-large)
export BIAS_BASE_MODEL=t5-large

# Optional: Device (default: auto)
export BIAS_DEVICE=auto  # Options: auto, mps, cuda, cpu
```

---

## Step 5: Run Tests

### Test 1: Installation Test Script

```bash
cd /Users/kyle/hub/propaganda
python t5/mcp_bias_server/test_bias_server.py
```

**Expected Output:**
```
[PASS] Module Import
[PASS] Model Files
[PASS] Single Analysis
[PASS] Batch Analysis
[PASS] Model Info
Total: 5/5 tests passed
```

### Test 2: stdin/stdout Script

```bash
# JSON output
echo "The president announced new policies today." | python t5/mcp_bias_server/bias_stdin.py

# Text output
echo "Your text" | python t5/mcp_bias_server/bias_stdin.py --format=text

# Quiet mode
echo "Your text" | python t5/mcp_bias_server/bias_stdin.py --quiet
```

### Test 3: Article Analyzer

```bash
python t5/mcp_bias_server/analyze_article.py
```

### Test 4: Python Interactive

```bash
python << 'EOF'
import os
os.environ["BIAS_MODEL_PATH"] = "/Users/kyle/hub/propaganda/t5/bias-detector-output"

from mcp_bias_server.bias_engine import BiasEngine

engine = BiasEngine()
result = engine.analyze("The senator proposed bipartisan legislation.")

print(f"Direction: {result.direction}")
print(f"Degree: {result.degree}")
print(f"Reason: {result.reason[:100]}...")
EOF
```

### Test 5: Batch Analysis

```bash
python << 'EOF'
import os
os.environ["BIAS_MODEL_PATH"] = "/Users/kyle/hub/propaganda/t5/bias-detector-output"

from mcp_bias_server.bias_engine import BiasEngine

texts = [
    "Markets rallied on positive economic news.",
    "The radical left continues their extreme agenda.",
    "The president signed the bill into law today.",
]

engine = BiasEngine()
results = engine.analyze_batch(texts)

for i, (text, result) in enumerate(zip(texts, results), 1):
    print(f"[{i}] L={result.direction['L']:.1f} C={result.direction['C']:.1f} R={result.direction['R']:.1f}")
EOF
```

---

## Step 6: MCP Server Configuration (VSCode)

### Configure MCP Settings

Edit the MCP settings file:

**macOS:**
```bash
code ~/Library/Application\ Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

**Add this configuration:**
```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/Users/kyle/hub/propaganda/t5/bias-detector-output",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

### Restart VSCode

```bash
# macOS
killall "Visual Studio Code"
code
```

### Test MCP in VSCode

In Kilo Code chat:
```
Use analyze_bias to check: "The president announced new policies."
```

---

## Step 7: Run Unit Tests (If Installed with Dev Dependencies)

```bash
cd t5/mcp_bias_server
pytest tests/ -v
```

---

## Troubleshooting

### Import Error
```bash
# Reinstall
pip uninstall bias-mcp-server
pip install -e .
```

### Model Not Found
```bash
# Check path
ls $BIAS_MODEL_PATH/adapter_config.json
ls $BIAS_MODEL_PATH/adapter_model.safetensors
```

### Device Error (MPS/CUDA)
```bash
# Force CPU
export BIAS_DEVICE=cpu
```

### Slow First Inference
Normal - model loads lazily on first use (10-30 seconds).

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -e t5/mcp_bias_server` |
| Test import | `python -c "from mcp_bias_server.bias_engine import BiasEngine"` |
| Run tests | `python t5/mcp_bias_server/test_bias_server.py` |
| Analyze text | `echo "text" \| python t5/mcp_bias_server/bias_stdin.py` |
| Analyze file | `cat file.txt \| python t5/mcp_bias_server/bias_stdin.py` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `test_bias_server.py` | Full installation test (5 tests) |
| `bias_stdin.py` | stdin/stdout analyzer |
| `analyze_article.py` | File-based analyzer |
| `examples/example_python_usage.py` | Python API examples |