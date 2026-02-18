# Cline Bias Server - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [MCP Server Implementation](#mcp-server-implementation)
4. [API Reference](#api-reference)
5. [MCP Publishing Guidelines](#mcp-publishing-guidelines)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Cline Bias Server is a comprehensive Model Context Protocol (MCP) server for political bias detection. It uses a fine-tuned T5-large model with LoRA adapters to analyze text for political bias.

### Key Capabilities

- **Political Direction Detection**: Identifies Left, Center, or Right leanings
- **Bias Intensity Analysis**: Measures Low, Medium, or High bias intensity
- **Confidence Scoring**: Provides confidence metrics for predictions
- **Batch Processing**: Efficiently processes multiple texts
- **Comparison Tools**: Compare bias across multiple sources

---

## Architecture

```
cline-bias-server/
├── src/
│   └── cline_bias_server/
│       ├── __init__.py         # Package entry point
│       ├── bias_engine.py      # Core ML engine
│       ├── server.py           # MCP server implementation
│       ├── cli.py              # Command-line interface
│       └── bias-detector-output/  # LoRA adapter weights
├── pyproject.toml              # Package configuration
├── README.md                   # Quick start guide
└── LICENSE                     # MIT license
```

### Components

#### BiasEngine
The core ML engine that handles:
- Model loading and management
- T5 inference
- JSON output parsing and repair
- Result normalization

#### BiasServer
MCP server implementation providing:
- Tool handlers
- Resource endpoints
- Prompt templates

#### BiasAnalyzer
High-level utilities for:
- Batch processing
- Directory scanning
- Statistical analysis

---

## MCP Server Implementation

### MCP Protocol Features

The server implements the full MCP specification:

#### Tools

```python
# List of available MCP tools
tools = [
    "analyze_bias",      # Single text analysis
    "analyze_batch",     # Multiple text analysis
    "analyze_file",      # File analysis
    "compare_bias",      # Compare multiple texts
    "get_model_info",    # Model metadata
    "unload_model"      # Memory management
]
```

#### Resources

```
bias://model-info          # Model information
bias://schemas/result     # Result JSON schema
```

#### Prompts

```
analyze-article    # Article analysis prompt
compare-sources    # Source comparison prompt
```

### MCP Server Lifecycle

```python
# Server initialization
server = BiasServer(
    model_path="./bias-detector-output",
    base_model="t5-large",
    device="auto"
)

# Run server
await server.run()
```

---

## API Reference

### Python API

#### quick_analyze(text: str) -> dict

One-line bias analysis.

```python
from cline_bias_server import quick_analyze

result = quick_analyze("The president announced new policies...")
print(result['dominant_direction'])  # 'Right'
```

#### BiasEngine

Main class for bias detection.

```python
engine = BiasEngine(
    model_path=None,           # Auto-resolved
    base_model_name="t5-large",
    device=None,               # Auto-detect
    lazy_load=True,            # Deferred loading
    use_huggingface=True      # Auto-download
)

result = engine.analyze("text")
```

#### BiasResult

Result dataclass.

```python
@dataclass
class BiasResult:
    direction: Dict[str, float]  # {"L": 0.1, "C": 0.2, "R": 0.7}
    degree: Dict[str, float]    # {"L": 0.1, "M": 0.3, "H": 0.6}
    reason: str                 # Explanation
    raw_output: Optional[str]   # Raw model output
    device: str                 # "cuda", "mps", "cpu"
    confidence: float           # 0.0 - 1.0
```

### CLI API

```bash
# Analyze text
cline-bias "Your text"

# File analysis
cline-bias --file article.txt

# Batch analysis
cline-bias --batch file1.txt file2.txt

# Server mode
cline-bias --server

# Specific device
cline-bias --device cuda "text"
```

---

## MCP Publishing Guidelines

This section documents how to publish the MCP server package to PyPI and make it available for MCP clients.

### Package Structure Requirements

For MCP servers to work properly with various clients, ensure:

1. **Entry Points**: Define CLI and server entry points in `pyproject.toml`:

```toml
[project.scripts]
cline-bias-server = "cline_bias_server.server:run_server"
cline-bias = "cline_bias_server.cli:main"

[project.entry-points."mcp.server"]
cline-bias-server = "cline_bias_server.server:create_server"
```

2. **Dependencies**: Include all required dependencies:

```toml
dependencies = [
    "mcp>=1.0.0",
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "peft>=0.7.0",
    "accelerate>=0.24.0",
]
```

3. **Python Version**: Support Python 3.10+:

```toml
requires-python = ">=3.10"
```

### Publishing to PyPI

#### Step 1: Prepare for Publishing

```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info
```

#### Step 2: Build the Package

```bash
# Build source and wheel
python -m build
```

#### Step 3: Upload to PyPI

```bash
# Test PyPI (recommended first)
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

#### Step 4: Verify Installation

```bash
pip install cline-bias-server
cline-bias --version
```

### MCP Client Integration

#### VS Code (Cline)

Add to `.vscode/mcp_settings.json`:

```json
{
  "mcpServers": {
    "cline-bias-server": {
      "command": "cline-bias-server",
      "env": {
        "BIAS_MODEL_PATH": "./bias-detector-output"
      }
    }
  }
}
```

#### Custom MCP Client

```python
from mcp import Client

async with Client("cline-bias-server") as client:
    result = await client.call_tool("analyze_bias", {
        "text": "Your text here"
    })
    print(result)
```

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BIAS_MODEL_PATH` | string | `./bias-detector-output` | Path to LoRA adapter |
| `BIAS_BASE_MODEL` | string | `t5-large` | Base T5 model |
| `BIAS_DEVICE` | string | `auto` | Force device (mps/cuda/cpu) |
| `BIAS_USE_HUGGINGFACE` | string | `1` | Auto-download from HF |

### Configuration File

Create `bias_config.json`:

```json
{
  "model_path": "./bias-detector-output",
  "base_model": "t5-large",
  "device": "auto",
  "max_input_length": 512,
  "max_output_length": 512,
  "num_beams": 4,
  "early_stopping": true
}
```

### Programmatic Configuration

```python
from cline_bias_server import BiasEngine

engine = BiasEngine(
    model_path="./custom-model",
    base_model_name="t5-base",
    device="cuda",
    max_input_length=256,
    max_output_length=256
)
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `FileNotFoundError: Model path not found`

**Solution**:
```bash
# Set model path
export BIAS_MODEL_PATH=/path/to/model

# Or use HuggingFace
export BIAS_USE_HUGGINGFACE=1
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Use CPU
export BIAS_DEVICE=cpu

# Or reduce batch size in code
engine = BiasEngine(device="cpu")
```

#### 3. MCP Server Not Starting

**Error**: `ModuleNotFoundError: No module named 'mcp'`

**Solution**:
```bash
pip install mcp
```

#### 4. Slow Inference

**Solution**:
- Use GPU (CUDA or MPS)
- Reduce `max_input_length`
- Use smaller base model (t5-base instead of t5-large)

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from cline_bias_server import BiasEngine
engine = BiasEngine()
```

### Getting Help

- GitHub Issues: https://github.com/yourusername/cline-bias-server/issues
- Documentation: https://github.com/yourusername/cline-bias-server#readme

---

## Appendix

### Output Schema

```json
{
  "type": "object",
  "properties": {
    "direction": {
      "type": "object",
      "properties": {
        "L": {"type": "number"},
        "C": {"type": "number"},
        "R": {"type": "number"}
      }
    },
    "degree": {
      "type": "object",
      "properties": {
        "L": {"type": "number"},
        "M": {"type": "number"},
        "H": {"type": "number"}
      }
    },
    "reason": {"type": "string"},
    "dominant_direction": {"type": "string"},
    "dominant_degree": {"type": "string"},
    "confidence": {"type": "number"}
  }
}
```

### Model Information

- **Base Model**: T5-large (800M parameters)
- **Fine-tuning**: LoRA adapters (3M trainable parameters)
- **Training Data**: News articles with political bias labels
- **Input**: Text up to 512 tokens
- **Output**: JSON with direction, degree, and reasoning

### Performance

- **Inference Speed** (T5-large, A100):
  - Single text: ~200ms
  - Batch (10 texts): ~800ms
- **Memory Usage**:
  - Model: ~3GB
  - With LoRA: ~3.5GB

---

*Last Updated: 2026-02-17*
*Version: 1.0.0*
