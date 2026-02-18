# Cline Bias Server

<p align="center">
  <a href="https://pypi.org/project/cline-bias-server/">
    <img src="https://img.shields.io/pypi/v/cline-bias-server.svg" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/cline-bias-server/">
    <img src="https://img.shields.io/pypi/pyversions/cline-bias-server.svg" alt="Python Versions">
  </a>
  <a href="https://github.com/yourusername/cline-bias-server/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/cline-bias-server.svg" alt="License">
  </a>
  <a href="https://github.com/yourusername/cline-bias-server/actions">
    <img src="https://github.com/yourusername/cline-bias-server/actions/workflows/test.yml/badge.svg" alt="Tests">
  </a>
</p>

A comprehensive Model Context Protocol (MCP) server for political bias detection using fine-tuned T5 models with LoRA adapters.

## Features

- **Political Bias Detection**: Analyze text for political leanings (Left, Center, Right)
- **Bias Intensity Analysis**: Detect the intensity of bias (Low, Medium, High)
- **MCP Protocol**: Full MCP server implementation with tools, resources, and prompts
- **Multiple Output Formats**: JSON, text, and compact output formats
- **Batch Processing**: Analyze multiple texts efficiently
- **Resource Management**: Automatic model loading/unloading for memory efficiency
- **Hardware Support**: MPS (Apple Silicon), CUDA, and CPU backends
- **Easy Integration**: Python API, CLI, and MCP server modes

## Installation

### From PyPI (Recommended)

```bash
pip install cline-bias-server
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/cline-bias-server.git
cd cline-bias-server

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Hardware-Specific Installation

```bash
# For Apple Silicon (MPS)
pip install cline-bias-server

# For NVIDIA GPU (CUDA)
pip install cline-bias-server
# Then install PyTorch with CUDA support separately
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Command Line

```bash
# Analyze text
cline-bias "The president announced new economic policies today that will benefit working families."

# Analyze a file
cline-bias --file article.txt

# Batch analysis
cline-bias --batch article1.txt article2.txt article3.txt

# Human-readable output
cline-bias --format text "Your text here"

# Run as MCP server
cline-bias --server
```

### Python API

```python
from cline_bias_server import quick_analyze, BiasEngine

# Quick one-liner
result = quick_analyze("Your text here")
print(result['dominant_direction'])  # 'Left', 'Center', or 'Right'
print(result['dominant_degree'])       # 'Low', 'Medium', or 'High'

# Or use the engine directly for more control
engine = BiasEngine()
result = engine.analyze("Your text here")
print(result.direction)   # {'L': 0.1, 'C': 0.2, 'R': 0.7}
print(result.degree)      # {'L': 0.1, 'M': 0.3, 'H': 0.6}
print(result.reason)       # Explanation of the classification
```

### MCP Server

The server implements the MCP protocol and can be used with any MCP-compatible client:

```json
// MCP Client configuration example
{
  "mcpServers": {
    "cline-bias-server": {
      "command": "cline-bias-server",
      "args": []
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `analyze_bias` | Analyze single text for political bias |
| `analyze_batch` | Analyze multiple texts with statistics |
| `analyze_file` | Analyze a text file for bias |
| `compare_bias` | Compare bias between multiple texts |
| `get_model_info` | Get information about the model |
| `unload_model` | Unload model from memory |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIAS_MODEL_PATH` | `./bias-detector-output` | Path to LoRA adapter |
| `BIAS_BASE_MODEL` | `t5-large` | Base T5 model name |
| `BIAS_DEVICE` | `auto` | Device to use (auto/mps/cuda/cpu) |
| `BIAS_USE_HUGGINGFACE` | `1` | Auto-download from HuggingFace |

## Output Format

The bias analysis returns the following structure:

```json
{
  "direction": {
    "L": 0.15,
    "C": 0.25,
    "R": 0.60
  },
  "degree": {
    "L": 0.10,
    "M": 0.35,
    "H": 0.55
  },
  "reason": "The article uses language that typically appears in conservative media...",
  "dominant_direction": "Right",
  "dominant_degree": "High",
  "direction_percent": {
    "L": 15.0,
    "C": 25.0,
    "R": 60.0
  },
  "degree_percent": {
    "L": 10.0,
    "M": 35.0,
    "H": 55.0
  },
  "confidence": 0.78,
  "device": "cuda"
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cline_bias_server --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy cline_bias_server
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

## Acknowledgments

- [T5](https://huggingface.co/t5-large) - Google's Text-to-Text Transfer Transformer
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [MCP](https://modelcontextprotocol.io) - Model Context Protocol
