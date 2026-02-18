# Bias MCP Server

[![PyPI version](https://badge.fury.io/py/bias-mcp-server.svg)](https://badge.fury.io/py/bias-mcp-server)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green.svg)](https://modelcontextprotocol.io/)

A **Model Context Protocol (MCP)** server for political bias detection using a fine-tuned T5-large model with LoRA adapters. This server provides tools for analyzing news articles, opinion pieces, and political commentary for political leaning.

## Features

- **Political Bias Detection**: Analyzes text for Left/Center/Right political leaning
- **Bias Intensity**: Measures Low/Medium/High degree of bias
- **Reasoning**: Provides explanations for classifications
- **Hardware Acceleration**: Supports MPS (Apple Silicon), CUDA, and CPU
- **MCP Integration**: Works with AI assistants that support the Model Context Protocol
- **Batch Processing**: Analyze multiple texts efficiently

## Installation

### From PyPI (Recommended)

```bash
pip install bias-mcp-server
```

**That's it!** The package includes the trained LoRA adapter (~18MB), so it works out of the box.

### Quick Start

```python
from mcp_bias_server.bias_engine import BiasEngine

# Works immediately - no additional setup needed
engine = BiasEngine()
result = engine.analyze("Your news text here...")
print(result.to_dict())
```

### From Source

```bash
git clone https://github.com/yourusername/bias-mcp-server.git
cd bias-mcp-server
pip install -e .
```

## Usage

### As an MCP Server

Add to your MCP settings configuration:

**macOS/Linux:**
```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "bias-mcp-server",
      "env": {
        "BIAS_MODEL_PATH": "/path/to/bias-detector-output",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

**Configuration Locations:**
- **Kilo Code**: `~/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json`
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Command Line

```bash
# Run the MCP server directly
bias-mcp-server

# Or with Python module
python -m mcp_bias_server
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BIAS_MODEL_PATH` | Path to LoRA adapter weights | `./bias-detector-output` |
| `BIAS_BASE_MODEL` | HuggingFace base model name | `t5-large` |
| `BIAS_DEVICE` | Device selection: `auto`, `mps`, `cuda`, `cpu` | `auto` |

## MCP Tools

### `analyze_bias`

Analyze a single text for political bias.

**Parameters:**
- `text` (string, required): The text to analyze

**Example:**
```json
{
  "name": "analyze_bias",
  "arguments": {
    "text": "The president announced new economic policies today, promising to reduce inflation and create jobs through targeted tax incentives for middle-class families."
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "dir": {
      "L": 0.25,
      "C": 0.50,
      "R": 0.25
    },
    "deg": {
      "L": 0.60,
      "M": 0.30,
      "H": 0.10
    },
    "reason": "The article presents policy announcements in a neutral tone without strong partisan language.",
    "device": "mps"
  }
}
```

### `analyze_batch`

Analyze multiple texts in a single call.

**Parameters:**
- `texts` (array of strings, required): List of texts to analyze

**Example:**
```json
{
  "name": "analyze_batch",
  "arguments": {
    "texts": [
      "First article text...",
      "Second article text..."
    ]
  }
}
```

### `get_model_info`

Get information about the loaded model.

**Example:**
```json
{
  "name": "get_model_info",
  "arguments": {}
}
```

**Response:**
```json
{
  "success": true,
  "model_info": {
    "base_model": "t5-large",
    "adapter_path": "./bias-detector-output",
    "device": "mps",
    "model_type": "T5 with LoRA",
    "max_input_length": 512,
    "max_output_length": 512,
    "supported_languages": ["English"],
    "output_format": {
      "dir": "Political direction scores (L=Left, C=Center, R=Right)",
      "deg": "Bias degree scores (L=Low, M=Medium, H=High)",
      "reason": "Explanation of the classification"
    }
  }
}
```

## Output Format

### Direction Scores (`dir`)

| Key | Meaning | Description |
|-----|---------|-------------|
| `L` | Left | Liberal/progressive political leaning |
| `C` | Center | Neutral/balanced perspective |
| `R` | Right | Conservative political leaning |

Values are probabilities that sum to approximately 1.0.

### Degree Scores (`deg`)

| Key | Meaning | Description |
|-----|---------|-------------|
| `L` | Low | Minimal bias, mostly factual |
| `M` | Medium | Moderate bias, some framing |
| `H` | High | Strong bias, clear partisan stance |

Values are probabilities that sum to approximately 1.0.

## Python API Usage

You can also use the BiasEngine directly from Python code:

```python
from mcp_bias_server.bias_engine import BiasEngine

# Create the engine
engine = BiasEngine(
    model_path="./bias-detector-output",
    base_model_name="t5-large",
    lazy_load=True
)

# Analyze a single text
result = engine.analyze("Your text here...")
print(f"Direction: {result.direction}")  # {'L': 0.25, 'C': 0.50, 'R': 0.25}
print(f"Degree: {result.degree}")        # {'L': 0.60, 'M': 0.30, 'H': 0.10}
print(f"Reason: {result.reason}")

# Batch analysis
results = engine.analyze_batch(["Text 1...", "Text 2..."])
```

See [`examples/example_python_usage.py`](examples/example_python_usage.py) for a complete example.

## Hardware Support

### Apple Silicon (MPS)

Best performance on M1/M2/M3 Macs:

```bash
export BIAS_DEVICE=mps
```

### NVIDIA GPU (CUDA)

For systems with CUDA-capable GPUs:

```bash
export BIAS_DEVICE=cuda
```

### CPU

Fallback for systems without GPU acceleration:

```bash
export BIAS_DEVICE=cpu
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/bias-mcp-server.git
cd bias-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black mcp_bias_server

# Lint
ruff check mcp_bias_server

# Type check
mypy mcp_bias_server
```

## Publishing

### Step 1: Build the Package

```bash
cd t5/mcp_bias_server

# Install build tool
pip install build

# Build wheel and source distribution
python -m build

# Output: dist/bias_mcp_server-0.1.0-py3-none-any.whl
#         dist/bias_mcp_server-0.1.0.tar.gz
```

### Step 2: Test the Build Locally

```bash
# Install the built wheel
pip install dist/bias_mcp_server-0.1.0-py3-none-any.whl

# Verify it works
python -c "from mcp_bias_server.bias_engine import BiasEngine; print('OK')"
```

### Step 3: Create PyPI Account & API Token

1. Create account at https://pypi.org/account/register/
2. Go to Account Settings → API tokens
3. Create new API token (scope: "Entire account" for first upload)
4. Save the token (starts with `pypi-`)

### Step 4: Configure Twine with API Token

```bash
# Install twine
pip install twine

# Create .pypirc with your token
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
EOF

# Secure the file
chmod 600 ~/.pypirc
```

### Step 5: Upload to TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ bias-mcp-server
```

### Step 6: Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Users can now install with:
pip install bias-mcp-server
```

### Step 7: Tag Release in Git

```bash
git tag v0.1.0
git push origin v0.1.0
```

### Alternative: GitHub Actions (Automated Publishing)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install build twine
      - name: Build
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Then add `PYPI_API_TOKEN` to GitHub repository secrets.

## Installation for Users

### Option 1: Install from PyPI

```bash
pip install bias-mcp-server
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/yourusername/bias-mcp-server.git
```

### Option 3: Install from Source

```bash
git clone https://github.com/yourusername/bias-mcp-server.git
cd bias-mcp-server
pip install -e .
```

## MCP Server Configuration for Users

After installing, users need to:

### 1. Download the LoRA Model

The trained LoRA adapter must be available. Users can:
- Copy from your repository: `t5/bias-detector-output/`
- Or train their own model

### 2. Configure MCP in VSCode

Add to `~/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json`:

```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "python3",
      "args": ["-m", "mcp_bias_server.server"],
      "env": {
        "BIAS_MODEL_PATH": "/path/to/bias-detector-output",
        "BIAS_BASE_MODEL": "t5-large",
        "BIAS_DEVICE": "auto"
      }
    }
  }
}
```

### 3. Verify Installation

```bash
# Test the package
python3 -c "
from mcp_bias_server.bias_engine import BiasEngine
import os
os.environ['BIAS_MODEL_PATH'] = '/path/to/bias-detector-output'
engine = BiasEngine()
print('Model loaded successfully')
"
```

## Architecture

```
t5/mcp_bias_server/
├── pyproject.toml           # Package configuration
├── README.md                # This file
├── LICENSE
├── src/mcp_bias_server/
│   ├── __init__.py          # Package initialization
│   ├── server.py            # MCP server implementation
│   └── bias_engine.py       # T5 model wrapper
├── examples/
│   └── example_python_usage.py
├── tests/
│   └── test_bias_engine.py
├── build_package.sh         # Build wheel
├── publish_testpypi.sh      # Publish to TestPyPI
├── publish_pypi.sh          # Publish to PyPI
└── install_mcp_vscode.sh    # VSCode setup script
```

### Model Architecture

- **Base Model**: T5-large (770M parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Text-to-text generation for bias classification
- **Input**: News article text (max 512 tokens)
- **Output**: JSON with direction, degree, and reasoning

## Ollama Integration

For systems without the trained T5 model, you can use Ollama for inference:

```bash
# Install Ollama
brew install ollama  # macOS
ollama serve

# Pull a model
ollama pull llama3.2

# Run the demo
python demo_ollama_bias.py
```

See [`demo_ollama_bias.py`](demo_ollama_bias.py) for the complete implementation.

## Full Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for:
- Complete API reference
- Docker deployment guide
- Troubleshooting guide
- All configuration options
- More examples

## Limitations

1. **Language**: Currently supports English text only
2. **Domain**: Trained primarily on news articles; may not perform well on other text types
3. **Context**: Limited to 512 tokens; longer texts are truncated
4. **Subjectivity**: Bias detection is inherently subjective and may not align with all perspectives

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main repository.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for the T5 model
- [PEFT](https://huggingface.co/docs/peft/) for LoRA adapter support
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification

## Changelog

### v0.1.0 (2024-01-XX)

- Initial release
- MCP server implementation with `analyze_bias`, `analyze_batch`, `get_model_info` tools
- Support for MPS, CUDA, and CPU devices
- pip-installable package