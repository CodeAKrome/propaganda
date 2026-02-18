# Bias MCP Server - Complete Documentation

## Overview

The Bias MCP Server is a comprehensive political bias detection system that provides multiple inference backends and integration options. It can detect political leaning (Left/Center/Right) and bias intensity (Low/Medium/High) in news articles and political commentary.

## Table of Contents

1. [Installation](#installation)
2. [Inference Backends](#inference-backends)
3. [MCP Server Integration](#mcp-server-integration)
4. [Python API](#python-api)
5. [Ollama Integration](#ollama-integration)
6. [Docker Deployment](#docker-deployment)
7. [Configuration](#configuration)
8. [Output Format](#output-format)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: PyPI Installation (Recommended)

```bash
pip install bias-mcp-server
```

### Option 2: From Source

```bash
git clone https://github.com/yourusername/bias-mcp-server.git
cd bias-mcp-server
pip install -e .
```

### Option 3: Automated VSCode Setup

```bash
./install_mcp_vscode.sh --model-path /path/to/bias-detector-output
```

### Dependencies

The package requires:
- Python 3.10+
- PyTorch
- Transformers
- PEFT (for LoRA adapters)

For Ollama backend:
```bash
pip install ollama
```

---

## Inference Backends

The system supports three inference backends:

### 1. T5-LoRA (Fine-tuned Model)

The primary backend uses a fine-tuned T5-large model with LoRA adapters.

**Pros:**
- Most accurate for news bias detection
- Consistent, reproducible results
- Fast inference on GPU/MPS

**Cons:**
- Requires trained model weights (~1GB)
- Higher memory usage

**Usage:**
```bash
export BIAS_MODEL_PATH=/path/to/bias-detector-output
export BIAS_DEVICE=mps  # or cuda, cpu
bias-mcp-server
```

### 2. Ollama (Local LLM)

Uses Ollama to run local LLMs for bias detection.

**Pros:**
- No trained model required
- Flexible model selection
- Works with any Ollama-compatible model

**Cons:**
- Less consistent results
- Slower inference
- Requires Ollama installation

**Usage:**
```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2

# Run the demo
python demo_ollama_bias.py
```

### 3. CPU-Only Docker

For deployment without GPU acceleration.

**Usage:**
```bash
cd t5/docker-inference
docker-compose -f docker-compose.yml up
```

---

## MCP Server Integration

### What is MCP?

The Model Context Protocol (MCP) is a standard for connecting AI assistants to external tools. This server implements MCP to provide bias analysis tools to any MCP-compatible client.

### Available Tools

#### `analyze_bias`

Analyze a single text for political bias.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| text | string | Yes | The text to analyze |

**Example:**
```json
{
  "name": "analyze_bias",
  "arguments": {
    "text": "The president announced new economic policies today..."
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "dir": {"L": 0.25, "C": 0.50, "R": 0.25},
    "deg": {"L": 0.60, "M": 0.30, "H": 0.10},
    "reason": "The article presents policy announcements in a neutral tone...",
    "device": "mps"
  }
}
```

#### `analyze_batch`

Analyze multiple texts in a single call.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| texts | array | Yes | List of texts to analyze |

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

#### `get_model_info`

Get information about the loaded model.

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

### VSCode (Kilo Code) Configuration

**macOS:**
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

**Config file location:**
```
~/Library/Application Support/Code/User/globalStorage/kilocode.kilo-code/settings/mcp_settings.json
```

### Claude Desktop Configuration

**macOS:**
```json
{
  "mcpServers": {
    "bias-analyzer": {
      "command": "bias-mcp-server",
      "env": {
        "BIAS_MODEL_PATH": "/path/to/bias-detector-output"
      }
    }
  }
}
```

**Config file location:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

---

## Python API

### Direct Usage (T5-LoRA)

```python
from mcp_bias_server.bias_engine import BiasEngine

# Initialize the engine
engine = BiasEngine(
    model_path="./bias-detector-output",
    base_model_name="t5-large",
    lazy_load=True  # Load model on first inference
)

# Analyze a single text
result = engine.analyze("Your text here...")

print(f"Direction: {result.direction}")
# Output: {'L': 0.25, 'C': 0.50, 'R': 0.25}

print(f"Degree: {result.degree}")
# Output: {'L': 0.60, 'M': 0.30, 'H': 0.10}

print(f"Reason: {result.reason}")
# Output: "The article presents..."

# Batch analysis
results = engine.analyze_batch(["Text 1...", "Text 2..."])
for r in results:
    print(r.direction, r.degree)

# Get model info
info = engine.get_model_info()
print(info)
```

### Ollama Usage

```python
from demo_ollama_bias import OllamaBiasAnalyzer

# Initialize with your preferred model
analyzer = OllamaBiasAnalyzer(model="llama3.2")

# Check connection
if analyzer.check_ollama():
    # Analyze text
    result = analyzer.analyze("Your text here...")
    print(result.direction)  # {'L': 0.3, 'C': 0.4, 'R': 0.3}
    print(result.degree)     # {'L': 0.5, 'M': 0.3, 'H': 0.2}
    print(result.reason)     # "Explanation..."
```

---

## Ollama Integration

### Setup

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Pull a Model:**
   ```bash
   # Recommended models for bias detection
   ollama pull llama3.2      # Fast, good quality
   ollama pull llama3.1:70b  # Higher quality, slower
   ollama pull mistral       # Alternative option
   ```

4. **Run the Demo:**
   ```bash
   python demo_ollama_bias.py
   ```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Model to use for inference | `llama3.2` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

### Example Output

```
Text 1:
  "The president announced today a sweeping new economic policy..."
  
  Direction:
    Left (L):   0.30
    Center (C): 0.45
    Right (R):  0.25
    
  Degree:
    Low (L):    0.55
    Medium (M): 0.30
    High (H):   0.15
    
  Reason: The article presents policy announcements with balanced 
  framing, including both supporters and critics.
```

---

## Docker Deployment

### CPU-Only Deployment

```bash
cd t5/docker-inference

# Build and run
docker-compose -f docker-compose.yml up

# The API will be available at http://localhost:8000
```

### Apple Silicon (MPS) Deployment

```bash
cd t5/docker-inference

# Build and run with MPS support
docker-compose -f docker-compose.yml -f docker-compose.mps.yml up
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze single text |
| `/analyze/batch` | POST | Analyze multiple texts |
| `/model/info` | GET | Get model information |
| `/health` | GET | Health check |

### Example API Usage

```bash
# Single analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'

# Batch analysis
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1...", "Text 2..."]}'
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BIAS_MODEL_PATH` | Path to LoRA adapter weights | `./bias-detector-output` |
| `BIAS_BASE_MODEL` | HuggingFace base model name | `t5-large` |
| `BIAS_DEVICE` | Device selection | `auto` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

### Device Selection

| Value | Description |
|-------|-------------|
| `auto` | Automatically select best available (MPS > CUDA > CPU) |
| `mps` | Apple Silicon GPU |
| `cuda` | NVIDIA GPU |
| `cpu` | CPU only |

---

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

### Reason Field

The `reason` field provides a human-readable explanation of the classification, including:
- Language framing analysis
- Partisan terminology detected
- Balance of perspectives
- Factual vs opinion content assessment

---

## Examples

### Example 1: Neutral News Article

**Input:**
```
The Federal Reserve announced today that it will maintain current 
interest rates, citing stable inflation and continued economic 
growth. Markets responded positively, with the S&P 500 gaining 
0.5% on the day.
```

**Output:**
```json
{
  "dir": {"L": 0.20, "C": 0.60, "R": 0.20},
  "deg": {"L": 0.70, "M": 0.20, "H": 0.10},
  "reason": "Factual reporting of economic news without partisan framing or loaded language."
}
```

### Example 2: Left-Leaning Opinion

**Input:**
```
The administration's cruel family separation policy represents 
a moral failure of epic proportions, demonstrating complete 
disregard for human dignity and basic human rights.
```

**Output:**
```json
{
  "dir": {"L": 0.70, "C": 0.15, "R": 0.15},
  "deg": {"L": 0.20, "M": 0.30, "H": 0.50},
  "reason": "Strong emotional language and clear progressive stance on immigration policy."
}
```

### Example 3: Right-Leaning Opinion

**Input:**
```
The radical left's relentless attack on traditional American 
values continues as they push their extreme agenda on 
hardworking taxpayers.
```

**Output:**
```json
{
  "dir": {"L": 0.10, "C": 0.15, "R": 0.75},
  "deg": {"L": 0.15, "M": 0.25, "H": 0.60},
  "reason": "Uses conservative framing with terms like 'radical left' and 'traditional values'."
}
```

---

## Troubleshooting

### Model Not Found

**Error:** `Model path does not exist`

**Solution:**
```bash
# Check the path
ls -la $BIAS_MODEL_PATH

# Set the correct path
export BIAS_MODEL_PATH=/correct/path/to/bias-detector-output
```

### Ollama Connection Failed

**Error:** `Error connecting to Ollama`

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Pull the model if not available
ollama pull llama3.2
```

### MPS Not Available

**Error:** `MPS device not available`

**Solution:**
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# Fall back to CPU
export BIAS_DEVICE=cpu
```

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Use CPU
export BIAS_DEVICE=cpu

# Or reduce batch size in your code
```

### Import Errors

**Error:** `Import "mcp" could not be resolved`

**Solution:**
```bash
pip install mcp
```

**Error:** `Import "ollama" could not be resolved`

**Solution:**
```bash
pip install ollama
```

---

## File Structure

```
t5/mcp_bias_server/
    __init__.py              # Package initialization
    server.py                # MCP server implementation
    bias_engine.py           # T5 model wrapper
    pyproject.toml           # Package configuration
    README.md                # Quick start guide
    DOCUMENTATION.md         # This file
    
    # Scripts
    build_package.sh         # Build wheel distribution
    publish_testpypi.sh      # Publish to TestPyPI
    publish_pypi.sh          # Publish to PyPI
    install_mcp_vscode.sh    # VSCode MCP setup
    
    # Examples
    examples/
        example_python_usage.py  # Python API example
    demo_ollama_bias.py      # Ollama integration demo
    demo_mcp_client.py       # MCP client demo
    
    # Tests
    tests/
        test_bias_engine.py  # Unit tests
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Support

For issues and feature requests, please open an issue on the GitHub repository.