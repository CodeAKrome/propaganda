# Political Bias Detector

[![PyPI version](https://badge.fury.io/py/bias-detector.svg)](https://badge.fury.io/py/bias-detector)
[![Python](https://img.shields.io/pypi/pyversions/bias-detector.svg)](https://pypi.org/project/bias-detector/)
[![Docker](https://img.shields.io/docker/v/yourname/bias-detector?label=docker)](https://hub.docker.com/r/yourname/bias-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green.svg)](https://modelcontextprotocol.io/)

**Assess political bias in news articles using AI.**

A T5-based machine learning model that analyzes news articles and returns structured political bias classifications with explanations.

## Components

| Component | Description |
|-----------|-------------|
| [`mcp_bias_server/`](mcp_bias_server/) | **MCP Server** - Model Context Protocol server for AI assistant integration |
| [`bias_detector/`](bias_detector/) | **Python Package** - CLI and MongoDB processing tools |
| [`docker-inference/`](docker-inference/) | **Docker** - CPU and MPS inference containers |

## Quick Start

### Option 0: MCP Server (For AI Assistants)

```bash
pip install bias-mcp-server

# Run the MCP server
bias-mcp-server
```

Add to your MCP configuration:
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

See [`mcp_bias_server/README.md`](mcp_bias_server/README.md) for full documentation.

### Option 1: Simple CLI (Easiest)

```bash
pip install bias-detector

# Analyze text directly
bias-detect "The news article text to analyze..."

# Or from a file
bias-detect --file article.txt

# Or pipe text
echo "Article text here" | bias-detect

# JSON output
bias-detect "text" --json
```

**Output:**
```
Political Bias Analysis
========================================
Direction: Center (60%)
Degree:    Medium (80%)

Direction scores:
  Left   20% ████
  Center 60% ████████████
  Right  20% ████

Degree scores:
  Low    10% ██
  Medium 80% ████████████████
  High   10% ██

Reason: The article maintains a neutral tone...
```

### Option 2: Python API

```bash
pip install bias-detector
```

```python
from bias_detector import detect_bias

# Simple detection
result = detect_bias("Your article text here")
print(result)
# {'dir': {'L': 0.2, 'C': 0.6, 'R': 0.2}, 'deg': {'L': 0.1, 'M': 0.8, 'H': 0.1}, 'reason': '...'}
```

### Option 3: MongoDB Batch Processing

```python
from bias_detector import BiasProcessor

# Process articles from MongoDB
processor = BiasProcessor(api_url="http://localhost:8000")
processor.process_articles(batch_size=100)
processor.close()
```

### Option 4: Docker

```bash
# Pull and run
docker pull yourname/bias-detector:latest
docker run --rm -e MONGO_USER=$MONGO_USER -e MONGO_PASS=$MONGO_PASS yourname/bias-detector --batch-size 10
```

## Output Format

```json
{
  "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
  "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
  "reason": "The article maintains a neutral tone..."
}
```

| Field | Description |
|-------|-------------|
| **dir** | Political direction: L (Left), C (Center), R (Right) |
| **deg** | Bias degree: L (Low), M (Medium), H (High) |
| **reason** | Explanation of the classification |

## Features

- **Accurate Classification**: Fine-tuned T5-large model
- **JSON Repair**: Automatically fixes malformed LLM output
- **Key Normalization**: Handles variations (left→L, center→C, etc.)
- **MongoDB Integration**: Process articles directly from database
- **Validation Tool**: Compare stored results with fresh predictions
- **Graceful Shutdown**: Saves progress on interruption
- **Progress Tracking**: Visual progress bar with detailed output

## Requirements

- Python 3.10+
- MongoDB (for batch processing)
- T5 bias detection server (included in `server_mps.py`)

## Starting the T5 Server

```bash
cd t5/
pip install -r requirements.txt
python server_mps.py
# Server runs at http://localhost:8000
```

## CLI Usage

```bash
# Process all articles without bias
bias-processor

# Process with limits
bias-processor --batch-size 100 --max-failures 5

# Dry run (don't write to database)
bias-processor --dry-run --batch-size 10

# Validate existing results
bias-validator --sample 50 --output results.json
```

## Python API

```python
from bias_detector import BiasProcessor, BiasValidator

# Process articles
processor = BiasProcessor(
    api_url="http://localhost:8000",
    output_file="processed.json"
)
processor.process_articles(
    batch_size=100,
    max_failures=3,
    dry_run=False
)
processor.close()

# Validate results
validator = BiasValidator(api_url="http://localhost:8000")
validator.validate(sample_size=10, output_file="validation.json")
validator.close()
```

## Installation from Source

```bash
git clone https://github.com/yourusername/bias-detector.git
cd bias-detector
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check bias_detector
```

## Model Training

The model was trained on manually labeled news articles. See `TRAINING_DATA_GUIDE.md` for details on training data format.

To train your own model:

```bash
python mongo2training.py  # Export from MongoDB
# Use train.json with your training pipeline
```

## API Reference

### POST /predict

Request:
```json
{
  "text": "Article text to analyze..."
}
```

Response:
```json
{
  "result": {
    "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
    "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
    "reason": "..."
  },
  "device": "mps"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this in your research, please cite:

```bibtex
@software{bias_detector,
  title = {Political Bias Detector},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bias-detector}
}
```
