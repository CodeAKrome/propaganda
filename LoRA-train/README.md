# LoRA Training & Serving

Fine-tune local LLM models with LoRA adapters for political bias detection.

## Quick Start

```bash
# Full pipeline: extract data → train → test
make lora-full

# Or step by step:
make lora-extract     # Extract training data from MongoDB
make lora-train       # Train LoRA model
make lora-test        # Test the model
make lora-serve       # Serve via API
```

## Directory Structure

```
LoRA-train/
├── mongo2lora.py     # Extract balanced training data from MongoDB
├── train_lora.py     # Fine-tune models with LoRA
└── test_lora.py      # Test trained models

LoRA-server/
└── server.py         # FastAPI server for serving LoRA models
```

## Requirements

- MongoDB with bias-labeled articles
- Python with transformers, peft, torch
- GPU (MPS/CUDA) recommended for training

## Environment Variables

```bash
MONGO_URI=mongodb://root:example@localhost:27017
MONGO_DB=rssnews
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `lora-extract` | Extract training data (balanced bias distribution) |
| `lora-extract-test` | Extract test data (different date range) |
| `lora-train` | Train LoRA model |
| `lora-test` | Evaluate model on test data |
| `lora-validate` | Validate training data quality |
| `lora-serve` | Start FastAPI server |
| `lora-stop` | Stop server |
| `lora-full` | Full pipeline: extract → train → test |
| `lora-quick` | Quick test with smaller data |

## Configuration

Edit Makefile variables or override on command line:

```bash
make lora-train LORA_MODEL=meta-llama/Llama-3.2-1B LORA_EPOCHS=3
make lora-serve LORA_PORT=8080
```

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LORA_MODEL` | meta-llama/Llama-3.2-1B | Model to fine-tune |
| `LORA_OUTPUT` | ./lora-output | Output directory |
| `LORA_TOTAL_SAMPLES` | 10000 | Target training samples |
| `LORA_MIN_SAMPLES` | 100 | Min samples per bias bin |
| `LORA_EPOCHS` | 3 | Training epochs |
| `LORA_BATCH_SIZE` | 4 | Batch size |
| `LORA_PORT` | 1337 | Server port |

## Supported Models

- **Llama 2/3** — `--model-type llama`
- **Qwen 2.5** — `--model-type qwen`
- **GLM-4** — `--model-type glm`
- **T5** — `--model-type t5`

## API Endpoints

When serving with `make lora-serve`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health |
| `/analyze` | POST | Analyze text for bias |
| `/analyze/batch` | POST | Batch analysis |
| `/generate` | POST | Generic text generation |
| `/models` | GET | Model info |

### Example Usage

```bash
# Analyze single text
curl -X POST http://localhost:1337/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text here..."}'

# Batch analysis
curl -X POST http://localhost:1337/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Article 1...", "Article 2..."]}'
```

## Data Format

Training data is extracted with balanced bias distribution:

- **Direction**: L (Left), C (Center), R (Right)
- **Degree**: L (Low), M (Medium), H (High)

Example balanced bin distribution:
- L-L, L-M, L-H
- C-L, C-M, C-H
- R-L, R-M, R-H

The script automatically allocates samples to create balanced training set.

## Output

Training produces:
- `lora-output/adapter_config.json` — LoRA config
- `lora-output/adapter_model.safetensors` — LoRA weights
- `lora-output/tokenizer.*` — Tokenizer files