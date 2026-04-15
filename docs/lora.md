# LoRA Training Pipeline

Fine-tune local LLM models with LoRA adapters for political bias detection.

## Components

| File | Description |
|------|-------------|
| `LoRA-train/mongo2lora.py` | Extract balanced training data |
| `LoRA-train/train_lora.py` | Fine-tune with LoRA |
| `LoRA-train/test_lora.py` | Evaluate models |
| `LoRA-train/train_lora_mlx.py` | MLX training (Apple Silicon) |
| `LoRA-train/test_lora_mlx.py` | MLX testing |
| `LoRA-server/server.py` | FastAPI server for serving |

---

## mongo2lora.py

Extract balanced training data from MongoDB.

### Usage

```bash
python LoRA-train/mongo2lora.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output FILE` | Output JSON file | Required |
| `--target-samples N` | Samples per bucket | 1000 |
| `--start-date DATE` | Start date (-N or ISO) | None |
| `--end-date DATE` | End date (-N or ISO) | None |
| `--model-type TYPE` | Model format (t5/llama/qwen) | `t5` |
| `--task TASK` | Task type | `bias_detection` |
| `--bias-field FIELD` | Bias field name | `bias` |

### Examples

```bash
# Extract 1000 samples per bias category
python LoRA-train/mongo2lora.py -o train.json --target-samples 1000

# Last 30 days
python LoRA-train/mongo2lora.py -o train.json --start-date -30

# Llama format
python LoRA-train/mongo2lora.py -o train.json --model-type llama
```

### Output Formats

#### T5 Format

```json
[
  {
    "input_text": "classify political bias: Article text...",
    "target_text": "{\"dir\": {\"L\": 0.2, \"C\": 0.6, \"R\": 0.2}, \"deg\": {\"L\": 0.1, \"M\": 0.8, \"H\": 0.1}}"
  }
]
```

#### Llama Format

```json
[
  {
    "instruction": "Analyze the political bias in this news article.",
    "input": "Article text...",
    "output": "The article shows..."
  }
]
```

---

## train_lora.py

Fine-tune with LoRA adapters.

### Usage

```bash
python LoRA-train/train_lora.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--data FILE` | Training data | Required |
| `--output-dir DIR` | Output directory | `lora-output` |
| `--base-model MODEL` | Base model | `google/t5-large-lm-adapt` |
| `--epochs N` | Epochs | 3 |
| `--batch-size N` | Batch size | 4 |
| `--learning-rate RATE` | Learning rate | `3e-4` |
| `--rank N` | LoRA rank | 16 |
| `--alpha N` | LoRA alpha | 32 |

### Examples

```bash
python LoRA-train/train_lora.py --data train.json --epochs 3
python LoRA-train/train_lora.py --data train.json --base-model t5-base --batch-size 8
```

---

## train_lora_mlx.py

MLX-optimized training for Apple Silicon.

### Usage

```bash
python LoRA-train/train_lora_mlx.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--data FILE` | Training data | Required |
| `--output-dir DIR` | Output directory | `lora_mlx_output` |
| `--base-model MODEL` | Base model | `llama-3.2-1b` |
| `--batch-size N` | Batch size | 1 |
| `--epochs N` | Epochs | 3 |

### Examples

```bash
python LoRA-train/train_lora_mlx.py --data train.json --epochs 5
```

---

## test_lora.py

Test trained models.

### Usage

```bash
python LoRA-train/test_lora.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path DIR` | LoRA weights path | Required |
| `--test-data FILE` | Test data JSON | Required |
| `--base-model MODEL` | Base model | `t5-large-lm-adapt` |

### Examples

```bash
python LoRA-train/test_lora.py --model-path lora-output --test-data test.json
```

---

## LoRA-server/server.py

FastAPI server for serving LoRA models.

### Usage

```bash
export MODEL_PATH=/path/to/lora-weights
python LoRA-server/server.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Analyze bias |
| `/health` | GET | Health check |

### Request

```json
{
  "text": "Article text to analyze..."
}
```

### Response

```json
{
  "dir": {"L": 0.15, "C": 0.60, "R": 0.25},
  "deg": {"L": 0.10, "M": 0.80, "H": 0.10},
  "reason": "Balanced coverage..."
}
```

---

## Environment Variables

```bash
MONGO_URI=mongodb://root:pass@localhost:27017
MONGO_DB=rssnews
MONGO_COLL=articles
MODEL_PATH=./lora-output
```

## Makefile Targets

```makefile
# Full pipeline
lora-full: lora-extract lora-train lora-test

# Extract training data
lora-extract:
	python LoRA-train/mongo2lora.py -o train.json

# Train model
lora-train:
	python LoRA-train/train_lora.py --data train.json --epochs 3

# Test model
lora-test:
	python LoRA-train/test_lora.py --model-path lora-output --test-data test.json

# Serve model
lora-serve:
	python LoRA-server/server.py
```

## Bias Categories (Buckets)

Training data is balanced across:

| Direction | Degree |
|-----------|--------|
| Left (L) | Low (L) |
| Center (C) | Medium (M) |
| Right (R) | High (H) |

Each bucket gets `--target-samples` articles for balanced training.

## Dependencies

```bash
pip install transformers peft torch sentence-transformers
pip install mlx  # For MLX training
pip install fastapi uvicorn  # For server
```