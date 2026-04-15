# T5 Bias Detector System

Complete documentation for the T5-based political bias detection pipeline.

## Components

| File | Description |
|------|-------------|
| `t5/bias_detector.py` | Core T5 training and inference |
| `t5/train_bias_detector.py` | Training script with telemetry |
| `t5/mongo2training.py` | MongoDB to training data exporter |
| `t5/bias_detector_mps_optimized.py` | MPS-optimized inference |
| `t5/BiasDetectorInference.py` | Inference class |
| `t5/server_mps.py` | MPS server |
| `t5/BiasDetectorInference_MPS.py` | MPS inference class |

---

## bias_detector.py

T5 model for political bias detection with CLI support and non-fatal error handling.

### Usage

```bash
python t5/bias_detector.py [arguments]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data FILE` | Training data JSON file | Built-in samples |
| `--model-name` | Base T5 model | `t5-large` |
| `--epochs N` | Training epochs | 10 |
| `--batch-size N` | Batch size | 2 |
| `--output-dir DIR` | Output directory | `./bias-detector-output` |
| `--predict-only` | Skip training, just predict | False |
| `--model-path DIR` | Path to trained model | Required with `--predict-only` |
| `--export-sample FILE` | Export sample data file | None |

### Examples

#### Train with Custom Data

```bash
python t5/bias_detector.py --data training_data.json --epochs 5 --batch-size 4
```

#### Predict with Existing Model

```bash
python t5/bias_detector.py --predict-only --model-path ./bias-detector-output
```

#### Export Sample Data

```bash
python t5/bias_detector.py --export-sample sample_data.json
```

### Training Data Format

```json
[
  {
    "article": "Article text to analyze...",
    "label": {
      "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
      "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
      "reason": "Explanation of bias assessment"
    }
  }
]
```

---

## train_bias_detector.py

Training script with MongoDB telemetry integration.

### Usage

```bash
python t5/train_bias_detector.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--data FILE` | Training data JSON | Required |
| `--output-dir DIR` | Model output directory | `bias-detector-output` |
| `--epochs N` | Number of epochs | 3 |
| `--batch-size N` | Batch size | 8 |
| `--base-model MODEL` | Base T5 model | `t5-large` |
| `--learning-rate RATE` | Learning rate | `5e-4` |
| `--run-id ID` | Telemetry run ID | Auto-generated |

### Examples

```bash
python t5/train_bias_detector.py --data train.json --epochs 3 --batch-size 8
python t5/train_bias_detector.py --data train.json --output-dir bias-v2 --epochs 5
```

### Telemetry

Training metrics are logged to MongoDB `training_telemetry` collection:

```json
{
  "run_id": "bias-v2-20250415",
  "timestamp": "2025-04-15T14:30:00Z",
  "step": 500,
  "epoch": 1.5,
  "metrics": {"loss": 0.42, "eval_loss": 0.38}
}
```

---

## mongo2training.py

Export MongoDB articles with bias labels to training data format.

### Usage

```bash
python t5/mongo2training.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--start-date DATE` | Start date (-N or ISO) | Required |
| `--end-date DATE` | End date (-N or ISO) | None |
| `--output FILE` | Output JSON file | `training_data.json` |
| `--min-length N` | Min article length | 100 |
| `--bias-field FIELD` | Bias field name | `bias` |
| `--limit N` | Max articles | None |

### Examples

```bash
# Export last 7 days
python t5/mongo2training.py --start-date -7 --output train.json

# Export specific date range
python t5/mongo2training.py --start-date 2025-01-01 --end-date 2025-03-31

# Limit results
python t5/mongo2training.py --start-date -30 --limit 1000 --output train.json
```

### Output Format

```json
[
  {
    "article": "Article text content...",
    "label": {
      "dir": {"L": 0.15, "C": 0.60, "R": 0.25},
      "deg": {"L": 0.10, "M": 0.80, "H": 0.10},
      "reason": "Balanced coverage with neutral language..."
    }
  }
]
```

---

## bias_detector_mps_optimized.py

MPS-optimized inference for Apple Silicon.

### Usage

```bash
python t5/bias_detector_mps_optimized.py [text]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `text` | Article text to analyze | Required (stdin if omitted) |
| `--model-path DIR` | Model path | `./bias-detector-output` |
| `--json` | Output JSON format | False |

### Examples

```bash
echo "Article text..." | python t5/bias_detector_mps_optimized.py
python t5/bias_detector_mps_optimized.py "Article text..." --json
```

---

## MongoDB Schema

### Articles Collection

```json
{
  "_id": ObjectId,
  "title": "Article Title",
  "article": "Full article text...",
  "source": "cnn",
  "published": ISODate,
  "ner": {
    "entities": [...],
    "entity_counts": {...}
  },
  "bias": {
    "dir": {"L": 0.15, "C": 0.60, "R": 0.25},
    "deg": {"L": 0.10, "M": 0.80, "H": 0.10},
    "reason": "Balanced coverage with neutral language..."
  }
}
```

### Training Telemetry Collection

```json
{
  "run_id": "bias-v2-20250415",
  "timestamp": ISODate,
  "step": 500,
  "epoch": 1.5,
  "metrics": {"loss": 0.42, "eval_loss": 0.38}
}
```

---

## Model Architecture

```
Input:  "classify political bias as json: {article text}"
Output: '{"dir": {"L": 0.2, "C": 0.6, "R": 0.2}, "deg": {"L": 0.1, "M": 0.8, "H": 0.1}, "reason": "..."}'
```

### LoRA Configuration

- Task Type: SEQ2SEQ
- Target modules: `q`, `v`
- Rank: 16
- Alpha: 32
- Dropout: 0.1

---

## Makefile Targets

```makefile
# Train bias detector
t5train:
	python t5/train_bias_detector.py --data $(TRAIN_DATA) --epochs 3

# Extract training data
t5extract:
	python t5/mongo2training.py --start-date -30 --output train.json

# Run inference server
t5serve:
	python t5/server_mps.py
```

---

## Dependencies

```bash
pip install torch transformers peft datasets
pip install sentence-transformers  # For embeddings
pip install rank-bm25  # For hybrid search
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid data, connection failure) |

---

## Troubleshooting

### "No valid entries found"

- Check JSON format matches expected schema
- Verify `article` and `label` fields exist
- Check `label` has `dir`, `deg`, `reason` fields

### "CUDA out of memory"

- Reduce batch size: `--batch-size 2`
- Use `t5-small` instead of `t5-large`
- Enable gradient checkpointing

### "MPS not available"

- Use CPU inference: modify script to use cpu device
- Or use docker-inference containers