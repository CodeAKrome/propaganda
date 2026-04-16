# Redist - ReDistributed Training

T5-based bias detection model training pipeline with LoRA fine-tuning.

## Quick Reference

| Task | Command |
|------|---------|
| Run training | `./run_training.sh` |
| Custom epochs | `./run_training.sh --epochs 5` |
| Train and test | `./run_training.sh --test` |

## Shell Scripts

### run_training.sh
Trains T5-large with LoRA for bias detection.

```bash
./run_training.sh [options]

Options:
  --epochs <n>      Number of training epochs (default: 3)
  --batch-size <n> Batch size (default: 8)
  --learning-rate  Learning rate (default: 5e-4)
  --test            Run evaluation after training
  --help            Show all options

Environment Variables:
  DATA_FILE         Training data JSON (default: train.json)
  OUTPUT_DIR        Output directory (default: bias-detector-output)
  MODEL_NAME        Base model (default: t5-large)
  DEVICE            Training device (auto-detect: cuda/mps/cpu)

Example:
# Train with defaults
./run_training.sh

# Train for 5 epochs
EPOCHS=5 ./run_training.sh

# Custom model and output
MODEL_NAME=t5-base OUTPUT_DIR=my-model ./run_training.sh --test
```

**What it does:**
1. Detects available device (CUDA/MPS/CPU)
2. Validates training data exists
3. Runs train_bias_detector.py with configured parameters
4. Saves model checkpoints to OUTPUT_DIR

## Prerequisites

### Training Data
Generate training data from MongoDB:

```bash
python mongo2training.py -o train.json
```

Training data format:
```json
[
  {
    "text": "Article text content...",
    "bias": {
      "dir": {"L": 0.1, "C": 0.4, "R": 0.5},
      "deg": {"L": 0.1, "M": 0.2, "H": 0.7},
      "reason": "Bias explanation..."
    }
  }
]
```

### Python Dependencies
```bash
pip install torch
pip install transformers
pip install peft        # LoRA
pip install accelerate
pip install jsonlines
```

## Configuration

### Default Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL_NAME | t5-large | Base T5 model |
| EPOCHS | 3 | Training epochs |
| BATCH_SIZE | 8 | Batch size |
| LEARNING_RATE | 5e-4 | Optimizer learning rate |
| OUTPUT_DIR | bias-detector-output | Model save path |

### Device Selection
Automatically selects best available device:
1. NVIDIA GPU (cuda)
2. Apple Silicon (mps)
3. CPU (fallback)

## Output

### Checkpoint Structure
```
bias-detector-output/
├── checkpoint-1000/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-2000/
└── ...
```

### Using Trained Model
```python
from mcp_bias_server.bias_engine import BiasEngine

engine = BiasEngine(model_path='bias-detector-output')
result = engine.analyze('Your text here')

# Result format:
# {
#   "dir": {"L": 0.1, "C": 0.3, "R": 0.6},
#   "deg": {"L": 0.1, "M": 0.2, "H": 0.7},
#   "reason": "..."
# }
```

## Training Workflow

```bash
# 1. Generate training data from MongoDB
cd ..
python t5/mongo2training.py -o train.json --limit 1000

# 2. Run training
cd redist
./run_training.sh --epochs 3

# 3. Test model (optional)
./run_training.sh --test

# 4. Deploy (see MCP server docs)
```

## Troubleshooting

### Out of memory
```bash
# Reduce batch size
BATCH_SIZE=4 ./run_training.sh

# Or use gradient accumulation
# (edit train_bias_detector.py)
```

### No CUDA available
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Install correct PyTorch version: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Training too slow
- Use smaller model: `MODEL_NAME=t5-base ./run_training.sh`
- Increase batch size if memory allows
- Use MPS on Apple Silicon: `DEVICE=mps ./run_training.sh`

## See Also

- Main README: [../README.md](../README.md)
- T5 Documentation: [../t5/README.md](../t5/README.md)
- LoRA Training: [../LoRA-train/README.md](../LoRA-train/README.md)
- MCP Server: [../t5/mcp_bias_server/README.md](../t5/mcp_bias_server/README.md)