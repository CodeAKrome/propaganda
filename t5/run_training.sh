#!/bin/bash
#
# T5 Bias Detector Training Script
# =================================
# Fine-tunes T5-large with LoRA for JSON output
#
# Usage:
#   ./run_training.sh                    # Train with defaults
#   ./run_training.sh --epochs 5         # Custom epochs
#   ./run_training.sh --test             # Train and test
#

set -e

# Configuration
DATA_FILE="${DATA_FILE:-train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-bias-detector-output}"
MODEL_NAME="${MODEL_NAME:-t5-large}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

# Detect device
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "========================================"
echo "T5 Bias Detector Training"
echo "========================================"
echo "Data file:    $DATA_FILE"
echo "Output dir:   $OUTPUT_DIR"
echo "Model:        $MODEL_NAME"
echo "Device:       $DEVICE"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "========================================"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Training data file not found: $DATA_FILE"
    echo ""
    echo "To generate training data from MongoDB, run:"
    echo "  python mongo2training.py -o train.json"
    exit 1
fi

# Count training samples
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_FILE'))))")
echo "Training samples: $SAMPLE_COUNT"
echo ""

# Run training
python3 train_bias_detector.py \
    --data "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    "$@"

echo ""
echo "========================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "To use the trained model:"
echo "  from mcp_bias_server.bias_engine import BiasEngine"
echo "  engine = BiasEngine(model_path='$OUTPUT_DIR')"
echo "  result = engine.analyze('Your text here')"
