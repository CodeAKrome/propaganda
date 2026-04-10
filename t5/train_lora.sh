#!/bin/bash
#
# T5 Bias Detector Training - Production Script
# ============================================
# Fine-tunes T5-large with LoRA for political bias detection
# Outputs structured JSON with direction (L/C/R) and degree (L/M/H) scores
#
# Usage:
#   ./train_lora.sh                    # Train with defaults
#   ./train_lora.sh --epochs 5         # Custom epochs
#   EPOCHS=5 BATCH_SIZE=4 ./train_lora.sh  # Environment variables
#

set -e

# Configuration - can override with env vars
DATA_FILE="${DATA_FILE:-train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-bias-detector-output}"
MODEL_NAME="${MODEL_NAME:-t5-large}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

# MongoDB config for telemetry
export MONGO_URI="${MONGO_URI:-mongodb://root:example@localhost:27017}"

# Detect device
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "========================================"
echo "T5 Bias Detector Training (LoRA)"
echo "========================================"
echo "Data file:     $DATA_FILE"
echo "Output dir:    $OUTPUT_DIR"
echo "Model:         $MODEL_NAME"
echo "Device:        $DEVICE"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA R:        $LORA_R"
echo "LoRA Alpha:    $LORA_ALPHA"
echo "LoRA Dropout:  $LORA_DROPOUT"
echo "MongoDB:       $MONGO_URI"
echo "========================================"
echo ""

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

# Run training with telemetry
echo "Starting training with telemetry..."
python3 train_bias_detector.py \
    --data "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "View training telemetry in dashboard:"
echo "  streamlit run ../dashboard/app.py"
echo ""
echo "To test the model:"
echo "  python -c \"from mcp_bias_server import BiasEngine; e = BiasEngine('$OUTPUT_DIR'); print(e.analyze('Your text'))\""
