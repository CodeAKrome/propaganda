#!/usr/bin/env python3

"""
MLX LoRA Fine-tuning for Apple Silicon
=========================================
Fine-tunes Qwen2.5-Coder, Llama, Mistral on Mac Silicon with MLX.

Uses Apple's MLX library for GPU-accelerated training on M-series chips.

Usage:
    python train_lora_mlx.py --data train.json --model Qwen/Qwen2.5-Coder-32B-Instruct --output lora-mlx
    python train_lora_mlx.py --data train.json --model meta-llama/Llama-3.2-1B --epochs 3
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LORA_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


def load_train_data(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} training samples")
    return data


def format_sample(data: Dict) -> str:
    """Format training sample for model."""
    text = data.get("text", "")[:2000]
    bias = data.get("bias", {})

    dir_vals = bias.get("dir", bias.get("direction", {}))
    deg_vals = bias.get("deg", bias.get("degree", {}))

    if dir_vals:
        direction = max(dir_vals, key=dir_vals.get)
        degree = max(deg_vals, key=deg_vals.get)
    else:
        direction = "C"
        degree = "M"

    label = f"{direction}-{degree}"

    return f"""<|im_start|>user
Analyze the political bias of this news text:
{text}<|im_end|>
<|im_start|>assistant
The political bias is: {label}<|im_end|>"""


def prepare_dataset(data_path: str, output_dir: str):
    """Prepare training data in MLX format."""
    data = load_train_data(data_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.tx"
    with open(train_file, "w") as f:
        for item in data:
            formatted = format_sample(item)
            f.write(formatted + "\n")

    logger.info(f"Wrote {len(data)} samples to {train_file}")
    return str(train_file)


def train_lora(
    data_path: str,
    model: str = LORA_MODEL,
    output: str = "lora_mlx",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 8,
    max_seq_length: int = 2048,
):
    """Train LoRA with MLX."""
    from mlx_lm.tuner import train
    from mlx_lm.tuner.trainer import TrainingArgs

    data_file = prepare_dataset(data_path, output)

    args = TrainingArgs(
        batch_size=batch_size,
        iters=epochs * 100,
        val_batches=10,
        steps_per_report=10,
        steps_per_save=50,
        max_seq_length=max_seq_length,
        adapter_file="adapters.safetensors",
        grad_checkpoint=False,
    )

    logger.info(f"Starting LoRA training with {model}")
    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logger.info(f"LoRA rank: {lora_r}")

    try:
        train(
            model=model,
            train_utils=None,
            args=args,
        )
        logger.info("Training completed!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-tuning")
    parser.add_argument("--data", type=str, default="lora_train.json")
    parser.add_argument("--model", type=str, default=LORA_MODEL)
    parser.add_argument("--output", type=str, default="lora_mlx")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)

    args = parser.parse_args()

    try:
        train_lora(
            data_path=args.data,
            model=args.model,
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            max_seq_length=args.max_length,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
