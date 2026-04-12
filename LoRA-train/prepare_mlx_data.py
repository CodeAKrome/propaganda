#!/usr/bin/env python3

"""
Prepare training data for MLX LoRA fine-tuning.
===============================================
Converts bias detection data to JSONL format for mlx-lm.

Usage:
    python prepare_mlx_data.py --input lora_train.json --output data/mlx_train
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_sample(data: Dict) -> dict:
    """Format a single training sample."""
    text = data.get("input", data.get("text", ""))[:2500]

    # Parse bias from output JSON field
    try:
        output = json.loads(data.get("output", "{}"))
        dir_vals = output.get("direction", {})
        deg_vals = output.get("degree", {})
    except:
        dir_vals, deg_vals = {}, {}

    # Also check for bias field directly
    if not dir_vals and "bias" in data:
        bias = data.get("bias", {})
        dir_vals = bias.get("dir", bias.get("direction", {}))
        deg_vals = bias.get("deg", bias.get("degree", {}))

    # Also check for extracted bias labels in MongoDB export format
    if not dir_vals and "bias_dir" in data:
        dir_vals = data.get("bias_dir", {})
        deg_vals = data.get("bias_deg", {})

    if dir_vals:
        direction = max(dir_vals, key=dir_vals.get)
        degree = max(deg_vals, key=deg_vals.get)
    else:
        direction = "C"
        degree = "M"

    label = f"{direction}-{degree}"

    return {
        "text": f"""<|im_start|>user
Analyze the political bias of this text:
{text}<|im_end|>
<|im_start|>assistant
The political bias label is: {label}<|im_end|>"""
    }


def prepare_jsonl(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.1,
):
    """Prepare train/valid JSONL files."""
    with open(input_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples from {input_path}")

    n_train = int(len(data) * train_ratio)
    n_valid = len(data) - n_train

    train_data = data[:n_train]
    valid_data = data[n_train:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_data), ("valid", valid_data)]:
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                sample = format_sample(item)
                f.write(json.dumps(sample) + "\n")

        logger.info(f"Wrote {len(split_data)} samples to {output_file}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.1)

    args = parser.parse_args()

    prepare_jsonl(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )


if __name__ == "__main__":
    main()
