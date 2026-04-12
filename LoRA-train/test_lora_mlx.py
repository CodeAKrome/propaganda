#!/usr/bin/env python3

"""
Test LoRA trained model against bias labels.
==============================================
Loads trained adapters and evaluates on test data.
"""

import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_bias_label(text: str) -> str:
    """Extract bias label from model output."""
    text = text.strip()

    # Try to find pattern like "L-M", "C-H", "R-L"
    import re

    match = re.search(r"([LRC])[-\s]([LMH])", text, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}-{match.group(2).upper()}"

    # Also try just L/M/H at start
    match = re.match(r"^([LRC])-\s*", text)
    if match:
        direction = match.group(1).upper()
        # Try to get degree too
        match2 = re.search(r"-\s*([LMH])", text)
        if match2:
            return f"{direction}-{match2.group(1).upper()}"
        return direction

    # Check for partial like just "L-" or "C-"
    if text.startswith("L-"):
        return "L-M"  # Default
    if text.startswith("C-"):
        return "C-M"
    if text.startswith("R-"):
        return "R-M"

    return "UNKNOWN"


def load_test_data(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test samples")
    return data


def get_ground_truth(data: Dict) -> str:
    """Extract ground truth bias label."""
    bias = data.get("bias", {})

    if not bias and "output" in data:
        try:
            output = json.loads(data["output"])
            dir_vals = output.get("direction", {})
            deg_vals = output.get("degree", {})
        except:
            dir_vals, deg_vals = {}, {}
    else:
        dir_vals = bias.get("dir", bias.get("direction", {}))
        deg_vals = bias.get("deg", bias.get("degree", {}))

    # Also check for bias_dir/bias_deg format
    if not dir_vals and "bias_dir" in data:
        dir_vals = data.get("bias_dir", {})
        deg_vals = data.get("bias_deg", {})

    if dir_vals:
        direction = max(dir_vals, key=dir_vals.get)
        degree = max(deg_vals, key=deg_vals.get)
    else:
        direction = "C"
        degree = "M"

    return f"{direction}-{degree}"


def run_inference(
    model: str,
    adapter_path: str,
    text: str,
    max_tokens: int = 10,
) -> str:
    """Run inference using mlx_lm Python API."""
    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    model_obj, tokenizer = load(model, adapter_path=adapter_path)

    prompt = f"""<|im_start|>user
Analyze the political bias of this text:
{text[:500]}<|im_end|>
<|im_start|>assistant
The political bias label is:"""

    result = []
    for response in stream_generate(
        model_obj, tokenizer, prompt, max_tokens=max_tokens
    ):
        result.append(response.text)

    return "".join(result)


def evaluate(
    test_data_path: str,
    model: str,
    adapter_path: str,
    output_path: str,
    max_samples: int = 50,
):
    """Evaluate model on test data."""
    test_data = load_test_data(test_data_path)[:max_samples]

    results = []
    correct = 0
    total = 0

    logger.info(f"Evaluating {len(test_data)} samples with {model}")

    for i, item in enumerate(test_data):
        text = item.get("input", item.get("text", ""))[:1000]
        ground_truth = get_ground_truth(item)

        prediction = run_inference(model, adapter_path, text)
        predicted = extract_bias_label(prediction)

        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "index": i,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "prediction_text": prediction[:100],
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(
                f"Processed {i + 1}/{len(test_data)}, Accuracy: {correct / total:.2%}"
            )

    accuracy = correct / total if total > 0 else 0

    by_category = {}
    summary = {
        "model": model,
        "adapter_path": adapter_path,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    for r in results:
        gt = r["ground_truth"]
        if gt not in by_category:
            by_category[gt] = {"correct": 0, "total": 0}
        by_category[gt]["total"] += 1
        if r["correct"]:
            by_category[gt]["correct"] += 1

    for cat in by_category:
        by_category[cat]["accuracy"] = (
            by_category[cat]["correct"] / by_category[cat]["total"]
        )

    with open(output_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "by_category": by_category,
                "results": results,
            },
            f,
            indent=2,
        )

    print_results(summary, by_category, results)

    return summary, by_category, results


def print_results(summary: Dict, by_category: Dict, results: List[Dict]):
    """Print results table."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {summary['model']}")
    print("=" * 70)

    print(
        f"\nOverall Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total_samples']})"
    )

    print("\n--- By Bias Category ---")
    print(f"{'Category':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 45)

    for cat in sorted(by_category.keys()):
        stats = by_category[cat]
        print(
            f"{cat:<12} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:.2%}"
        )

    print("\n--- Sample Predictions ---")
    print(f"{'#':<4} {'Ground Truth':<12} {'Predicted':<12} {'Status':<8}")
    print("-" * 40)

    for r in results[:20]:
        status = "✓" if r["correct"] else "✗"
        print(
            f"{r['index']:<4} {r['ground_truth']:<12} {r['predicted']:<12} {status:<8}"
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test LoRA model")
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="mlx-community/Qwen3.5-35B-A3B-4bit"
    )
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="lora_eval_results.json")
    parser.add_argument("--max-samples", type=int, default=50)

    args = parser.parse_args()

    evaluate(
        test_data_path=args.test_data,
        model=args.model,
        adapter_path=args.adapter_path,
        output_path=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
