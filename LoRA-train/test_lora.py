#!/usr/bin/env python3

"""
Test LoRA Model Quality
========================
Tests a trained LoRA model on held-out data.

Usage:
    python test_lora.py --model-path ./lora-output --test-data test.json
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_path: str):
    device = get_device()
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def analyze(text: str, model, tokenizer, device, max_tokens: int = 512) -> str:
    prompt = f"""Analyze the political bias of this article. 
Return JSON with direction (L/C/R with probabilities) and degree (L/M/H with probabilities).

Article: {text[:2000]}

Response:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in result:
        return result[len(prompt) :].strip()
    return result.strip()


def evaluate(test_data: List[Dict], model, tokenizer, device) -> Dict[str, Any]:
    correct_direction = 0
    correct_degree = 0
    total = 0
    errors = 0

    predictions = []

    for item in test_data:
        try:
            text = item.get("text", item.get("input", ""))
            expected = item.get("bias", item.get("output", {}))

            result = analyze(text, model, tokenizer, device)

            try:
                predicted = json.loads(result)
            except json.JSONDecodeError:
                predicted = {"raw": result}
                errors += 1

            pred_dir = (
                max(
                    predicted.get("direction", predicted.get("dir", {})).items(),
                    key=lambda x: x[1],
                )[0]
                if "direction" in predicted or "dir" in predicted
                else None
            )
            pred_deg = (
                max(
                    predicted.get("degree", predicted.get("deg", {})).items(),
                    key=lambda x: x[1],
                )[0]
                if "degree" in predicted or "deg" in predicted
                else None
            )

            exp_dir = (
                max(
                    expected.get("direction", expected.get("dir", {})).items(),
                    key=lambda x: x[1],
                )[0]
                if "direction" in expected or "dir" in expected
                else None
            )
            exp_deg = (
                max(
                    expected.get("degree", expected.get("deg", {})).items(),
                    key=lambda x: x[1],
                )[0]
                if "degree" in expected or "deg" in expected
                else None
            )

            if pred_dir == exp_dir:
                correct_direction += 1
            if pred_deg == exp_deg:
                correct_degree += 1

            total += 1

            predictions.append(
                {
                    "expected": {"direction": exp_dir, "degree": exp_deg},
                    "predicted": {"direction": pred_dir, "degree": pred_deg},
                    "raw": result[:200],
                }
            )

        except Exception as e:
            errors += 1
            predictions.append({"error": str(e)})

    return {
        "total": total,
        "direction_accuracy": correct_direction / total if total > 0 else 0,
        "degree_accuracy": correct_degree / total if total > 0 else 0,
        "errors": errors,
        "predictions": predictions[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Test LoRA model")
    parser.add_argument("--model-path", required=True, help="Path to LoRA model")
    parser.add_argument("--test-data", required=True, help="Test data JSON file")
    parser.add_argument("--output", help="Output file for results")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model, tokenizer, device = load_model(args.model_path)
    print(f"Model loaded on {device}")

    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    print(f"Evaluating {len(test_data)} test samples...")
    results = evaluate(test_data, model, tokenizer, device)

    print(f"\nResults:")
    print(f"  Total: {results['total']}")
    print(f"  Direction Accuracy: {results['direction_accuracy']:.2%}")
    print(f"  Degree Accuracy: {results['degree_accuracy']:.2%}")
    print(f"  Errors: {results['errors']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
