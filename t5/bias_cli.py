#!/usr/bin/env python3
"""
bias_cli.py - Command-line interface for T5 bias detection

Loads t5-large model with LoRA adapters and runs bias classification.

Usage:
  python bias_cli.py --text "Your article text here"
  python bias_cli.py --input-file article.txt
  python bias_cli.py --lora ./my-adapter --model t5-large --text "text"

Environment:
  BIAS_LORA_PATH   LoRA adapter path (default: ./bias-detector-output)
  BIAS_MODEL       Base model name (default: t5-large)
  T5_DEVICE        Force device: mps, cuda, cpu (default: auto-detect)
"""

import os
import sys
import json
import re
import argparse
import torch
import warnings

# Suppress library noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel


# Global model cache
_model_cache = {}


def get_device():
    """Get the best available device: MPS > CUDA > CPU"""
    # Check for forced device
    forced = os.getenv("T5_DEVICE", "").lower()
    if forced == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif forced == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif forced == "cpu":
        return "cpu"

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(lora_path: str, base_model: str):
    """Load and cache the model.

    Args:
        lora_path: Path to LoRA adapter
        base_model: Base model name (e.g., t5-large)

    Returns:
        Dict with model, tokenizer, device
    """
    cache_key = f"{lora_path}:{base_model}"

    if cache_key not in _model_cache:
        device = get_device()
        print(f"Loading model on device: {device}", file=sys.stderr)

        tokenizer = T5Tokenizer.from_pretrained(base_model, verbose=False)
        base_model_obj = T5ForConditionalGeneration.from_pretrained(
            base_model, low_cpu_mem_usage=True
        )

        model = PeftModel.from_pretrained(base_model_obj, lora_path)
        model.to(device)
        model.eval()

        _model_cache[cache_key] = {"model": model, "tokenizer": tokenizer, "device": device}

    return _model_cache[cache_key]


def predict_bias(text: str, lora_path: str, base_model: str) -> dict:
    """Run bias prediction on text.

    Args:
        text: Text to classify
        lora_path: Path to LoRA adapter
        base_model: Base model name

    Returns:
        Dict with bias classification (dir, deg, reason)
    """
    instance = load_model(lora_path, base_model)

    formatted_input = f"classify political bias as json: {text}"

    inputs = instance["tokenizer"](
        formatted_input, return_tensors="pt", max_length=512, truncation=True
    ).to(instance["device"])

    with torch.no_grad():
        outputs = instance["model"].generate(
            **inputs, max_length=512, num_beams=4, early_stopping=True
        )

    raw_result = instance["tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()

    # Use the repair function to fix malformed JSON
    return repair_json(raw_result)


def main():
    """Repair malformed JSON from T5 model output.

    The model outputs: "dir":"L":0.2,"C":0.3,"R":0.5,"deg":"L":0.1,"M":0.4,"H":0.5,"reason":"..."
    Needs to become: {"dir":{"L":0.2,"C":0.3,"R":0.5},"deg":{"L":0.1,"M":0.4,"H":0.5},"reason":"..."}
    """
    s = output.strip()

    # Remove outer quotes if present
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # Unescape escaped quotes
    s = s.replace('\\"', '"')

    # Fix missing opening quote on first key
    if s.startswith("{") and not s.startswith('{"'):
        first_brace = s.find("{")
        rest = s[first_brace + 1 :]
        if rest and not rest.startswith('"'):
            s = s[: first_brace + 1] + '"' + rest

    # Fix missing quotes on known keys (dir, deg, reason)
    s = re.sub(r'(?<!")\b(dir|deg|reason)\b(?!"):', r'"\1":', s)

    # Add outer braces if missing
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"

    # Fix missing braces after "dir": and "deg":
    # Pattern: "dir":"L":0.2 → "dir":{"L":0.2}
    s = re.sub(r'"dir"\s*:\s*"([LRC])"\s*:', r'"dir":{"\1":', s)
    s = re.sub(r'"deg"\s*:\s*"([LMH])"\s*:', r'"deg":{"\1":', s)

    # Find positions and close braces properly
    dir_match = re.search(r'"dir"\s*:\s*\{', s)
    deg_match = re.search(r'"deg"\s*:\s*\{', s)
    reason_match = re.search(r'"reason"\s*:', s)

    if dir_match and deg_match:
        deg_start = deg_match.start()
        dir_section = s[:deg_start]
        open_braces = dir_section.count("{") - dir_section.count("}")
        if open_braces > 0:
            s = s[:deg_start] + "}" * (open_braces - 1) + "," + s[deg_start:]

    deg_match = re.search(r'"deg"\s*:\s*\{', s)
    reason_match = re.search(r'"reason"\s*:', s)

    if deg_match and reason_match:
        reason_start = reason_match.start()
        deg_section = s[:reason_start]
        open_braces = deg_section.count("{") - deg_section.count("}")
        if open_braces > 0:
            s = s[:reason_start] + "}" * (open_braces - 1) + "," + s[reason_start:]

    # Fix unterminated reason string
    if s.endswith("}"):
        reason_val_match = re.search(r'"reason"\s*:\s*"([^"]*)$', s)
        if reason_val_match:
            s = s[:-1] + '"}'

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"raw_output": output}


def main():
    parser = argparse.ArgumentParser(
        description="T5 Bias Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text from command line
  python bias_cli.py --text "Your article text here"

  # Analyze from file
  python bias_cli.py --input-file article.txt

  # Analyze from stdin
  echo "Article text" | python bias_cli.py

  # Custom LoRA adapter
  python bias_cli.py --lora ./my-adapter --text "text"

  # Custom base model
  python bias_cli.py --model t5-base --text "text"

Environment variables:
  BIAS_LORA_PATH   LoRA adapter path (default: ./bias-detector-output)
  BIAS_MODEL       Base model (default: t5-large)
  T5_DEVICE        Force device: mps, cuda, cpu
        """,
    )

    parser.add_argument("--text", "-t", type=str, default=None, help="Text to classify for bias")

    parser.add_argument(
        "--input-file", "-i", type=str, default=None, help="File containing text to classify"
    )

    parser.add_argument(
        "--lora",
        "-l",
        type=str,
        default=os.getenv("BIAS_LORA_PATH", "./bias-detector-output"),
        help="Path to LoRA adapter (default: ./bias-detector-output)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=os.getenv("BIAS_MODEL", "t5-large"),
        help="Base model name (default: t5-large)",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["mps", "cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto-detect)",
    )

    parser.add_argument(
        "--json", "-j", action="store_true", help="Output as compact JSON (no pretty print)"
    )

    args = parser.parse_args()

    # Handle device override
    if args.device != "auto":
        os.environ["T5_DEVICE"] = args.device

    # Determine input text
    text = args.text

    if text is None and args.input_file is not None:
        with open(args.input_file, "r") as f:
            text = f.read().strip()
    elif text is None and args.input_file is None:
        # Check stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()

    if not text:
        parser.error("No text provided. Use --text, --input-file, or stdin.")

    try:
        result = predict_bias(text, args.lora, args.model)

        if args.json:
            print(json.dumps(result), file=sys.stdout)
        else:
            print(json.dumps(result, indent=2), file=sys.stdout)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
