#!/usr/bin/env python
import sys
import argparse
from mlx_lm import load, generate

# Create the parser
parser = argparse.ArgumentParser(
    description="Generate text from a prompt using an MLX language model."
)

# Add arguments
parser.add_argument(
    "prompt_source", help="File path for the prompt or '-' to read from stdin."
)
parser.add_argument(
    "--model",
    type=str,
    default="mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    help="The model name to use.",
)
parser.add_argument(
    "--tokens",
    type=int,
    default=38912,
    help="The maximum number of tokens to generate.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose output to show thinking process.",
)
parser.add_argument(
    "--think",
    action="store_true",
    help="Enable verbose output to show thinking process.",
)
parser.add_argument(
    "--temp",
    type=float,
    default=0.7,
    help="The temperature for text generation.",
)
parser.add_argument(
    "--TopP",
    type=float,
    default=0.8,
    help="The TopP value for text generation.",
)
parser.add_argument(
    "--TopK",
    type=int,
    default=20,
    help="The TopK value for text generation.",
)
parser.add_argument(
    "--MinP",
    type=float,
    default=0.0,
    help="The MinP value for text generation.",
)

# Parse the arguments
args = parser.parse_args()

# Read prompt
if args.prompt_source == "-":
    # Read from stdin
    prompt = sys.stdin.read().strip()
else:
    # Read from file
    try:
        with open(args.prompt_source, "r") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{args.prompt_source}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

# Load model
print(f"Loading model: {args.model}...", file=sys.stderr)
model, tokenizer = load(args.model)

# Prepare messages
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=args.think,
    Temperature=args.temp,
    TopP=args.TopP,
    TopK=args.TopK,
    MinP=args.MinP,
)

# Generate response
text = generate(
    model, tokenizer, prompt=prompt, verbose=args.verbose, max_tokens=args.tokens
)
print(text)
