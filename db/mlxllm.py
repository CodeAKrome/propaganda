#!/usr/bin/env python
import sys
import argparse
import signal
import time
from functools import wraps
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
parser.add_argument(
    "--time_limit",
    type=int,
    default=0,
    help="Time limit in seconds for LLM generation (0 = no limit).",
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
)

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("LLM generation timed out")

def generate_with_timeout(model, tokenizer, prompt, verbose, max_tokens, time_limit):
    # Consolidate generation arguments
    gen_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "verbose": verbose,
        "max_tokens": max_tokens,
    }

    if time_limit > 0:
        # Set the signal handler and alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)

        try:
            text = generate(
                **gen_kwargs
            )
            # Disable the alarm if generation completes successfully
            signal.alarm(0)
            return text
        except TimeoutException:
            # Graceful timeout handling with detailed reporting
            error_msg = f"LLM generation timed out after {time_limit} seconds"
            print(f"TIMEOUT: {error_msg}", file=sys.stderr)
            print(f"REPORT: Generation was terminated due to exceeding time limit of {time_limit} seconds", file=sys.stderr)
            print(f"STATUS: FAILED", file=sys.stderr)
            print(f"REASON: Timeout", file=sys.stderr)
            print(f"TIME_LIMIT: {time_limit}", file=sys.stderr)
            print(f"MODEL: {args.model}", file=sys.stderr)
            print(f"PROMPT_LENGTH: {len(prompt)}", file=sys.stderr)
            print(f"MAX_TOKENS: {max_tokens}", file=sys.stderr)
            return f"[TIMEOUT_ERROR] {error_msg}"
        except Exception as e:
            signal.alarm(0)  # Disable alarm before re-raising
            raise e
    else:
        # No time limit
        return generate(**gen_kwargs)

# Generate response
text = generate_with_timeout(
    model, tokenizer, prompt=prompt, verbose=args.verbose, max_tokens=args.tokens,
    time_limit=args.time_limit
)
print(text)
