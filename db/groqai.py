#!/usr/bin/env python3
import sys
import argparse

try:
    from groq import Groq
except ImportError:
    print("Error: The 'groq' library is not installed.", file=sys.stderr)
    print("Please install it using: pip install groq", file=sys.stderr)
    sys.exit(1)

# Create the parser
parser = argparse.ArgumentParser(
    description="Get a completion from the Groq API."
)

# Add arguments
parser.add_argument(
    "--model",
    type=str,
    default="llama-3.1-8b-instant",
    help="The model name to use.",
)

# Parse the arguments
args = parser.parse_args()

# Read prompt from stdin
prompt = sys.stdin.read().strip()

if not prompt:
    print("Error: No prompt provided via stdin.", file=sys.stderr)
    sys.exit(1)

client = Groq()
completion = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": prompt}],
)
print(completion.choices[0].message.content)
