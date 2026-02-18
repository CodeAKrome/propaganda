#!/usr/bin/env python3
"""
Bias Analyzer - stdin/stdout version.

Reads text from stdin, analyzes for political bias, outputs JSON to stdout.

USAGE:
    echo "Your text here" | python bias_stdin.py
    cat article.txt | python bias_stdin.py
    python bias_stdin.py < article.txt

OUTPUT FORMAT (JSON):
    {
        "direction": {"L": 0.3, "C": 0.6, "R": 0.1},
        "degree": {"L": 0.2, "M": 0.3, "H": 0.5},
        "reason": "Explanation..."
    }

OPTIONS:
    --format=text    Output human-readable text instead of JSON
    --quiet          Only output the primary leaning (L/C/R)
"""

import os
import sys
import json
import argparse

# Set model path before importing
os.environ["BIAS_MODEL_PATH"] = "/Users/kyle/hub/propaganda/t5/bias-detector-output"

from mcp_bias_server.bias_engine import BiasEngine


def main():
    parser = argparse.ArgumentParser(description="Analyze text for political bias")
    parser.add_argument("--format", choices=["json", "text"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only output primary leaning (L/C/R)")
    args = parser.parse_args()

    # Read from stdin
    if sys.stdin.isatty():
        print("Error: No input provided. Pipe text to stdin.", file=sys.stderr)
        print("Example: echo 'text' | python bias_stdin.py", file=sys.stderr)
        sys.exit(1)

    text = sys.stdin.read().strip()
    
    if not text:
        print("Error: Empty input.", file=sys.stderr)
        sys.exit(1)

    # Analyze
    engine = BiasEngine()
    result = engine.analyze(text)

    # Output
    if args.quiet:
        dir_max = max(result.direction, key=result.direction.get)
        print(dir_max)
    elif args.format == "text":
        dir_max = max(result.direction, key=result.direction.get)
        deg_max = max(result.degree, key=result.degree.get)
        dir_label = {"L": "Left", "C": "Center", "R": "Right"}[dir_max]
        deg_label = {"L": "Low", "M": "Medium", "H": "High"}[deg_max]
        
        print(f"Direction: {dir_label} ({result.direction[dir_max]:.0%})")
        print(f"  L={result.direction['L']:.2f} C={result.direction['C']:.2f} R={result.direction['R']:.2f}")
        print(f"Intensity: {deg_label} ({result.degree[deg_max]:.0%})")
        print(f"  L={result.degree['L']:.2f} M={result.degree['M']:.2f} H={result.degree['H']:.2f}")
        print(f"Reason: {result.reason}")
    else:
        output = {
            "direction": result.direction,
            "degree": result.degree,
            "reason": result.reason
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()