#!/usr/bin/env python3
"""
CLI tool for bias detection - instant out-of-the-box experience.

Usage:
    bias "Your text here"
    bias --file article.txt
    echo "text" | bias
    bias --json "text"
"""

import sys
import os
import json
import argparse
from typing import Optional

# Suppress noise before imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_progress(msg: str, end: str = "\n", flush: bool = True):
    """Print progress message to stderr so it doesn't pollute stdout."""
    print(msg, end=end, file=sys.stderr, flush=flush)

def main():
    parser = argparse.ArgumentParser(
        prog="bias",
        description="Analyze political bias in text - instant out-of-the-box experience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bias "The president announced new policies today."
  bias --file article.txt
  echo "Your text" | bias
  bias --json "text"                    # JSON output
  bias --quiet "text"                   # Just output L, C, or R
  bias --info                           # Show model info
"""
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or use --file, or pipe via stdin)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Read text from file"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (just direction: L, C, or R)"
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show model information"
    )
    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Disable HuggingFace auto-download (use local model only)"
    )
    
    args = parser.parse_args()
    
    # Get text from various sources
    text = None
    
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # Read from pipe
        text = sys.stdin.read().strip()
    
    if not text and not args.info:
        parser.print_help()
        sys.exit(1)
    
    # Import and initialize engine with progress
    print_progress("Loading bias detector...", end="")
    
    try:
        from mcp_bias_server.bias_engine import BiasEngine
        
        engine = BiasEngine(
            use_huggingface=not args.no_huggingface
        )
        
        if args.info:
            print_progress(" done!")
            info = engine.get_model_info()
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print(f"Base Model: {info['base_model']}")
                print(f"Adapter: {info['adapter_path']}")
                print(f"Device: {info['device']}")
                print(f"Source: {info.get('source', 'Local')}")
            sys.exit(0)
        
        # Perform analysis
        print_progress(" analyzing...", end="")
        result = engine.analyze(text)
        print_progress(" done!")
        
        # Output results
        if args.quiet:
            # Just output the dominant direction
            direction = result.direction
            dominant = max(direction, key=direction.get)
            print(dominant)
        elif args.json:
            output = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "direction": result.direction,
                "degree": result.degree,
                "reason": result.reason
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            direction = result.direction
            degree = result.degree
            
            # Find dominant values
            dom_dir = max(direction, key=direction.get)
            dom_deg = max(degree, key=degree.get)
            
            dir_labels = {"L": "Left", "C": "Center", "R": "Right"}
            deg_labels = {"L": "Low", "M": "Medium", "H": "High"}
            
            print(f"\n{'='*50}")
            print(f"BIAS ANALYSIS")
            print(f"{'='*50}")
            print(f"\nDirection: {dir_labels[dom_dir]}-leaning ({direction[dom_dir]*100:.0f}%)")
            print(f"  Left: {direction['L']*100:.0f}%  Center: {direction['C']*100:.0f}%  Right: {direction['R']*100:.0f}%")
            print(f"\nIntensity: {deg_labels[dom_deg]} ({degree[dom_deg]*100:.0f}%)")
            print(f"  Low: {degree['L']*100:.0f}%  Medium: {degree['M']*100:.0f}%  High: {degree['H']*100:.0f}%")
            print(f"\nReasoning: {result.reason}")
            print(f"{'='*50}\n")
    
    except Exception as e:
        print_progress(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()