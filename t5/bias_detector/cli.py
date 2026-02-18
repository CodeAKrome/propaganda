#!/usr/bin/env python3
"""
Simple bias detection CLI - analyze text directly.

Usage:
    python -m bias_detector.cli "Your text here"
    echo "Article text" | python -m bias_detector.cli
    python -m bias_detector.cli --file article.txt
"""

import sys
import json
import argparse
import requests
from typing import Optional


def detect_bias(text: str, api_url: str = "http://localhost:8000") -> Optional[dict]:
    """
    Detect political bias in text.
    
    Args:
        text: Text to analyze
        api_url: URL of T5 bias detection server
        
    Returns:
        Bias classification dict or None on error
    """
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"text": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("result", {})
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Detect political bias in text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "The article text to analyze..."
  echo "Article text" | %(prog)s
  %(prog)s --file article.txt
  %(prog)s "text" --json
        """
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or read from stdin)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Read text from file"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="T5 server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    args = parser.parse_args()
    
    # Get text
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)
    
    if not text.strip():
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)
    
    # Detect bias
    result = detect_bias(text, args.api_url)
    
    if result is None:
        sys.exit(1)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        if "dir" in result and "deg" in result:
            dir_labels = {"L": "Left", "C": "Center", "R": "Right"}
            deg_labels = {"L": "Low", "M": "Medium", "H": "High"}
            
            dir_max = max(result["dir"].items(), key=lambda x: x[1])
            deg_max = max(result["deg"].items(), key=lambda x: x[1])
            
            print(f"\nPolitical Bias Analysis")
            print("=" * 40)
            print(f"Direction: {dir_labels.get(dir_max[0], dir_max[0])} ({dir_max[1]:.0%})")
            print(f"Degree:    {deg_labels.get(deg_max[0], deg_max[0])} ({deg_max[1]:.0%})")
            print(f"\nDirection scores:")
            for k in ["L", "C", "R"]:
                if k in result["dir"]:
                    bar = "█" * int(result["dir"][k] * 20)
                    print(f"  {dir_labels.get(k, k):6} {result['dir'][k]:.0%} {bar}")
            print(f"\nDegree scores:")
            for k in ["L", "M", "H"]:
                if k in result["deg"]:
                    bar = "█" * int(result["deg"][k] * 20)
                    print(f"  {deg_labels.get(k, k):6} {result['deg'][k]:.0%} {bar}")
            if "reason" in result:
                print(f"\nReason: {result['reason']}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
