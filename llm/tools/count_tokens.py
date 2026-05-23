#!/usr/bin/env python3
"""
Token counter for estimating context window usage.

Estimates or calculates token counts for text files to help plan LLM context usage.
Supports both quick estimation and accurate tiktoken-based calculation.

Usage:
    python count_tokens.py [--estimate] [--accurate] [--encoder NAME] [--context-window SIZE] FILE...
    cat FILE | python count_tokens.py [--estimate] [--accurate]

Options:
    --estimate         Use character/4 estimation (fast, default)
    --accurate         Use tiktoken for accurate count
    --encoder NAME     Tokenizer encoding (cl100k_base, p50k_base, r50k_base)
                       Default: cl100k_base (GPT-4, Claude, etc.)
    --context-window SIZE  Show percentage of context window (e.g., 4k, 8k, 32k, 128k, 200k)
    --help             Show this help message

Examples:
    python count_tokens.py article.txt
    python count_tokens.py --accurate article.txt
    python count_tokens.py --context-window 128k article.txt
    python count_tokens.py --encoder p50k_base file.txt
    cat article.txt | python count_tokens.py --estimate
"""

import argparse
import sys
import os
from pathlib import Path

CONTEXT_WINDOWS = {
    "4k": 4000,
    "8k": 8000,
    "16k": 16000,
    "32k": 32000,
    "64k": 64000,
    "128k": 128000,
    "200k": 200000,
    "1m": 1000000,
}


def estimate_tokens(text: str) -> int:
    """Estimate token count using character/4 approximation."""
    return len(text) // 4


def accurate_tokens(text: str, encoder: str = "cl100k_base") -> int:
    """Calculate accurate token count using tiktoken."""
    try:
        import tiktoken
    except ImportError:
        print("Error: tiktoken not installed. Install with: pip install tiktoken", file=sys.stderr)
        sys.exit(1)

    try:
        enc = tiktoken.get_encoding(encoder)
    except Exception as e:
        print(f"Error: Failed to load encoding '{encoder}': {e}", file=sys.stderr)
        sys.exit(1)

    return len(enc.encode(text))


def parse_context_window(size_str: str) -> int:
    """Parse context window size string (e.g., '128k', '32k', '200k')."""
    size_str = size_str.lower().strip()
    if size_str in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[size_str]

    # Try to parse manually (e.g., "128000", "128k", "200k")
    try:
        if size_str.endswith("k"):
            return int(size_str[:-1]) * 1000
        elif size_str.endswith("m"):
            return int(size_str[:-1]) * 1000000
        elif size_str.endswith("K"):
            return int(size_str[:-1]) * 1000
        elif size_str.endswith("M"):
            return int(size_str[:-1]) * 1000000
        else:
            return int(size_str)
    except ValueError:
        print(f"Error: Invalid context window '{size_str}'. Valid: {list(CONTEXT_WINDOWS.keys())}", file=sys.stderr)
        sys.exit(1)


def format_tokens(n: int) -> str:
    """Format token count with commas."""
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in text files to estimate context window usage.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Use character/4 estimation (fast, default if --accurate not specified)",
    )
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Use tiktoken for accurate token count",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="cl100k_base",
        choices=["cl100k_base", "p50k_base", "r50k_base"],
        help="Tokenization encoding (default: cl100k_base)",
    )
    parser.add_argument(
        "--context-window",
        type=str,
        help="Show percentage of context window (e.g., 4k, 8k, 32k, 128k, 200k)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to process (use - for stdin)",
    )

    args = parser.parse_args()

    # Determine mode
    use_accurate = args.accurate
    use_estimate = args.estimate and not args.accurate

    # Parse context window if specified
    context_limit = None
    if args.context_window:
        context_limit = parse_context_window(args.context_window)

    # Read input
    text_parts = []

    # Check stdin first (only if explicitly piped)
    if not sys.stdin.isatty() and not args.files:
        # Stdin has data and no files specified
        text_parts.append(sys.stdin.read())

    # Read from files if specified
    if args.files:
        for filepath in args.files:
            if filepath == "-":
                # Explicit stdin indicator
                if not sys.stdin.isatty():
                    text_parts.append(sys.stdin.read())
            else:
                path = Path(filepath)
                if not path.exists():
                    print(f"Error: File not found: {filepath}", file=sys.stderr)
                    continue
                try:
                    text_parts.append(path.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}", file=sys.stderr)
    else:
        parser.print_help()
        sys.exit(1)

    # Combine all text
    full_text = "\n".join(text_parts)

    # Calculate tokens
    if use_accurate:
        token_count = accurate_tokens(full_text, args.encoder)
        mode = f"accurate ({args.encoder})"
    else:
        token_count = estimate_tokens(full_text)
        mode = "estimated (chars/4)"

    # Output results
    print(f"Tokens: {format_tokens(token_count)} ({mode})")
    print(f"Characters: {format_tokens(len(full_text))}")

    if context_limit:
        percentage = (token_count / context_limit) * 100
        print(f"Context usage: {percentage:.1f}% of {args.context_window.upper()} window")

    # Multiple files info
    if args.files and len(args.files) > 1:
        print(f"\nPer file:")
        for filepath in args.files:
            if filepath != "-":
                path = Path(filepath)
                if path.exists():
                    try:
                        text = path.read_text(encoding="utf-8")
                        if use_accurate:
                            cnt = accurate_tokens(text, args.encoder)
                        else:
                            cnt = estimate_tokens(text)
                        print(f"  {filepath}: {format_tokens(cnt)} tokens")
                    except:
                        pass


if __name__ == "__main__":
    main()