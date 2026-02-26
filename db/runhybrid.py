#!/usr/bin/env python3
"""
runhybrid.py
Batch runner for hybrid.py using a tab-delimited configuration file.

Each row in the TSV maps to one hybrid.py invocation.
Calls hybrid.main(argv) in-process so the embedding model is loaded only once.

Usage:
    ./runhybrid.py batch.tsv [--dry-run] [--row N] [--output DIR]
"""

import os
import sys
import csv
import argparse
from contextlib import redirect_stdout

# Column name -> CLI flag mapping
# Positional arg "text" is handled specially (uses "vector" column)
COLUMN_TO_FLAG = {
    "n":            "-n",
    "andentity":    "--andentity",
    "orentity":     "--orentity",
    "start_date":   "--start-date",
    "end_date":     "--end-date",
    "fulltext":     "--fulltext",
    "showentity":   "--showentity",
    "search":       "--search",
    "orsearch":     "--orsearch",
    "ids":          "--ids",
    "embedding":    "--embedding",
    "query":        "--bm25-query",
}

# Boolean flags (no value, just present/absent)
BOOL_FLAGS = {
    "substr":       "--substr",
    "bm25":         "--bm25",
    "flair_pooled": "--flair-pooled",
    "bge_large":    "--bge-large",
}

TRUE_VALUES = {"1", "true", "yes"}


def is_truthy(val: str) -> bool:
    return val.strip().lower() in TRUE_VALUES


def build_argv(row: dict) -> list[str]:
    """Build an argv list for hybrid.main() from one TSV row."""
    argv = []

    # Positional text argument — only use "vector" column
    text = (row.get("vector") or "").strip()

    # Value-bearing flags
    for col, flag in COLUMN_TO_FLAG.items():
        val = (row.get(col) or "").strip()
        if not val:
            continue
        if col == "showentity":
            if val == "*":
                argv.append(flag)        # flag with no value
            else:
                argv.extend([flag, val])  # flag with value
        else:
            argv.extend([flag, val])

    # Boolean flags
    for col, flag in BOOL_FLAGS.items():
        val = (row.get(col) or "").strip()
        if val and is_truthy(val):
            argv.append(flag)

    # Positional text goes at the end (after all flags)
    if text:
        argv.append(text)

    return argv


def format_command(argv: list[str]) -> str:
    """Format argv as a shell command for display."""
    parts = []
    for a in argv:
        if " " in a or '"' in a or "'" in a:
            parts.append(f'"{a}"')
        else:
            parts.append(a)
    return "./hybrid.py " + " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for hybrid.py using a TSV config file"
    )
    parser.add_argument("tsv", help="Path to the tab-delimited batch file")
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Run only this row number (1-indexed, excluding header)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Read TSV
    rows = []
    with open(args.tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for lineno, row in enumerate(reader, start=1):
            # Skip comment rows (first non-empty value starts with #)
            first_val = next((v for v in row.values() if v and v.strip()), "")
            if first_val.startswith("#"):
                continue
            rows.append((lineno, row))

    if not rows:
        print("No runnable rows found in TSV.", file=sys.stderr)
        sys.exit(1)

    # Filter to a specific row if requested
    if args.row is not None:
        rows = [(n, r) for n, r in rows if n == args.row]
        if not rows:
            print(f"Row {args.row} not found.", file=sys.stderr)
            sys.exit(1)

    total = len(rows)
    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {total} runs from {args.tsv}", file=sys.stderr)

    if not args.dry_run:
        # Import hybrid only when actually running
        import hybrid

    for i, (row_num, row) in enumerate(rows, start=1):
        argv = build_argv(row)
        label = (row.get("label") or "").strip() or f"row-{row_num}"
        cmd_str = format_command(argv)
        out_path = os.path.join(args.output, f"{label}.vec")

        print(f"\n{'='*72}", file=sys.stderr)
        print(f"[{i}/{total}] {label}: {cmd_str}", file=sys.stderr)
        print(f"  -> {out_path}", file=sys.stderr)
        print(f"{'='*72}", file=sys.stderr)

        if args.dry_run:
            print(f"{cmd_str}  > {out_path}")
        else:
            try:
                with open(out_path, "w") as out_f:
                    with redirect_stdout(out_f):
                        hybrid.main(argv)
                print(f"  wrote {out_path}", file=sys.stderr)
            except SystemExit:
                pass  # argparse may call sys.exit on --help etc.
            except Exception as e:
                print(f"ERROR on row {row_num} ({label}): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
