#!/usr/bin/env python3

"""
Database utility module for managing and accessing data.
"""

"""
runreport.py
Batch runner for report.py using a tab-delimited configuration file.

Each row in the TSV maps to one report.py invocation.
Reads the same hybrid_batch.tsv as runhybrid.py.

Usage:
    ./runreport.py batch.tsv [--dry-run] [--row N] [--output DIR] [--svoprompt FILE]
"""

import os
import sys
import csv
import argparse


def read_tsv(tsv_path):
    """Read TSV and return list of (lineno, row_dict) tuples, skipping comments."""
    rows = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for lineno, row in enumerate(reader, start=1):
            first_val = next((v for v in row.values() if v and v.strip()), "")
            if first_val.startswith("#"):
                continue
            rows.append((lineno, row))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for report.py using a TSV config file"
    )
    parser.add_argument("tsv", help="Path to the tab-delimited batch file")
    parser.add_argument(
        "--output",
        default="output",
        help="Working directory for input/output files (default: output)",
    )
    parser.add_argument(
        "--svoprompt",
        default="prompt/kgsvo.txt",
        help="Path to SVO prompt file (default: prompt/kgsvo.txt)",
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
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Number of articles to process concurrently (default: 1)",
    )
    args = parser.parse_args()

    # Read TSV
    rows = read_tsv(args.tsv)

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
    print(
        f"{'DRY RUN: ' if args.dry_run else ''}Processing {total} reports from {args.tsv}",
        file=sys.stderr,
    )
    if args.parallel > 1:
        print(f"  Running with --parallel {args.parallel}", file=sys.stderr)

    if not args.dry_run:
        import report
        import concurrent.futures

        def process_row(args_tuple):
            row_num, row, svoprompt, output = args_tuple
            label = (row.get("label") or "").strip() or f"row-{row_num}"
            query = (row.get("query") or "").strip()
            entity = (row.get("orentity") or row.get("andentity") or "").strip()
            start_date = (row.get("start_date") or "").strip()

            vec_path = os.path.join(output, f"{label}.vec")
            news_path = os.path.join(output, f"{label}.md")

            if not os.path.exists(vec_path):
                return (label, "SKIP", "vec not found")

            try:
                report.main(
                    startdate=start_date,
                    filename=label,
                    entity=entity,
                    query=query,
                    svoprompt=svoprompt,
                    workdir=output,
                )
                return (label, "OK", news_path)
            except Exception as e:
                return (label, "ERROR", str(e))

        # Prepare args for parallel processing
        process_args = [
            (row_num, row, args.svoprompt, args.output) for row_num, row in rows
        ]

        # Run in parallel
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel
        ) as executor:
            futures = {executor.submit(process_row, pa): pa for pa in process_args}
            for future in concurrent.futures.as_completed(futures):
                label, status, detail = future.result()
                completed += 1
                if status == "OK":
                    print(
                        f"[{completed}/{total}] {label}: wrote {detail}",
                        file=sys.stderr,
                    )
                elif status == "SKIP":
                    print(
                        f"[{completed}/{total}] {label}: SKIP - {detail}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[{completed}/{total}] {label}: ERROR - {detail}",
                        file=sys.stderr,
                    )
    else:
        # Sequential dry-run mode
        for i, (row_num, row) in enumerate(rows, start=1):
            label = (row.get("label") or "").strip() or f"row-{row_num}"
            query = (row.get("query") or "").strip()
            entity = (row.get("orentity") or row.get("andentity") or "").strip()
            start_date = (row.get("start_date") or "").strip()

            vec_path = os.path.join(args.output, f"{label}.vec")
            news_path = os.path.join(args.output, f"{label}.md")

            print(f"\n{'=' * 72}", file=sys.stderr)
            print(f"[{i}/{total}] {label}", file=sys.stderr)
            print(f"  vec:   {vec_path}", file=sys.stderr)
            print(f"  out:   {news_path}", file=sys.stderr)
            print(
                f"  query: {query[:80]}{'...' if len(query) > 80 else ''}",
                file=sys.stderr,
            )
            print(f"{'=' * 72}", file=sys.stderr)

            exists = os.path.exists(vec_path)
            print(
                f"report.main('{start_date}', '{label}', '{entity}', '{query}', svoprompt='{args.svoprompt}', workdir='{args.output}')  [{vec_path} {'exists' if exists else 'MISSING'}]"
            )


if __name__ == "__main__":
    main()
