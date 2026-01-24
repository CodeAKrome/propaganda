#!/usr/bin/env python3
"""
Read file paths from stdin and print *all* MongoDB IDs
found in each file (every line that starts with "ID: ").
"""

import sys
import re
from pathlib import Path

ID_PATTERN = re.compile(r"^ID:\s*([0-9a-f]{24})\s*$", re.I)


def extract_ids(file_path: str):
    """Yield every MongoDB ObjectId string found in the file."""
    try:
        txt = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"extract_ids.py: {exc}  ->  {file_path}", file=sys.stderr)
        return

    for line in txt.splitlines():
        if m := ID_PATTERN.match(line):
            yield m.group(1)


def main() -> None:
    for raw in sys.stdin:
        path = raw.rstrip("\n")
        for _id in extract_ids(path):
            print(_id)


if __name__ == "__main__":
    main()
