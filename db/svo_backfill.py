#!/usr/bin/env python3

"""
svo_backfill.py
Backfill SVO relationships for existing articles in MongoDB.

Usage:
    python db/svo_backfill.py                          # all articles without svo
    python db/svo_backfill.py --start-date -30       # last 30 days
    python db/svo_backfill.py --end-date -7           # last 7 days
    python db/svo_backfill.py -n 10                   # limit 10
    python db/svo_backfill.py --start-date -30 --end-date -7  # specific range
"""

import os
import sys
import argparse
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone, timedelta

import pymongo
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

SVO_PROMPT = os.path.join(os.path.dirname(__file__), "prompt", "kgsvo.txt")
FILTER_ANSI = os.path.join(os.path.dirname(__file__), "filter_ansi.py")
ERROR_LOG = os.path.join(os.path.dirname(__file__), "output", "svo_errors.txt")
TIMEOUT_SEC = 600


def parse_date(date_str: str) -> datetime:
    if not date_str:
        raise ValueError("Date string cannot be empty")
    if date_str.startswith("-"):
        remainder = date_str[1:]
        if remainder.isdigit():
            return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                days=int(remainder)
            )
        raise ValueError(f"Invalid relative date format: {date_str}")
    return datetime.fromisoformat(date_str)


def run_ollama(article_text: str) -> str:
    tmp_id = uuid.uuid4().hex[:8]
    tmp_out = f"/tmp/svo_bf_{tmp_id}.out"
    tmp_filt = f"/tmp/svo_bf_{tmp_id}.filtered"
    try:
        # Read prompt file
        with open(SVO_PROMPT, "r") as f:
            prompt_content = f.read()

        # Combine prompt + article text
        full_input = prompt_content + "\n" + article_text

        # Call ollama with input via stdin (no shell injection)
        result = subprocess.run(
            ["ollama", "run", "--hidethinking", "--nowordwrap", "gpt-oss:120b"],
            input=full_input,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )

        if result.returncode != 0:
            return f"[ERROR: exit {result.returncode}]"

        # Filter ANSI codes
        result2 = subprocess.run(
            ["python3", FILTER_ANSI],
            input=result.stdout,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result2.stdout if result2.returncode == 0 else result.stdout
    except subprocess.TimeoutExpired:
        return "[ERROR: timeout]"
    except Exception as e:
        return f"[ERROR: {e}]"
    finally:
        for f in (tmp_out, tmp_filt):
            try:
                os.unlink(f)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Backfill SVO for existing MongoDB articles"
    )
    parser.add_argument("--start-date", help="Start date: ISO or -N")
    parser.add_argument("--end-date", help="End date: ISO or -N")
    parser.add_argument(
        "-n", "--limit", type=int, default=0, help="Max articles to process (0=all)"
    )
    args = parser.parse_args()

    client = pymongo.MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_COLL]

    mongo_filter = {
        "article": {"$exists": True, "$ne": None},
        "svo": {"$exists": False},
    }
    if args.start_date:
        mongo_filter.setdefault("published", {})["$gte"] = parse_date(args.start_date)
    if args.end_date:
        mongo_filter.setdefault("published", {})["$lte"] = parse_date(args.end_date)

    total = coll.count_documents(mongo_filter)
    print(f"Found {total} articles without svo", file=sys.stderr)

    if total == 0:
        return

    limit = args.limit or total
    cursor = (
        coll.find(mongo_filter, {"_id": 1, "article": 1})
        .sort("published", -1)
        .limit(limit)
    )

    os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
    processed = 0
    errors = 0

    for doc in cursor:
        oid = doc["_id"]
        text = doc.get("article", "") or ""
        if not text:
            processed += 1
            continue

        if len(text) > 80000:
            text = text[:80000]

        output = run_ollama(text)

        if output.startswith("[ERROR:"):
            errors += 1
            with open(ERROR_LOG, "a") as elog:
                elog.write(f"{str(oid)}: {output}\n")
            coll.update_one({"_id": oid}, {"$set": {"svo": f"[ERROR: {output[8:-1]}]"}})
        else:
            triplet_lines = [
                l.strip() for l in output.split("\n") if l.strip().startswith('("')
            ]
            coll.update_one(
                {"_id": oid},
                {"$set": {"svo": "\n".join(triplet_lines), "svo_llm": output}},
            )

        processed += 1
        if processed % 10 == 0:
            print(f"Progress: {processed}/{min(limit, total)}", file=sys.stderr)

        if processed >= limit:
            break

    print(
        f"Done: {processed} processed, {errors} errors. Log: {ERROR_LOG}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
