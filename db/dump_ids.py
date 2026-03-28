#!/usr/bin/env python3
"""
dump_ids.py
Output MongoDB IDs of records with empty/missing fields.
"""

import os
import argparse
from typing import List, Optional
from urllib.parse import quote_plus

import pymongo
from bson import ObjectId
from datetime import datetime, timedelta

# Config — same as mongo2chroma.py
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
# URL-encode credentials for pymongo
def _build_mongo_uri(uri):
    if not uri.startswith("mongodb://"):
        # Add prefix if missing
        uri = "mongodb://" + uri
    # Only process if has credentials
    if "@" in uri:
        prefix = "mongodb://"
        rest = uri[len(prefix):]
        if "@" in rest:
            creds, hostpart = rest.rsplit("@", 1)
            if ":" in creds:
                user, password = creds.split(":", 1)
                return prefix + quote_plus(user) + ":" + quote_plus(password) + "@" + hostpart
    return uri

MONGO_URI = _build_mongo_uri(_MONGO_URI)
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
mongo_coll = mongo_db[MONGO_COLL]


def parse_date_arg(date_str: str) -> datetime:
    """Parse date argument. Supports ISO-8601 or negative days."""
    if date_str.startswith("-") and date_str[1:].isdigit():
        days_ago = int(date_str)
        return datetime.now() + timedelta(days=days_ago)
    return datetime.fromisoformat(date_str)


def build_empty_query(fields: List[str], mode: str) -> dict:
    """
    Build MongoDB query for records with empty/missing fields.

    Args:
        fields: List of field names to check
        mode: 'and' for ALL fields empty, 'or' for ANY field empty

    Returns:
        MongoDB query dict
    """
    clauses = []
    for field in fields:
        # Field is empty if: doesn't exist, is None, or is empty string
        clauses.append({
            "$or": [
                {field: {"$exists": False}},
                {field: None},
                {field: ""},
            ]
        })

    if mode == "and":
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]
    else:  # or
        return {"$or": clauses} if len(clauses) > 1 else clauses[0]


def dump_ids(
    and_fields: Optional[List[str]] = None,
    or_fields: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """Find and output MongoDB IDs with empty fields."""
    and_fields = and_fields or []
    or_fields = or_fields or []

    if not and_fields and not or_fields:
        print("Error: Must specify --and-fields and/or --or-fields", file=__import__('sys').stderr)
        return

    # Build base query
    q = {}

    # Add date filter if specified
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Add field emptiness conditions
    field_clauses = []

    for field in and_fields:
        field_clauses.append({
            "$or": [
                {field: {"$exists": False}},
                {field: None},
                {field: ""},
            ]
        })

    for field in or_fields:
        field_clauses.append({
            "$or": [
                {field: {"$exists": False}},
                {field: None},
                {field: ""},
            ]
        })

    if len(field_clauses) == 1:
        q.update(field_clauses[0])
    elif and_fields and or_fields:
        # Mix of and/or: need to group properly
        and_clause = {"$and": field_clauses[:len(and_fields)]} if len(and_fields) > 1 else field_clauses[0]
        or_clause = {"$or": field_clauses[len(and_fields):]} if len(or_fields) > 1 else field_clauses[len(and_fields)]
        q["$and"] = [and_clause, or_clause]
    elif and_fields:
        q["$and"] = field_clauses
    else:
        q["$or"] = field_clauses

    # Execute query
    cursor = mongo_coll.find(
        q,
        {"_id": 1},
        no_cursor_timeout=True
    ).sort("published", -1)

    if limit:
        cursor = cursor.limit(limit)

    count = 0
    for doc in cursor:
        print(str(doc["_id"]))
        count += 1

    print(f"# Total: {count}", file=__import__('sys').stderr)


def main():
    parser = argparse.ArgumentParser(description="Dump MongoDB IDs of records with empty fields")
    parser.add_argument(
        "--and-fields",
        help="Comma-separated field names. Output IDs of records missing ALL these fields.",
    )
    parser.add_argument(
        "--or-fields",
        help="Comma-separated field names. Output IDs of records missing ANY of these fields.",
    )
    parser.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)",
    )
    parser.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="Limit number of IDs output",
    )

    args = parser.parse_args()

    and_fields = None
    or_fields = None

    if args.and_fields:
        and_fields = [f.strip() for f in args.and_fields.split(",") if f.strip()]

    if args.or_fields:
        or_fields = [f.strip() for f in args.or_fields.split(",") if f.strip()]

    dump_ids(
        and_fields=and_fields,
        or_fields=or_fields,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
