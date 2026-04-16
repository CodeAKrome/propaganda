#!/usr/bin/env python3
"""
mongo_rw - MongoDB field reader/writer tool.

Reads and writes individual fields in MongoDB documents.
Target collection: rssnews.articles

Commands:
  read  --field <field> --id <id> [--idfile <file>] [--data <file>]
  write --field <field> --id <id> [--idfile <file>] [--data <file>] [--force]

Examples:
  # READ (default: output to stdout)
  ./mongo_rw.py read --field title --id 696282dd5f8dd0157bb3d388
  ./mongo_rw.py read --field bias --id "id1,id2,id3"
  ./mongo_rw.py read --field title --idfile ids.txt
  ./mongo_rw.py read --field title --idfile -          # from stdin
  ./mongo_rw.py read --field title --id ID --data -      # to stdout
  ./mongo_rw.py read --field title --id ID --data file.txt

  # WRITE (default: read from stdin)
  echo "value" | ./mongo_rw.py write --field myfield --id ID
  ./mongo_rw.py write --field bias --id ID --data -      # stdin explicit
  ./mongo_rw.py write --field bias --id ID --data file.json
  ./mongo_rw.py write --field status --id ID --force    # overwrite

Environment:
  MONGO_URI=mongodb://root:example@localhost:27017
"""

import os
import sys
import json
import fire
from pymongo import MongoClient
from bson.objectid import ObjectId


def parse_ids(id_str=None, idfile=None):
    """Parse IDs from --id and --idfile arguments."""
    ids = []

    if id_str:
        for part in id_str.split(","):
            part = part.strip()
            if part:
                ids.append(part)

    if idfile:
        if idfile == "-":
            content = sys.stdin.read()
        else:
            with open(idfile, "r") as f:
                content = f.read()

        for line in content.split("\n"):
            line = line.strip()
            if line:
                ids.append(line)

    return ids


def read(id=None, idfile=None, field=None, data=None):
    """Read a specific field from document(s).

    Arguments:
      --id      MongoDB ID(s) - single or comma-separated
      --idfile  File with IDs (one per line) or - for stdin
      --field   Field name to read (required)
      --data    Output file path, - for stdout

    Examples:
      ./mongo_rw.py read --field title --id ID
      ./mongo_rw.py read --field bias --id "id1,id2"
      ./mongo_rw.py read --field title --idfile ids.txt
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
    client = MongoClient(mongo_uri)
    collection = client["rssnews"]["articles"]

    if not field:
        print("Error: --field is required")
        return

    ids = parse_ids(id, idfile)
    if not ids:
        print("Error: No IDs provided (use --id or --idfile)")
        return

    results = []
    errors = []

    for id_str in ids:
        try:
            doc = collection.find_one({"_id": ObjectId(id_str)})

            if doc is None:
                errors.append(f"Document '{id_str}' not found")
                continue

            if field not in doc:
                errors.append(f"Field '{field}' not found in '{id_str}'")
                continue

            results.append({"id": id_str, "field": field, "value": doc[field]})

        except Exception as e:
            errors.append(f"Error reading '{id_str}': {e}")

    # Output results
    if data is not None:
        if data == "-":
            for r in results:
                val = r["value"]
                print(json.dumps(val) if isinstance(val, (dict, list)) else val)
        else:
            with open(data, "w") as f:
                for r in results:
                    val = r["value"]
                    f.write(
                        (json.dumps(val) if isinstance(val, (dict, list)) else str(val))
                        + "\n"
                    )
    else:
        for r in results:
            val = r["value"]
            print(json.dumps(val) if isinstance(val, (dict, list)) else val)

    for err in errors:
        print(f"Warning: {err}", file=sys.stderr)


def write(id=None, idfile=None, field=None, data=None, force=False):
    """Write data to a specific field in document(s).

    Arguments:
      --id      MongoDB ID(s) - single or comma-separated
      --idfile  File with IDs (one per line) or - for stdin
      --field   Field name to write (required)
      --data    Data source: file path, - for stdin
      --force   Overwrite existing field data

    Examples:
      echo "value" | ./mongo_rw.py write --field myfield --id ID
      ./mongo_rw.py write --field bias --id ID --data file.json
      ./mongo_rw.py write --field status --id ID --force
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
    client = MongoClient(mongo_uri)
    collection = client["rssnews"]["articles"]

    if not field:
        print("Error: --field is required")
        return

    ids = parse_ids(id, idfile)
    if not ids:
        print("Error: No IDs provided (use --id or --idfile)")
        return

    # Load data
    if data is None:
        data_content = sys.stdin.read().strip()
        if not data_content:
            print("Error: No data provided (use --data or stdin)")
            return
    elif data == "-":
        data_content = sys.stdin.read().strip()
        if not data_content:
            print("Error: No data provided via stdin")
            return
    else:
        with open(data, "r") as f:
            data_content = f.read().strip()

    # Parse JSON for bias field
    final_data = data_content
    if field == "bias":
        try:
            final_data = json.loads(data_content)
        except json.JSONDecodeError:
            pass

    success_count = 0
    errors = []

    for id_str in ids:
        try:
            if not force:
                doc = collection.find_one({"_id": ObjectId(id_str)}, {field: 1})

                if doc is None:
                    errors.append(f"Document '{id_str}' not found")
                    continue

                if field in doc and doc[field] not in (None, ""):
                    errors.append(
                        f"Skipped '{id_str}': field '{field}' already has data (use --force)"
                    )
                    continue

            result = collection.update_one(
                {"_id": ObjectId(id_str)}, {"$set": {field: final_data}}
            )

            if result.matched_count == 0:
                errors.append(f"Document '{id_str}' not found")
            else:
                success_count += 1

        except Exception as e:
            errors.append(f"Error writing to '{id_str}': {e}")

    if success_count > 0:
        print(f"Successfully updated {success_count} document(s)")

    for err in errors:
        print(f"Error: {err}", file=sys.stderr)


if __name__ == "__main__":
    commands = {"read": read, "write": write}
    fire.Fire(commands, name="mongo_rw")
