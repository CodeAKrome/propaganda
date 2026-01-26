#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId


def get_mongo_connection():
    """Initialize and return MongoDB connection."""
    mongo_user = os.getenv("MONGO_USER")
    mongo_pass = os.getenv("MONGO_PASS")

    if not mongo_user or not mongo_pass:
        raise ValueError(
            "MONGO_USER and MONGO_PASS environment variables must be set"
        )

    uri = f"mongodb://{mongo_user}:{mongo_pass}@localhost:27017"
    client = MongoClient(uri)
    db = client["rssnews"]
    collection = db["articles"]
    return collection


def check_field_exists(collection, mongo_id, field):
    """
    Check if a field exists and has a value in the MongoDB document.
    
    Returns:
        True if field exists and has a non-empty value, False otherwise
    """
    try:
        doc = collection.find_one(
            {"_id": ObjectId(mongo_id)},
            {field: 1}
        )
        
        if doc is None:
            return False
        
        # Check if field exists and has data (not None and not empty string)
        return field in doc and doc[field] not in (None, "")
    
    except Exception as e:
        tqdm.write(f"Error checking field for {mongo_id}: {e}")
        return False


def fmt_command(model, mongo_id, cache, src, dest):
    """
    Format a shell command using the model name and MongoDB ID.
    
    Args:
        model: MLX model name
        mongo_id: MongoDB document ID
        cache: Cache file path
        src: Source field name in MongoDB
        dest: Destination field name in MongoDB
    """
    command = (
        f"(cat {cache}; ./mongo_rw.py read --id={mongo_id} --field={src}) | "
        f"mlx_lm.generate --model {model} --prompt - --max-tokens 100000 "
        f"--verbose FALSE | "
        f'grep "final<|message" | '
        f"perl -0777 -ne '@m = /\\{{(?:[^{{}}]|(?0))*\\}}/g; print $m[-1]' | "
        f"./mongo_rw.py write --id={mongo_id} --field={dest} --data=-"
    )
    return command


def main():
    parser = argparse.ArgumentParser(description="Process MongoDB IDs with MLX model")
    parser.add_argument("model", help="Name of MLX model to use")
    parser.add_argument("--cache", required=True, help="Path to prompt cache file")
    parser.add_argument("--src", required=True, help="Source field name in MongoDB")
    parser.add_argument("--dest", required=True, help="Destination field name in MongoDB")
    parser.add_argument("--force", action="store_true", 
                        help="Force update even if destination field has a value")
    args = parser.parse_args()

    # Connect to MongoDB
    try:
        collection = get_mongo_connection()
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}", file=sys.stderr)
        return

    # Read all MongoDB IDs from stdin
    mongo_ids = [line.strip() for line in sys.stdin if line.strip()]
    
    if not mongo_ids:
        print("No MongoDB IDs provided", file=sys.stderr)
        return

    # Deduplicate IDs while preserving order
    seen = set()
    unique_ids = []
    duplicates = 0
    for mongo_id in mongo_ids:
        if mongo_id not in seen:
            seen.add(mongo_id)
            unique_ids.append(mongo_id)
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"Found {duplicates} duplicate IDs (removed)", file=sys.stderr)
    
    mongo_ids = unique_ids

    # Filter out IDs that already have data (unless --force is used)
    ids_to_process = []
    skipped = 0
    
    if not args.force:
        for mongo_id in mongo_ids:
            if check_field_exists(collection, mongo_id, args.dest):
                skipped += 1
            else:
                ids_to_process.append(mongo_id)
    else:
        ids_to_process = mongo_ids
    
    # Exit if nothing to process
    if not ids_to_process:
        print(f"No IDs to process. All {skipped} IDs already have data in field '{args.dest}'", file=sys.stderr)
        print("Use --force to overwrite existing data", file=sys.stderr)
        return
    
    # Report pre-processing summary
    if skipped > 0:
        print(f"Pre-scan: {len(ids_to_process)} IDs to process, {skipped} already have data (skipped)", file=sys.stderr)
    
    # Process each ID with progress bar
    processed = 0
    
    for mongo_id in tqdm(ids_to_process, desc="Processing IDs", unit="id"):
        # Get the command for this ID
        command = fmt_command(args.model, mongo_id, args.cache, args.src, args.dest)

        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout, adjust as needed
            )

            # Print output
            if result.stdout:
                tqdm.write(f"[{mongo_id}] Output: {result.stdout.strip()}")

            # Print errors if any
            if result.returncode != 0:
                tqdm.write(
                    f"[{mongo_id}] Error (exit code {result.returncode}): {result.stderr.strip()}"
                )
                tqdm.write(f"[{mongo_id}] Command: {command}")
            else:
                processed += 1

        except subprocess.TimeoutExpired:
            tqdm.write(f"[{mongo_id}] Error: Command timeout")
            tqdm.write(f"[{mongo_id}] Command: {command}")
        except Exception as e:
            tqdm.write(f"[{mongo_id}] Error: {str(e)}")
            tqdm.write(f"[{mongo_id}] Command: {command}")

    # Print summary
    print(f"\nSummary: {processed} processed, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()