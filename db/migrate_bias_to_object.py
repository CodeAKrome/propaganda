#!/usr/bin/env python3
"""
Migration script to convert bias field from JSON string to MongoDB object.

This script finds all documents where the 'bias' field is a string (legacy format)
and converts it to a proper MongoDB object for better queryability.

Usage:
    python migrate_bias_to_object.py [--dry-run] [--batch-size N] [--idfile FILE] [--failfile FILE]

Example queries after migration:
    # Find all articles with strong left bias (L > 0.5)
    db.articles.find({"bias.dir.L": {"$gt": 0.5}})
    
    # Find all articles with high bias degree (H > 0.5)
    db.articles.find({"bias.deg.H": {"$gt": 0.5}})
    
    # Combined query: strong left bias with high degree
    db.articles.find({"bias.dir.L": {"$gt": 0.5}, "bias.deg.H": {"$gt": 0.5}})
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId


def parse_bias_string(bias_str: str) -> dict | None:
    """
    Parse a bias JSON string into a dictionary.
    
    Args:
        bias_str: JSON string containing bias data
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not bias_str or not isinstance(bias_str, str):
        return None
    
    try:
        parsed = json.loads(bias_str)
        
        # Validate the structure
        if not isinstance(parsed, dict):
            return None
        
        # Check for required fields
        if "dir" not in parsed or "deg" not in parsed:
            return None
        
        return parsed
    except json.JSONDecodeError:
        return None


def migrate_bias_fields(
    dry_run: bool = False, 
    batch_size: int | None = None,
    idfile: str = "migrate_ids.txt",
    failfile: str = "migrate_fail_ids.txt"
):
    """
    Migrate all bias fields from string to object format.
    
    Args:
        dry_run: If True, only report what would be changed without making changes
        batch_size: Optional limit on number of documents to process
        idfile: Path to file for storing migrated MongoDB IDs
        failfile: Path to file for storing failed MongoDB IDs
    """
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
    
    # Find all documents where bias is a string
    query = {"bias": {"$type": "string"}}
    
    total_count = collection.count_documents(query)
    
    if total_count == 0:
        print("No documents found with bias as string. Migration not needed.")
        client.close()
        return
    
    if batch_size:
        total_count = min(total_count, batch_size)
        print(f"Processing up to {total_count} documents (batch limit)...")
    else:
        print(f"Found {total_count} documents with bias as string to migrate.")
    
    if dry_run:
        print("\n[DRY RUN] No changes will be made to the database.\n")
    
    cursor = collection.find(query)
    if batch_size:
        cursor = cursor.limit(batch_size)
    
    migrated = 0
    failed = 0
    skipped = 0
    migrated_ids = []
    failed_ids = []
    
    for doc in tqdm(cursor, total=total_count, desc="Migrating bias fields"):
        doc_id = doc["_id"]
        bias_data = doc.get("bias", "")
        
        # Skip if bias is already an object (not a string)
        if isinstance(bias_data, dict):
            skipped += 1
            tqdm.write(f"  SKIPPED {doc_id} (bias already an object)")
            continue
        
        # Print the MongoDB ID being processed
        tqdm.write(f"  Processing: {doc_id}")
        
        # Parse the bias string
        bias_obj = parse_bias_string(bias_data)
        
        if bias_obj is None:
            failed += 1
            failed_ids.append(str(doc_id))
            tqdm.write(f"  FAILED to parse bias for {doc_id}: {bias_data[:100]}...")
            continue
        
        if dry_run:
            tqdm.write(f"  Would update {doc_id}:")
            tqdm.write(f"    From: {bias_data[:100]}...")
            tqdm.write(f"    To: {json.dumps(bias_obj)[:100]}...")
            migrated += 1
            migrated_ids.append(str(doc_id))
        else:
            # Update the document
            try:
                result = collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"bias": bias_obj}}
                )
                
                if result.modified_count > 0:
                    migrated += 1
                    migrated_ids.append(str(doc_id))
                else:
                    skipped += 1
                    tqdm.write(f"  SKIPPED {doc_id} (no modification needed)")
            except Exception as e:
                failed += 1
                failed_ids.append(str(doc_id))
                tqdm.write(f"  FAILED to update {doc_id}: {e}")
    
    # Write migrated IDs to file
    if migrated_ids:
        try:
            with open(idfile, "w") as f:
                f.write(f"# Migration: {datetime.now().isoformat()}\n")
                f.write(f"# Total migrated: {migrated}\n")
                for mongo_id in migrated_ids:
                    f.write(f"{mongo_id}\n")
            print(f"\nWrote {len(migrated_ids)} migrated IDs to: {idfile}")
        except Exception as e:
            print(f"\nError writing ID file: {e}")
    
    # Write failed IDs to file
    if failed_ids:
        try:
            with open(failfile, "w") as f:
                f.write(f"# Migration: {datetime.now().isoformat()}\n")
                f.write(f"# Total failed: {failed}\n")
                for mongo_id in failed_ids:
                    f.write(f"{mongo_id}\n")
            print(f"Wrote {len(failed_ids)} failed IDs to: {failfile}")
        except Exception as e:
            print(f"Error writing fail file: {e}")
    
    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    
    if dry_run:
        print("\n[DRY RUN] Run without --dry-run to apply changes.")
    
    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate bias field from JSON string to MongoDB object"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--idfile",
        type=str,
        default="migrate_ids.txt",
        help="File to write migrated MongoDB IDs to (default: migrate_ids.txt)",
    )
    parser.add_argument(
        "--failfile",
        type=str,
        default="migrate_fail_ids.txt",
        help="File to write failed MongoDB IDs to (default: migrate_fail_ids.txt)",
    )
    args = parser.parse_args()
    
    try:
        migrate_bias_fields(
            dry_run=args.dry_run, 
            batch_size=args.batch_size,
            idfile=args.idfile,
            failfile=args.failfile
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
