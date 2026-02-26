#!/usr/bin/env python3
"""
clean_article_text.py

Cleans article text in MongoDB by:
1. Removing lines that contain only a single word
2. Removing double newlines (replacing with single newlines)

Provides a detailed report of changes made.
"""

import os
import re
import sys
from datetime import datetime

import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId

# ------------------------------------------------------------------
# Config — change if necessary
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

BATCH_SIZE = 100


def connect_to_mongo():
    """Connect to MongoDB and return the collection."""
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print(f"Connected to MongoDB at {MONGO_URI}")
        return client[MONGO_DB][MONGO_COLL]
    except ServerSelectionTimeoutError as e:
        print(f"Server selection timeout: {e}")
        sys.exit(1)


def is_single_word_line(line: str) -> bool:
    """Check if a line contains only a single word (after stripping whitespace)."""
    stripped = line.strip()
    if not stripped:
        return False  # Empty lines are not single word lines
    # Split by whitespace and check if there's only one token
    words = stripped.split()
    return len(words) == 1


def clean_article_text(article: str) -> tuple[str, dict]:
    """
    Clean article text by:
    1. Removing lines that are only one word
    2. Removing double newlines
    
    Returns the cleaned article and a report of changes.
    """
    if not article:
        return article, {"unchanged": True, "reason": "empty article"}
    
    lines = article.split('\n')
    original_line_count = len(lines)
    original_char_count = len(article)
    
    # Track statistics
    stats = {
        "single_word_lines_removed": 0,
        "double_newlines_replaced": 0,
        "original_lines": original_line_count,
        "original_chars": original_char_count,
    }
    
    # Step 1: Remove lines that are only one word
    cleaned_lines = []
    for line in lines:
        if is_single_word_line(line):
            stats["single_word_lines_removed"] += 1
        else:
            cleaned_lines.append(line)
    
    # Step 2: Remove double newlines (replace with single newlines)
    # Join lines and then replace multiple newlines with single newline
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Count double newlines before replacing
    double_newline_count = len(re.findall(r'\n\n+', cleaned_text))
    stats["double_newlines_replaced"] = double_newline_count
    
    # Replace multiple consecutive newlines with a single newline
    cleaned_text = re.sub(r'\n\n+', '\n', cleaned_text)
    
    # Also handle cases where there might be more than 2 newlines
    cleaned_text = re.sub(r'\n{3,}', '\n', cleaned_text)
    
    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    stats["new_lines"] = cleaned_text.count('\n') + 1 if cleaned_text else 0
    stats["new_chars"] = len(cleaned_text)
    
    # Determine if any changes were made
    was_changed = (
        stats["single_word_lines_removed"] > 0 or 
        stats["double_newlines_replaced"] > 0 or
        cleaned_text != article
    )
    
    stats["unchanged"] = not was_changed
    
    return cleaned_text, stats


def process_articles(coll, limit: int | None = None, dry_run: bool = False, batch_size: int = BATCH_SIZE):
    """
    Process all articles in the collection and clean them.
    
    Args:
        coll: MongoDB collection
        limit: Optional limit on number of documents to process
        dry_run: If True, don't actually update the database
        batch_size: Number of documents to process in each batch
    
    Returns:
        Summary report dictionary
    """
    # Find all documents with an article field
    query = {"article": {"$exists": True, "$ne": None}}
    
    # Get total count
    total_docs = coll.count_documents(query)
    print(f"Found {total_docs} documents with 'article' field")
    
    if limit:
        print(f"Processing up to {limit} documents (limit set)")
        total_to_process = min(limit, total_docs)
    else:
        total_to_process = total_docs
    
    # Initialize statistics
    report = {
        "total_documents": total_docs,
        "documents_processed": 0,
        "documents_modified": 0,
        "documents_unchanged": 0,
        "total_single_word_lines_removed": 0,
        "total_double_newlines_replaced": 0,
        "total_chars_before": 0,
        "total_chars_after": 0,
        "errors": [],
        "sample_changes": [],
    }
    
    # Process in batches
    processed = 0
    batch_num = 0
    
    # Use projection to get only _id and article fields
    projection = {"_id": 1, "article": 1}
    
    cursor = coll.find(query, projection)
    if limit:
        cursor = cursor.limit(limit)
    
    batch = []
    for doc in cursor:
        batch.append(doc)
        
        if len(batch) >= batch_size:
            # Process batch
            batch_num += 1
            batch_result = process_batch(
                batch, coll, dry_run, 
                report, 
                max_sample_changes=5
            )
            processed += len(batch)
            batch = []
            
            # Progress update
            progress = (processed / total_to_process) * 100 if total_to_process > 0 else 0
            print(f"Progress: {processed}/{total_to_process} ({progress:.1f}%) - Modified: {report['documents_modified']}")
    
    # Process remaining batch
    if batch:
        batch_num += 1
        process_batch(batch, coll, dry_run, report, max_sample_changes=5)
        processed += len(batch)
    
    report["documents_processed"] = processed
    report["batches_processed"] = batch_num
    
    return report


def process_batch(batch, coll, dry_run, report, max_sample_changes=5):
    """Process a batch of documents."""
    for doc in batch:
        doc_id = doc["_id"]
        article = doc.get("article", "")
        
        if not article:
            continue
        
        # Clean the article
        cleaned_article, stats = clean_article_text(article)
        
        # Update report
        report["total_chars_before"] += stats.get("original_chars", 0)
        report["total_chars_after"] += stats.get("new_chars", 0)
        
        if stats["unchanged"]:
            report["documents_unchanged"] += 1
        else:
            report["documents_modified"] += 1
            report["total_single_word_lines_removed"] += stats["single_word_lines_removed"]
            report["total_double_newlines_replaced"] += stats["double_newlines_replaced"]
            
            # Store sample change (limit to max_sample_changes)
            if len(report["sample_changes"]) < max_sample_changes:
                report["sample_changes"].append({
                    "doc_id": str(doc_id),
                    "single_word_lines_removed": stats["single_word_lines_removed"],
                    "double_newlines_replaced": stats["double_newlines_replaced"],
                    "original_length": stats["original_chars"],
                    "new_length": stats["new_chars"],
                })
            
            # Update MongoDB if not dry run
            if not dry_run:
                try:
                    coll.update_one(
                        {"_id": doc_id},
                        {"$set": {"article": cleaned_article}}
                    )
                except Exception as e:
                    report["errors"].append({
                        "doc_id": str(doc_id),
                        "error": str(e)
                    })


def print_report(report: dict):
    """Print a formatted report of the cleaning operation."""
    print("\n" + "=" * 60)
    print("ARTICLE CLEANING REPORT")
    print("=" * 60)
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total documents in database:     {report['total_documents']:,}")
    print(f"   Documents processed:             {report['documents_processed']:,}")
    print(f"   Documents modified:              {report['documents_modified']:,}")
    print(f"   Documents unchanged:             {report['documents_unchanged']:,}")
    print(f"   Batches processed:               {report['batches_processed']}")
    
    print(f"\n🧹 CLEANING STATISTICS:")
    print(f"   Single-word lines removed:       {report['total_single_word_lines_removed']:,}")
    print(f"   Double newlines replaced:        {report['total_double_newlines_replaced']:,}")
    
    print(f"\n📏 SIZE CHANGES:")
    print(f"   Total characters before:         {report['total_chars_before']:,}")
    print(f"   Total characters after:          {report['total_chars_after']:,}")
    chars_saved = report['total_chars_before'] - report['total_chars_after']
    print(f"   Characters saved:                {chars_saved:,}")
    
    if report['total_chars_before'] > 0:
        pct_saved = (chars_saved / report['total_chars_before']) * 100
        print(f"   Percentage saved:               {pct_saved:.2f}%")
    
    if report['sample_changes']:
        print(f"\n📝 SAMPLE CHANGES (first {len(report['sample_changes'])}):")
        for i, change in enumerate(report['sample_changes'], 1):
            print(f"   {i}. Doc ID: {change['doc_id']}")
            print(f"      - Single-word lines removed: {change['single_word_lines_removed']}")
            print(f"      - Double newlines replaced:   {change['double_newlines_replaced']}")
            print(f"      - Size: {change['original_length']:,} → {change['new_length']:,} chars")
    
    if report['errors']:
        print(f"\n⚠️  ERRORS ({len(report['errors'])}):")
        for error in report['errors'][:10]:  # Show first 10 errors
            print(f"   - {error['doc_id']}: {error['error']}")
        if len(report['errors']) > 10:
            print(f"   ... and {len(report['errors']) - 10} more errors")
    
    print("\n" + "=" * 60)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"article_cleaning_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("ARTICLE CLEANING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total documents in database:     {report['total_documents']:,}\n")
        f.write(f"Documents processed:             {report['documents_processed']:,}\n")
        f.write(f"Documents modified:              {report['documents_modified']:,}\n")
        f.write(f"Documents unchanged:             {report['documents_unchanged']:,}\n")
        f.write(f"Batches processed:               {report['batches_processed']}\n\n")
        f.write(f"Single-word lines removed:       {report['total_single_word_lines_removed']:,}\n")
        f.write(f"Double newlines replaced:        {report['total_double_newlines_replaced']:,}\n\n")
        f.write(f"Total characters before:         {report['total_chars_before']:,}\n")
        f.write(f"Total characters after:          {report['total_chars_after']:,}\n")
        chars_saved = report['total_chars_before'] - report['total_chars_after']
        f.write(f"Characters saved:                {chars_saved:,}\n")
        
        if report['errors']:
            f.write(f"\nErrors: {len(report['errors'])}\n")
            for error in report['errors']:
                f.write(f"  - {error['doc_id']}: {error['error']}\n")
    
    print(f"📄 Report saved to: {report_file}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean article text in MongoDB by removing single-word lines and double newlines"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of documents to process"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't actually update the database, just report what would change"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=MONGO_URI,
        help=f"MongoDB URI (default: {MONGO_URI})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ARTICLE TEXT CLEANER")
    print("=" * 60)
    print(f"MongoDB URI: {args.uri}")
    print(f"Database: {MONGO_DB}")
    print(f"Collection: {MONGO_COLL}")
    print(f"Dry run: {args.dry_run}")
    if args.limit:
        print(f"Limit: {args.limit} documents")
    print("=" * 60)
    print()
    
    # Connect to MongoDB
    coll = connect_to_mongo()
    
    # Process articles
    report = process_articles(
        coll, 
        limit=args.limit, 
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )
    
    # Print report
    print_report(report)
    
    if args.dry_run:
        print("⚠️  DRY RUN - No changes were made to the database")


if __name__ == "__main__":
    main()
