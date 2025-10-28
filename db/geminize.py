#!/usr/bin/env python3
"""
mongo_gemini_rag.py
Process MongoDB articles through Google Gemini with RAG input.
Fetches source field content, appends to query prompt, sends to Gemini,
and stores results in destination field.
"""

import os
import sys
import argparse
from typing import Optional, List
from datetime import datetime, timedelta

import pymongo
from bson import ObjectId
import google.generativeai as genai
from tqdm import tqdm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"
DEFAULT_MODEL = "models/gemini-2.5-flash"

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# Ensure the GEMINI_API_KEY environment variable is set
if not os.environ.get("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

# Configure the API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# MongoDB connection
mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def parse_date_arg(date_str: str) -> datetime:
    """
    Parse date argument. Supports:
    - Negative integers (e.g., '-1' = 1 day ago, '-7' = 7 days ago)
    - ISO-8601 date strings (e.g., '2025-08-28T07:50:13.000Z')
    """
    if date_str.startswith("-") and date_str[1:].isdigit():
        days_ago = int(date_str)
        return datetime.now() + timedelta(days=days_ago)
    else:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))


def build_mongo_query(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    news_sources: Optional[List[str]] = None,
    src_field: str = "article",
) -> dict:
    """
    Build MongoDB query filter.
    """
    q = {
        src_field: {"$exists": True, "$ne": None, "$ne": ""},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

    # Add date filter
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Add news source filter
    if news_sources:
        q["source"] = {"$in": news_sources}

    return q


def process_articles(
    model_name: str,
    src_field: str,
    dst_field: str,
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    news_sources: Optional[List[str]] = None,
    limit: Optional[int] = None,
    id_file: str = "ids.txt",
    dry_run: bool = False,
) -> dict:
    """
    Process articles through Gemini LLM.
    Returns statistics about the processing run.
    """
    # Build query
    mongo_query = build_mongo_query(start_date, end_date, news_sources, src_field)
    
    # Count documents
    print("Counting documents to process...")
    total_docs = mongo_coll.count_documents(mongo_query)
    
    if total_docs == 0:
        print("No documents found matching criteria.")
        return {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "model": model_name,
            "modified_ids": [],
        }
    
    print(f"Found {total_docs} documents to process")
    
    # Apply limit if specified
    if limit:
        total_docs = min(total_docs, limit)
        print(f"Limiting to {total_docs} documents")
    
    # Initialize Gemini model
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model '{model_name}': {e}")
        sys.exit(1)
    
    # Fetch documents
    cursor = mongo_coll.find(
        mongo_query,
        {"_id": 1, "title": 1, "source": 1, "published": 1, src_field: 1, dst_field: 1}
    ).sort("published", -1)
    
    # Apply limit if specified
    if limit:
        cursor = cursor.limit(limit)
    
    # Statistics
    stats = {
        "total": total_docs,
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "model": model_name,
        "modified_ids": [],
    }
    
    # Process each document
    for doc in tqdm(cursor, total=total_docs, desc="Processing articles"):
        _id = doc["_id"]
        src_content = doc.get(src_field, "")
        
        # Skip if source field is empty
        if not src_content or not src_content.strip():
            stats["skipped"] += 1
            continue
        
        # Skip if destination field already exists and has content
        if dst_field in doc and doc[dst_field]:
            stats["skipped"] += 1
            continue
        
        # Build prompt with RAG input
        title = doc.get("title", "")
        source = doc.get("source", "")
        published = doc.get("published", "")
        
        # Format published date if it's a datetime object
        if isinstance(published, datetime):
            published = published.isoformat()
        
        prompt = f"""Query: {query}

Article Title: {title}
Source: {source}
Published: {published}

Article Content:
{src_content}
"""
        
        # Process through Gemini
        try:
            response = model.generate_content(prompt)
            result_text = response.text
            
            # Clean up response - remove markdown code fences and common wrappers
            result_text_cleaned = result_text.strip()
            
            # Remove markdown code blocks (```json, ```python, etc.)
            if result_text_cleaned.startswith("```"):
                lines = result_text_cleaned.split("\n")
                # Remove first line if it's a code fence
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove last line if it's a closing code fence
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result_text_cleaned = "\n".join(lines).strip()
            
            # Prepare update data
            update_timestamp = datetime.now()
            update_data = {
                dst_field: result_text_cleaned,
                f"{dst_field}_model": model_name,
                f"{dst_field}_timestamp": update_timestamp,
            }
            
            # Print what will be/was updated
            print(f"\n{'=' * 70}")
            print(f"{'[DRY RUN] ' if dry_run else ''}Article ID: {_id}")
            print(f"Title: {title}")
            print(f"Source: {source}")
            print(f"Published: {published}")
            print(f"\nPrompt sent to Gemini ({len(prompt)} chars):")
            print("-" * 70)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("-" * 70)
            print(f"\nGemini Response ({len(result_text_cleaned)} chars):")
            print("-" * 70)
            print(result_text_cleaned)
            print("-" * 70)
            print(f"\n{'Would insert' if dry_run else 'Inserted'} into MongoDB:")
            print(f"  {dst_field}: {result_text_cleaned[:100]}..." if len(result_text_cleaned) > 100 else f"  {dst_field}: {result_text_cleaned}")
            print(f"  {dst_field}_model: {model_name}")
            print(f"  {dst_field}_timestamp: {update_timestamp.isoformat()}")
            print("=" * 70)
            
            if not dry_run:
                # Store result in MongoDB
                mongo_coll.update_one(
                    {"_id": _id},
                    {"$set": update_data}
                )
            
            # Track modified ID
            stats["modified_ids"].append(str(_id))
            stats["processed"] += 1
                
        except Exception as e:
            print(f"\nError processing article {_id}: {e}")
            stats["errors"] += 1
            
            # Store error in MongoDB
            if not dry_run:
                mongo_coll.update_one(
                    {"_id": _id},
                    {
                        "$set": {
                            f"{dst_field}_error": str(e),
                            f"{dst_field}_error_timestamp": datetime.now(),
                        }
                    }
                )
    
    # Save modified IDs to file
    if stats["modified_ids"]:
        try:
            with open(id_file, "w") as f:
                for _id in stats["modified_ids"]:
                    f.write(f"{_id}\n")
            print(f"\n✅ Saved {len(stats['modified_ids'])} {'processed' if dry_run else 'modified'} IDs to {id_file}")
        except Exception as e:
            print(f"\n⚠️  Error saving IDs to file: {e}")
    
    return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Process MongoDB articles through Google Gemini with RAG"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)"
    )
    parser.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)"
    )
    parser.add_argument(
        "--news",
        help="Comma-separated list of news sources to filter"
    )
    parser.add_argument(
        "--src",
        default="article",
        help="Source field name in MongoDB (default: article)"
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination field name in MongoDB (required)"
    )
    parser.add_argument(
        "-q", "--query",
        help="Query/prompt for LLM"
    )
    parser.add_argument(
        "--queryfile",
        help="Read query/prompt from file"
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Limit number of records to process"
    )
    parser.add_argument(
        "--idfile",
        default="ids.txt",
        help="File to save modified MongoDB IDs (default: ids.txt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    
    args = parser.parse_args(argv)
    
    # Validate query input - must provide either --query or --queryfile
    if not args.query and not args.queryfile:
        parser.error("Either -q/--query or --queryfile must be provided")
    
    if args.query and args.queryfile:
        parser.error("Cannot specify both -q/--query and --queryfile")
    
    # Read query from file if --queryfile is specified
    query = args.query
    if args.queryfile:
        try:
            with open(args.queryfile, "r") as f:
                query = f.read()
        except FileNotFoundError:
            print(f"Error: Query file '{args.queryfile}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading query file '{args.queryfile}': {e}")
            sys.exit(1)
    
    # Parse news sources
    news_sources = None
    if args.news:
        news_sources = [s.strip() for s in args.news.split(",")]
    
    # Print configuration
    print("=" * 70)
    print("MongoDB to Gemini RAG Processor")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Source field: {args.src}")
    print(f"Destination field: {args.dst}")
    if args.queryfile:
        print(f"Query file: {args.queryfile}")
        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
    else:
        print(f"Query: {query}")
    if args.n:
        print(f"Limit: {args.n} records")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")
    if news_sources:
        print(f"News sources: {', '.join(news_sources)}")
    print(f"ID file: {args.idfile}")
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
    print("=" * 70)
    print()
    
    # Process articles
    stats = process_articles(
        model_name=args.model,
        src_field=args.src,
        dst_field=args.dst,
        query=query,
        start_date=args.start_date,
        end_date=args.end_date,
        news_sources=news_sources,
        limit=args.n,
        id_file=args.idfile,
        dry_run=args.dry_run,
    )
    
    # Print final report
    print("\n" + "=" * 70)
    print("Processing Report")
    print("=" * 70)
    print(f"Model used: {stats['model']}")
    print(f"Total documents found: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (empty or already processed): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 70)
    
    # Exit with error code if there were errors
    if stats['errors'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()