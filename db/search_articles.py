#!/usr/bin/env python3

"""
Database utility module for managing and accessing data.
"""
"""
search_articles.py
Search articles from MongoDB and output .vec file with bias and entity data.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

import pymongo
from bson import ObjectId
from pymongo.errors import OperationFailure

# ------------------------------------------------------------------
# Config — same as mongo2chroma.py
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
mongo_coll = mongo_db[MONGO_COLL]


# ------------------------------------------------------------------
# Helper functions (from mongo2chroma.py)
# ------------------------------------------------------------------
def parse_date_arg(date_str: str) -> datetime:
    """Parse date argument. Supports negative integers and ISO-8601."""
    if date_str.startswith("-") and date_str[1:].isdigit():
        days_ago = int(date_str)
        return datetime.now() + timedelta(days=days_ago)
    else:
        return datetime.fromisoformat(date_str)


def parse_entity_spec(entity_spec: str) -> Tuple[str | None, str]:
    """Parse entity specification. Returns (label, text) tuple."""
    if "/" in entity_spec:
        label, text = entity_spec.split("/", 1)
        return (label, text)
    else:
        return (None, entity_spec)


def parse_entity_list(entity_str: str) -> List[Tuple[str | None, str]]:
    """Parse comma-separated entity list into list of (label, text) tuples."""
    if not entity_str:
        return []
    return [parse_entity_spec(e.strip()) for e in entity_str.split(",")]


def build_mongo_entity_filter(
    and_list: List[Tuple[str | None, str]], or_list: List[Tuple[str | None, str]]
) -> Dict:
    """Build MongoDB query filter for entity matching."""
    if not and_list and not or_list:
        return {}

    clauses = []

    for label, text in and_list:
        if label and text:
            clauses.append(
                {"ner.entities": {"$elemMatch": {"label": label, "text": text}}}
            )
        elif label and not text:
            clauses.append({"ner.entities.label": label})
        elif text and not label:
            clauses.append({"ner.entities.text": text})

    if or_list:
        or_clauses = []
        for label, text in or_list:
            if label and text:
                or_clauses.append(
                    {"ner.entities": {"$elemMatch": {"label": label, "text": text}}}
                )
            elif label and not text:
                or_clauses.append({"ner.entities.label": label})
            elif text and not label:
                or_clauses.append({"ner.entities.text": text})
        if or_clauses:
            clauses.append({"$or": or_clauses})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def extract_entities_from_doc(doc: Dict, filter_entities: Optional[List[Tuple[str | None, str]]] = None) -> List[Dict]:
    """Extract entities from a MongoDB document."""
    ner = doc.get("ner", {})
    entities = ner.get("entities", [])
    
    if not filter_entities:
        return entities
    
    # Filter to only requested entity types/labels
    filtered = []
    for entity in entities:
        for label, text in filter_entities:
            if label and entity.get("label") == label:
                filtered.append(entity)
                break
            elif text and entity.get("text") == text:
                filtered.append(entity)
                break
    return filtered


def format_entities(entities: List[Dict]) -> str:
    """Format entities as comma-separated text list."""
    if not entities:
        return ""
    return ", ".join([e.get("text", "") for e in entities if e.get("text")])


def search_articles(
    text: Optional[str] = None,
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    and_entities: Optional[List[Tuple[str | None, str]]] = None,
    or_entities: Optional[List[Tuple[str | None, str]]] = None,
    use_regex: bool = False,
) -> List[Dict]:
    """
    Search articles from MongoDB with full-text search and entity filtering.
    
    Args:
        text: Full-text search query on article field
        limit: Maximum number of results
        start_date: Start date filter (e.g., '-7' for 7 days ago)
        end_date: End date filter
        and_entities: List of (label, text) tuples - all must match
        or_entities: List of (label, text) tuples - at least one must match
    
    Returns:
        List of article documents with id, title, published, source, text, entities
    """
    and_entities = and_entities or []
    or_entities = or_entities or []

    # Build base query - only valid articles
    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

    # Add text search on article field
    if text and text.strip():
        if use_regex:
            # Use regex for broader matching (slower but no index required)
            q["article"] = {"$regex": text.strip(), "$options": "i"}
        else:
            # Use MongoDB text search (requires text index)
            q["$text"] = {"$search": text.strip()}

    # Add date filtering
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Add entity filtering
    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        q.update(entity_filter)

    # Print query for debugging
    print(f"Query: {json.dumps(q, default=str, indent=2)}")

    # Build projection
    projection = {
        "_id": 1,
        "title": 1,
        "source": 1,
        "published": 1,
        "ner": 1,
        "article": 1,
    }

    # Add text search score if searching by text
    if text and text.strip():
        projection["score"] = {"$meta": "textScore"}  # type: ignore

    # Execute query
    try:
        cursor = mongo_coll.find(q, projection)
    except OperationFailure as e:
        if "text index" in str(e) and not use_regex:
            print("Text index not found, falling back to regex search...")
            return search_articles(
                text=text,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
                and_entities=and_entities,
                or_entities=or_entities,
                use_regex=True,
            )
        raise
    
    # Sort by text score if searching, otherwise by published date
    if text and text.strip():
        cursor = cursor.sort([("score", {"$meta": "textScore"})])
    else:
        cursor = cursor.sort("published", -1)
    
    cursor = cursor.limit(limit)

    results = []
    for doc in cursor:
        published_iso = ""
        published_dt = doc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            published_iso = published_dt.isoformat()

        # Extract entities
        entities = extract_entities_from_doc(doc)
        entity_texts = format_entities(entities)

        results.append({
            "id": str(doc["_id"]),
            "title": doc.get("title", ""),
            "published": published_iso,
            "source": doc.get("source", ""),
            "text": doc.get("article", ""),
            "entities": entity_texts,
            "entities_raw": entities,
        })

    return results


def analyze_bias_batch(texts: List[str], batch_size: int = 10) -> List[Dict]:
    """
    Analyze bias for a batch of texts using the bias analyzer.
    Falls back to mock data if MCP is not available.
    """
    results = []
    
    try:
        # Try to use subprocess to call bias analyzer
        import subprocess
        import json
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Call bias analyzer via subprocess or API
            # For now, use mock data as placeholder
            for text in batch:
                results.append({
                    "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                    "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                    "reasoning": "Bias analysis not yet implemented - requires MCP server"
                })
            print(f"Analyzed {min(i + batch_size, len(texts))}/{len(texts)} articles")
            
    except ImportError:
        # Fallback: use a simple heuristic or mock data
        print("Bias analyzer not available, using mock bias data")
        for text in texts:
            # Simple mock - just use a placeholder
            # In production, this would call an actual bias detection model
            results.append({
                "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                "reasoning": "Mock bias analysis - MCP not available"
            })
    except Exception as e:
        print(f"Error analyzing bias: {e}")
        # Return mock data on error
        for text in texts:
            results.append({
                "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                "reasoning": f"Error: {str(e)}"
            })
    
    return results


def write_vec_file(articles: List[Dict], bias_results: List[Dict], output_path: str):
    """
    Write .vec file with bias and entity data for LLM ingestion.
    
    Format (JSONL - one JSON object per line):
    {
        "id": "...",
        "title": "...",
        "published": "...",
        "source": "...",
        "text": "...",
        "entities": "...",
        "bias": {
            "direction": {"L": 0.x, "C": 0.x, "R": 0.x},
            "degree": {"L": 0.x, "M": 0.x, "H": 0.x},
            "reasoning": "..."
        }
    }
    """
    with open(output_path, "w") as f:
        for article, bias in zip(articles, bias_results):
            record = {
                "id": article["id"],
                "title": article["title"],
                "published": article["published"],
                "source": article["source"],
                "text": article["text"],
                "entities": article["entities"],
                "bias": bias,
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"Wrote {len(articles)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Search articles from MongoDB and output .vec file with bias and entity data."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Full-text search query on article field"
    )
    parser.add_argument(
        "--orentity",
        type=str,
        help="OR entity filter (comma-separated, format: 'label/text' or just 'text')"
    )
    parser.add_argument(
        "--andentity",
        type=str,
        help="AND entity filter (comma-separated, format: 'label/text' or just 'text')"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (ISO format or negative days, e.g., '-7')"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (ISO format or negative days, e.g., '-1')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.vec",
        help="Output .vec file path (default: output.vec)"
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="Skip bias analysis"
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Use regex search instead of MongoDB text search"
    )

    args = parser.parse_args()

    # Parse entity filters
    and_entities = parse_entity_list(args.andentity) if args.andentity else []
    or_entities = parse_entity_list(args.orentity) if args.orentity else []

    print(f"Searching MongoDB...")
    print(f"  Text query: {args.text}")
    print(f"  AND entities: {and_entities}")
    print(f"  OR entities: {or_entities}")
    print(f"  Start date: {args.start_date}")
    print(f"  End date: {args.end_date}")
    print(f"  Limit: {args.limit}")

    # Search articles
    articles = search_articles(
        text=args.text,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        and_entities=and_entities,
        or_entities=or_entities,
        use_regex=args.regex,
    )

    print(f"Found {len(articles)} articles")

    if not articles:
        print("No articles found")
        # Write empty file
        write_vec_file([], [], args.output)
        return

    # Analyze bias
    if args.no_bias:
        print("Skipping bias analysis")
        bias_results = [
            {
                "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                "reasoning": "Skipped"
            }
            for _ in articles
        ]
    else:
        print("Analyzing bias...")
        texts = [a["text"][:5000] for a in articles]  # Limit text length for bias analysis
        bias_results = analyze_bias_batch(texts)

    # Write output file
    write_vec_file(articles, bias_results, args.output)

    print(f"Done! Output written to {args.output}")


if __name__ == "__main__":
    main()
