#!/usr/bin/env python3

"""
Database utility module for managing and accessing data.
"""
"""
llm_analysis.py
Load articles from MongoDB and produce full records with bias analysis
for LLM ingestion. Uses the bias analyzer MCP server.

Usage:
    python llm_analysis.py --output records.jsonl [--limit N] [--start-date -7]
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta

import pymongo
from bson import ObjectId

# ------------------------------------------------------------------
# Config — same as mongo2chroma.py
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

BATCH_SIZE = 32  # Number of articles to process per bias analysis batch
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
mongo_coll = mongo_db[MONGO_COLL]


# ------------------------------------------------------------------
# Helper functions (from mongo2chroma.py)
# ------------------------------------------------------------------
def parse_date_arg(date_str: str) -> datetime:
    """
    Parse date argument. Supports:
    - Negative integers (e.g., '-1' = 1 day ago, '-7' = 7 days ago)
    - ISO-8601 date strings (e.g., '2025-09-06T08:00:58+00:00')
    """
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


def parse_id_file(filepath: str) -> List[str]:
    """Read MongoDB IDs from a file, one per line. Lines starting with # are skipped."""
    ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


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


def extract_entities_from_doc(doc: Dict) -> List[Dict]:
    """Extract entities from a MongoDB document."""
    ner = doc.get("ner", {})
    return ner.get("entities", [])


def format_entities(entities: List[Dict]) -> str:
    """Format entities as comma-separated text list."""
    if not entities:
        return ""
    return ", ".join([e.get("text", "") for e in entities if e.get("text")])


# ------------------------------------------------------------------
# Bias Analysis - uses subprocess to call the bias analyzer
# ------------------------------------------------------------------
def analyze_bias_batch(texts: List[str]) -> List[Dict]:
    """
    Analyze bias for a batch of texts.
    
    This function tries multiple methods in order:
    1. Try to use the T5 bias server via HTTP
    2. Fall back to keyword-based heuristic
    
    Args:
        texts: List of article texts to analyze
        
    Returns:
        List of bias analysis results with direction, degree, and reasoning
    """
    if not texts:
        return []
    
    # First try: T5 bias server (HTTP API)
    try:
        import requests
        results = []
        for text in texts:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": text[:5000]},
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "direction": result.get("direction", result.get("dir", {"L": 0.33, "C": 0.34, "R": 0.33})),
                    "degree": result.get("degree", result.get("deg", {"L": 0.33, "M": 0.34, "H": 0.33})),
                    "reasoning": result.get("reasoning", result.get("reason", ""))
                })
            else:
                results.append({
                    "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                    "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                    "reasoning": f"API error: {response.status_code}"
                })
        print(f"Analyzed {len(texts)} articles via T5 server")
        return results
    except Exception as e:
        print(f"T5 server not available: {e}")
    
    # Fallback: Keyword-based heuristic
    print("Using keyword-based bias analysis...")
    return analyze_bias_fallback(texts)


def analyze_bias_fallback(texts: List[str]) -> List[Dict]:
    """
    Fallback bias analysis when T5 server is not available.
    Uses a simple heuristic based on keywords (for demo purposes).
    
    In production, this should call an actual bias detection model.
    """
    results = []
    
    # Keywords associated with political leanings
    left_keywords = ["progressive", "liberal", "democrat", "equality", "climate", 
                     "healthcare", "social", "workers", "unions", "immigration",
                     "environment", "renewable", "green", "welfare", "civil rights"]
    right_keywords = ["conservative", "republican", "tradition", "freedom", "capitalism",
                      "business", "taxes", "border", "family", "faith", "gun", 
                      "second amendment", "free market", "deregulation", "austerity"]
    
    center_keywords = ["report", "according to", "officials said", "statement",
                      "announced", "conference", "meeting", "committee"]
    
    for text in texts:
        text_lower = text.lower()
        
        left_count = sum(1 for kw in left_keywords if kw in text_lower)
        right_count = sum(1 for kw in right_keywords if kw in text_lower)
        center_count = sum(1 for kw in center_keywords if kw in text_lower)
        
        total = left_count + right_count + center_count
        if total == 0:
            # Default to center
            results.append({
                "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
                "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
                "reasoning": "Insufficient signals for bias detection - defaulted to center"
            })
        else:
            # Calculate direction
            left_pct = left_count / total
            right_pct = right_count / total
            center_pct = center_count / total
            
            # Normalize to sum to 1
            dir_total = left_pct + right_pct + center_pct
            left_pct = left_pct / dir_total if dir_total > 0 else 0.33
            right_pct = right_pct / dir_total if dir_total > 0 else 0.33
            center_pct = center_pct / dir_total if dir_total > 0 else 0.34
            
            if left_pct > right_pct and left_pct > center_pct:
                direction = {"L": round(left_pct * 0.8 + 0.1, 2), "C": round(center_pct * 0.5, 2), "R": round(max(0, 1 - left_pct * 0.8 - center_pct * 0.5 - 0.1), 2)}
            elif right_pct > left_pct and right_pct > center_pct:
                direction = {"L": round(max(0, 1 - right_pct * 0.8 - center_pct * 0.5 - 0.1), 2), "C": round(center_pct * 0.5, 2), "R": round(right_pct * 0.8 + 0.1, 2)}
            else:
                direction = {"L": round(left_pct, 2), "C": round(center_pct + 0.1, 2), "R": round(right_pct, 2)}
            
            # Normalize direction
            dir_sum = direction["L"] + direction["C"] + direction["R"]
            direction = {k: round(v / dir_sum, 2) for k, v in direction.items()}
            
            # Calculate degree based on signal strength
            if total < 3:
                degree = {"L": 0.2, "M": 0.6, "H": 0.2}
            elif total < 5:
                degree = {"L": 0.2, "M": 0.5, "H": 0.3}
            else:
                degree = {"L": 0.15, "M": 0.4, "H": 0.45}
            
            results.append({
                "direction": direction,
                "degree": degree,
                "reasoning": f"Keyword analysis: {left_count} left, {right_count} right, {center_count} neutral signals"
            })
    
    return results


# ------------------------------------------------------------------
# Main Functions
# ------------------------------------------------------------------
def get_articles(
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    and_entities: Optional[List[Tuple[str | None, str]]] = None,
    or_entities: Optional[List[Tuple[str | None, str]]] = None,
    id_file: Optional[str] = None,
    text_search: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch articles from MongoDB with optional filtering.
    
    Args:
        limit: Maximum number of articles to return
        start_date: Start date filter (ISO format or -N days)
        end_date: End date filter (ISO format or -N days)
        and_entities: Entity filter - all must match
        or_entities: Entity filter - at least one must match
        id_file: File containing specific MongoDB IDs to fetch
        text_search: Full-text search query on article field
        
    Returns:
        List of article documents with full metadata
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
    if text_search and text_search.strip():
        # Use regex for broader matching
        q["article"] = {"$regex": text_search.strip(), "$options": "i"}
    
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
    
    # Add ID filtering from file
    if id_file:
        ids = parse_id_file(id_file)
        if ids:
            q["_id"] = {"$in": [ObjectId(i) for i in ids]}
    
    # Build projection - include all fields for full analysis
    projection = {
        "_id": 1,
        "title": 1,
        "source": 1,
        "published": 1,
        "ner": 1,
        "article": 1,
        "url": 1,
        "bias": 1,  # Include existing bias if present
    }
    
    # Execute query
    cursor = mongo_coll.find(q, projection)
    
    # Sort by published date (newest first)
    cursor = cursor.sort("published", -1)
    
    # Apply limit (0 or None means no limit)
    if limit:
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
        
        # Get existing bias if present
        existing_bias = doc.get("bias")
        
        results.append({
            "id": str(doc["_id"]),
            "title": doc.get("title", ""),
            "published": published_iso,
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
            "text": doc.get("article", ""),
            "entities": entity_texts,
            "entities_raw": entities,
            "existing_bias": existing_bias,
        })
    
    return results


def build_full_record(article: Dict, bias_result: Dict) -> Dict:
    """
    Build a full record for LLM analysis including all metadata and bias.
    
    Args:
        article: Article data from MongoDB
        bias_result: Bias analysis result
        
    Returns:
        Full record dictionary ready for LLM ingestion
    """
    return {
        "id": article["id"],
        "title": article["title"],
        "published": article["published"],
        "source": article["source"],
        "url": article["url"],
        "text": article["text"],
        "entities": article["entities"],
        "entities_raw": article["entities_raw"],
        "bias": {
            "direction": bias_result.get("direction", {}),
            "degree": bias_result.get("degree", {}),
            "reasoning": bias_result.get("reasoning", "")
        }
    }


def normalize_bias_result(bias: Dict) -> Dict:
    """
    Normalize bias result from various formats to standard format.
    
    Handles different field names from various sources:
    - dir/direction -> direction
    - deg/degree -> degree
    - reason/reasoning -> reasoning
    """
    if not bias:
        return {
            "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
            "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
            "reasoning": "No bias data"
        }
    
    # Handle nested structure
    if isinstance(bias, dict):
        direction = bias.get("direction") or bias.get("dir") or {}
        degree = bias.get("degree") or bias.get("deg") or {}
        reasoning = bias.get("reasoning") or bias.get("reason") or ""
        
        return {
            "direction": direction if direction else {"L": 0.33, "C": 0.34, "R": 0.33},
            "degree": degree if degree else {"L": 0.33, "M": 0.34, "H": 0.33},
            "reasoning": reasoning if reasoning else "From MongoDB record"
        }
    
    return {
        "direction": {"L": 0.33, "C": 0.34, "R": 0.33},
        "degree": {"L": 0.33, "M": 0.34, "H": 0.33},
        "reasoning": "Invalid bias format"
    }


def write_output(records: List[Dict], output_path: str, format: str = "jsonl"):
    """
    Write records to output file.
    
    Args:
        records: List of full record dictionaries
        output_path: Path to output file
        format: Output format - 'jsonl' (JSON Lines) or 'json' (pretty JSON array)
    """
    if format == "jsonl":
        with open(output_path, "w") as f:
            for record in records:
                # Remove raw entities from JSONL output to save space
                output_record = {k: v for k, v in record.items() if k != "entities_raw"}
                f.write(json.dumps(output_record) + "\n")
        print(f"Wrote {len(records)} records to {output_path} (JSONL)")
    else:
        # Pretty JSON
        output_records = [{k: v for k, v in r.items() if k != "entities_raw"} for r in records]
        with open(output_path, "w") as f:
            json.dump(output_records, f, indent=2)
        print(f"Wrote {len(records)} records to {output_path} (JSON)")


def main():
    parser = argparse.ArgumentParser(
        description="Load articles from MongoDB and produce full records with bias analysis."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_records.jsonl",
        help="Output file path (default: llm_records.jsonl)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format: jsonl (one JSON per line) or json (pretty printed array)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of articles to process (default: 1000, use 0 for all)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (ISO format or negative days, e.g., '-7' for 7 days ago)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (ISO format or negative days, e.g., '-1' for yesterday)"
    )
    parser.add_argument(
        "--andentity",
        type=str,
        help="AND entity filter (comma-separated, format: 'label/text' or just 'text')"
    )
    parser.add_argument(
        "--orentity",
        type=str,
        help="OR entity filter (comma-separated, format: 'label/text' or just 'text')"
    )
    parser.add_argument(
        "--id-file",
        type=str,
        help="File containing MongoDB IDs to process (one per line, # for comments)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Full-text search query on article field"
    )
    parser.add_argument(
        "--skip-existing-bias",
        action="store_true",
        help="Skip articles that already have bias data in MongoDB"
    )
    
    args = parser.parse_args()
    
    # Parse entity filters
    and_entities = parse_entity_list(args.andentity) if args.andentity else []
    or_entities = parse_entity_list(args.orentity) if args.orentity else []
    
    print("=" * 60)
    print("LLM Analysis - Full Record Generator")
    print("=" * 60)
    print(f"Output file: {args.output}")
    print(f"Format: {args.format}")
    print(f"Limit: {args.limit}")
    print(f"Start date: {args.start_date or 'none'}")
    print(f"End date: {args.end_date or 'none'}")
    print(f"AND entities: {and_entities}")
    print(f"OR entities: {or_entities}")
    print(f"ID file: {args.id_file or 'none'}")
    print(f"Text search: {args.text or 'none'}")
    print(f"Skip existing bias: {args.skip_existing_bias}")
    print("-" * 60)
    
    # Fetch articles
    print("Fetching articles from MongoDB...")
    articles = get_articles(
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        and_entities=and_entities,
        or_entities=or_entities,
        id_file=args.id_file,
        text_search=args.text,
    )
    
    # Filter out articles with existing bias if requested
    if args.skip_existing_bias:
        articles = [a for a in articles if not a.get("existing_bias")]
        print(f"After filtering for missing bias: {len(articles)} articles")
    
    print(f"Found {len(articles)} articles to process")
    
    if not articles:
        print("No articles found. Writing empty output file.")
        write_output([], args.output, args.format)
        return
    
    # Analyze bias in batches
    print("\nAnalyzing bias for articles...")
    all_records = []
    texts_to_analyze = []
    article_indices = []
    
    for i, article in enumerate(articles):
        # Use existing bias if available, otherwise queue for analysis
        if article.get("existing_bias"):
            # Use existing bias data - normalize to standard format
            bias_result = normalize_bias_result(article["existing_bias"])
            record = build_full_record(article, bias_result)
            all_records.append(record)
        else:
            # Queue text for bias analysis
            texts_to_analyze.append(article["text"][:8000])  # Limit text length
            article_indices.append(i)
    
    # Analyze texts that don't have existing bias
    if texts_to_analyze:
        print(f"Analyzing {len(texts_to_analyze)} new articles for bias...")
        
        # Process in batches
        for i in range(0, len(texts_to_analyze), BATCH_SIZE):
            batch = texts_to_analyze[i:i + BATCH_SIZE]
            batch_results = analyze_bias_batch(batch)
            
            # Assign results back to articles
            for j, result in enumerate(batch_results):
                article_idx = article_indices[i + j]
                record = build_full_record(articles[article_idx], result)
                all_records.append(record)
            
            print(f"  Processed {min(i + BATCH_SIZE, len(texts_to_analyze))}/{len(texts_to_analyze)}")
    
    # Sort by published date (newest first)
    all_records.sort(key=lambda x: x["published"] or "", reverse=True)
    
    # Write output
    print(f"\nWriting output to {args.output}...")
    write_output(all_records, args.output, args.format)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Output: {args.output}")
    print(f"Records: {len(all_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
