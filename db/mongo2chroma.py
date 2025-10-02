#!/usr/bin/env python3
"""
mongo2chroma.py
Load cleaned articles from MongoDB into Chroma vector DB
and query them by text similarity.
"""

import os
import sys
import argparse
from typing import List, Dict
from datetime import datetime, timedelta

import pymongo
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Config – change if necessary
# ------------------------------------------------------------------
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB       = "rssnews"
MONGO_COLL     = "articles"

CHROMA_PATH    = "./chroma_db"          # persisted to disk
CHROMA_COLL    = "articles"
EMBED_MODEL    = "all-MiniLM-L6-v2"     # 384-dim, ~50 MB
BATCH_SIZE     = 64
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll   = mongo_client[MONGO_DB][MONGO_COLL]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Create collection idempotently
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLL,
    metadata={"hnsw:space": "cosine"}
)

encoder = SentenceTransformer(EMBED_MODEL)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def parse_date_arg(date_str: str) -> datetime:
    """
    Parse date argument. Supports:
    - Negative integers (e.g., '-1' = 1 day ago, '-7' = 7 days ago)
    - ISO-8601 date strings (e.g., '2025-09-06T08:00:58+00:00')
    """
    if date_str.startswith('-') and date_str[1:].isdigit():
        days_ago = int(date_str)
        return datetime.now() + timedelta(days=days_ago)
    else:
        return datetime.fromisoformat(date_str)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def load_into_chroma(limit: int = None) -> int:
    """
    Embed every article that has `article` text and add into Chroma.
    Already-existing IDs are kept (not overwritten).
    Returns number of vectors stored.
    """
    q = {"article": {"$exists": True, "$ne": None}}

    cursor = mongo_coll.find(q, {"_id": 1, "article": 1}).limit(limit) if limit else mongo_coll.find(q, {"_id": 1, "article": 1})

    docs, ids = [], []
    stored = 0

    for doc in tqdm(cursor, desc="Loading articles"):
        _id  = str(doc["_id"])
        text = doc["article"]
        if not text.strip():
            continue

        # Skip if ID already exists in Chroma
        existing = collection.get(ids=[_id])
        if existing["ids"]:
            continue

        docs.append(text)
        ids.append(_id)

        # Batch-encode for speed
        if len(docs) >= BATCH_SIZE:
            embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
            collection.add(
                documents=docs,
                embeddings=embs,
                ids=ids
            )
            stored += len(ids)
            docs, ids = [], []

    # Final leftovers
    if docs:
        embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
        collection.add(documents=docs, embeddings=embs, ids=ids)
        stored += len(ids)

    return stored


def query_chroma(text: str,
                 n: int = 5,
                 start_date: str = None,
                 end_date: str = None) -> List[Dict[str, str]]:
    """
    Return the `n` most similar articles as:
        [{"id": <mongo _id>, "text": <article body>}, ...]
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    """
    # 1. Build the Mongo-side filter for the candidate set
    q = {"article": {"$exists": True, "$ne": None}}

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # 2. Pull only the IDs that satisfy the filter
    allowed_ids = {str(d["_id"]) for d in mongo_coll.find(q, {"_id": 1})}

    # 3. Run the vector search
    emb = encoder.encode(text, convert_to_tensor=True).cpu().numpy().tolist()
    res = collection.query(
        query_embeddings=[emb],
        n_results=n * 3,               # over-fetch in case many fall outside the date window
        include=["documents", "distances"]
    )

    # 4. Keep only hits that respect the date window
    filtered = [
        {"id": _id, "text": doc}
        for _id, doc in zip(res["ids"][0], res["documents"][0])
        if _id in allowed_ids
    ]
    return filtered[:n]


def export_titles(start_date: str = None, end_date: str = None) -> None:
    """
    Export tab-delimited published date, source, MongoDB ID, and article title to stdout.
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    """
    # Build the query filter
    q = {}
    
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter
    
    # Fetch documents with _id, title, published, and source
    cursor = mongo_coll.find(q, {"_id": 1, "title": 1, "published": 1, "source": 1})
    
    # Write tab-delimited output
    for doc in cursor:
        _id = str(doc["_id"])
        title = doc.get("title", "")
        source = doc.get("source", "")
        published = doc.get("published")
        
        # Format published date as YYYY-MM-DD
        if published and isinstance(published, datetime):
            published_str = published.strftime("%Y-%m-%d")
        else:
            published_str = ""
        
        # Escape tabs and newlines in title and source
        title = title.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        source = source.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        
        print(f"{published_str}\t{source}\t{_id}\t{title}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="MongoDB → Chroma vector loader / querier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="Embed all articles into Chroma")
    p_load.add_argument("-l", "--limit", type=int, help="Only process N articles")

    p_query = sub.add_parser("query", help="Search articles by text")
    p_query.add_argument("text", help="Query string")
    p_query.add_argument("-n", "--top", type=int, default=13, help="How many results to return")
    p_query.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_query.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")

    p_title = sub.add_parser("title", help="Export tab-delimited MongoDB ID and article title")
    p_title.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_title.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")

    args = parser.parse_args(argv)

    if args.cmd == "load":
        count = load_into_chroma(limit=args.limit)
        print(f"✅  Stored {count} new vectors in Chroma")
        return

    if args.cmd == "query":
        hits = query_chroma(args.text,
                            n=args.top,
                            start_date=args.start_date,
                            end_date=args.end_date)
        for h in hits:
            print("---")
            print(f"ID: {h['id']}")
            print(f"Text: {h['text']}")
        return

    if args.cmd == "title":
        export_titles(start_date=args.start_date,
                     end_date=args.end_date)
        return


if __name__ == "__main__":
    main()