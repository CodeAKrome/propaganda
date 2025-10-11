#!/usr/bin/env python3
"""
mongo2chroma.py
Load cleaned articles from MongoDB into Chroma vector DB
and query them by text similarity.
"""

import os
import sys
import argparse
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

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

CHROMA_PATH    = "./chroma"          # persisted to disk
CHROMA_COLL    = "articles"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"  # 1024-dim, excellent retrieval model
BATCH_SIZE     = 32
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

# Load BGE encoder
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


def parse_entity_spec(entity_spec: str) -> Tuple[str, str]:
    """
    Parse entity specification. Returns (label, text) tuple.
    If no slash, label is None and text is the full spec.
    Example: "LOC/U.S." -> ("LOC", "U.S.")
             "LOC/" -> ("LOC", "")  (all entities of type LOC)
             "Ukraine" -> (None, "Ukraine")
    """
    if '/' in entity_spec:
        label, text = entity_spec.split('/', 1)
        return (label, text)
    else:
        return (None, entity_spec)


def parse_entity_list(entity_str: str) -> List[Tuple[str, str]]:
    """
    Parse comma-separated entity list into list of (label, text) tuples.
    """
    if not entity_str:
        return []
    return [parse_entity_spec(e.strip()) for e in entity_str.split(',')]


def doc_has_entity(doc: Dict, label: str, text: str) -> bool:
    """
    Check if document has the specified entity.
    If label is None, match on text only across all entity types.
    If label is specified, match on both label and text.
    """
    if 'ner' not in doc or 'entities' not in doc['ner']:
        return False
    
    for entity in doc['ner']['entities']:
        entity_text = entity.get('text', '')
        entity_label = entity.get('label', '')
        
        if label is None:
            # Match on text only
            if entity_text == text:
                return True
        else:
            # Match on both label and text
            if entity_label == label and entity_text == text:
                return True
    
    return False


def doc_matches_entity_filters(doc: Dict, and_entities: List[Tuple[str, str]], 
                                or_entities: List[Tuple[str, str]]) -> bool:
    """
    Check if document matches entity filters.
    - Must have ALL entities in and_entities
    - Must have AT LEAST ONE entity in or_entities (if or_entities is not empty)
    """
    # Check AND entities - must have all
    if and_entities:
        for label, text in and_entities:
            if not doc_has_entity(doc, label, text):
                return False
    
    # Check OR entities - must have at least one
    if or_entities:
        has_any = False
        for label, text in or_entities:
            if doc_has_entity(doc, label, text):
                has_any = True
                break
        if not has_any:
            return False
    
    return True


def extract_entities_from_doc(doc: Dict, show_entities: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """
    Extract entities from document that match the show_entities filter.
    Returns dict mapping entity labels to sets of entity texts.
    If show_entities is empty, returns all entities.
    
    Filter rules:
    - (None, "text"): Match entity with text="text" across all types
    - ("LABEL", "text"): Match entity with label="LABEL" and text="text"
    - ("LABEL", ""): Match all entities with label="LABEL"
    """
    result = defaultdict(set)
    
    if 'ner' not in doc or 'entities' not in doc['ner']:
        return result
    
    for entity in doc['ner']['entities']:
        entity_text = entity.get('text', '')
        entity_label = entity.get('label', '')
        
        if not show_entities:
            # Show all entities
            result[entity_label].add(entity_text)
        else:
            # Check if this entity matches any in show_entities
            for show_label, show_text in show_entities:
                if show_label is None:
                    # Match on text only
                    if entity_text == show_text:
                        result[entity_label].add(entity_text)
                elif show_text == "":
                    # Match all entities of this type (e.g., "LOC/")
                    if entity_label == show_label:
                        result[entity_label].add(entity_text)
                else:
                    # Match on both label and text
                    if entity_label == show_label and entity_text == show_text:
                        result[entity_label].add(entity_text)
    
    return result


def format_entities(entities_by_type: Dict[str, Set[str]]) -> str:
    """
    Format entities for display.
    """
    if not entities_by_type:
        return "<entities>\n(none)\n</entities>"
    
    lines = ["<entities>"]
    for label in sorted(entities_by_type.keys()):
        entity_list = sorted(entities_by_type[label])
        lines.append(f"{label}: {', '.join(entity_list)}")
    lines.append("</entities>")
    
    return '\n'.join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def load_into_chroma(limit: int = None) -> int:
    """
    Embed every article that has `article` text **and no fetch_error**
    and add into Chroma. Already-existing IDs are kept (not overwritten).
    Returns number of vectors stored.
    """
    # --- FILTER: ignore any doc that has a non-empty fetch_error ---
    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},   # field missing
            {"fetch_error": {"$in": [None, ""]}}   # field nil or empty string
        ]
    }
    # ---------------------------------------------------------------------

    cursor = mongo_coll.find(q, {"_id": 1, "article": 1}).limit(limit) if limit \
        else mongo_coll.find(q, {"_id": 1, "article": 1})

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
            collection.add(documents=docs, embeddings=embs, ids=ids)
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
                 end_date: str = None,
                 and_entities: List[Tuple[str, str]] = None,
                 or_entities: List[Tuple[str, str]] = None,
                 show_entities: List[Tuple[str, str]] = None) -> List[Dict[str, str]]:
    """
    Return the `n` most similar articles as:
        [{"id": <mongo _id>, "text": <article body>, "entities": <formatted entities>}, ...]
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    Optional entity filters:
    - and_entities: must have ALL of these entities
    - or_entities: must have AT LEAST ONE of these entities
    - show_entities: display these entities in results (empty list = show all)
    """
    and_entities = and_entities or []
    or_entities = or_entities or []
    show_entities = show_entities or []
    
    # 1. Build the Mongo-side filter for the candidate set
    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ]
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # 2. Pull docs with IDs and NER data for entity filtering
    projection = {"_id": 1}
    if and_entities or or_entities or show_entities:
        projection["ner"] = 1
    
    all_docs = list(mongo_coll.find(q, projection))
    
    # 3. Apply entity filters
    if and_entities or or_entities:
        filtered_docs = [
            doc for doc in all_docs
            if doc_matches_entity_filters(doc, and_entities, or_entities)
        ]
    else:
        filtered_docs = all_docs
    
    allowed_ids = {str(d["_id"]) for d in filtered_docs}
    
    # Store full docs for entity extraction later
    docs_by_id = {str(d["_id"]): d for d in filtered_docs}

    # 4. Run the vector search
    # BGE models benefit from query instruction prefix
    query_text = f"Represent this sentence for searching relevant passages: {text}"
    emb = encoder.encode(query_text, convert_to_tensor=True).cpu().numpy().tolist()
    res = collection.query(
        query_embeddings=[emb],
        n_results=n * 3,               # over-fetch in case many fall outside filters
        include=["documents", "distances"]
    )

    # 5. Keep only hits that respect all filters and extract entities
    filtered = []
    for _id, doc_text in zip(res["ids"][0], res["documents"][0]):
        if _id in allowed_ids:
            result = {"id": _id, "text": doc_text}
            
            # Add entities if show_entities is specified
            if show_entities is not None:
                mongo_doc = docs_by_id.get(_id, {})
                entities = extract_entities_from_doc(mongo_doc, show_entities)
                result["entities"] = format_entities(entities)
            
            filtered.append(result)
            
            if len(filtered) >= n:
                break
    
    return filtered


def export_titles(start_date: str = None, end_date: str = None) -> None:
    """
    Export tab-delimited published date, source, MongoDB ID, and article title to stdout.
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    """
    # Build the query filter
    q = {
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ]
    }
    
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
    p_query.add_argument("--andentity", help="Comma-separated entities (all required). Format: [LABEL/]TEXT")
    p_query.add_argument("--orentity", help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT")
    p_query.add_argument("--showentity", help="Comma-separated entities to display. Format: [LABEL/]TEXT or LABEL/ for all of type (empty = show all)")

    p_title = sub.add_parser("title", help="Export tab-delimited MongoDB ID and article title")
    p_title.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_title.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")

    args = parser.parse_args(argv)

    if args.cmd == "load":
        count = load_into_chroma(limit=args.limit)
        print(f"✅  Stored {count} new vectors in Chroma")
        return

    if args.cmd == "query":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []
        show_entities = parse_entity_list(args.showentity) if args.showentity else None
        
        hits = query_chroma(args.text,
                            n=args.top,
                            start_date=args.start_date,
                            end_date=args.end_date,
                            and_entities=and_entities,
                            or_entities=or_entities,
                            show_entities=show_entities)
        for h in hits:
            print("---")
            print(f"ID: {h['id']}")
            if 'entities' in h:
                print(h['entities'])
            print(f"Text: {h['text']}")
        return

    if args.cmd == "title":
        export_titles(start_date=args.start_date,
                     end_date=args.end_date)
        return


if __name__ == "__main__":
    main()