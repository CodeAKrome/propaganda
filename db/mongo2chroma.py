#!/usr/bin/env python3
"""
mongo2chroma.py
Load cleaned articles from MongoDB into Chroma vector DB
and query them by text similarity.
ChromaDB stores only vectors and IDs. All filtering and data lookup via MongoDB.
"""

import os
import sys
import argparse
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import pymongo
from bson import ObjectId
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Config â€" change if necessary
# ------------------------------------------------------------------
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB       = "rssnews"
MONGO_COLL     = "articles"

CHROMA_PATH    = "./chroma"
CHROMA_COLL    = "articles"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"
BATCH_SIZE     = 32
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll   = mongo_client[MONGO_DB][MONGO_COLL]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Create collection idempotently - no metadata needed
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


def build_mongo_entity_filter(and_list: List[Tuple[str, str]],
                               or_list: List[Tuple[str, str]]) -> Dict:
    """
    Build MongoDB query filter for entity matching.
    """
    if not and_list and not or_list:
        return {}
    
    clauses = []
    
    # AND entities - all must be present
    for label, text in and_list:
        if label and text:
            # Match specific entity with label and text
            clauses.append({
                "ner.entities": {
                    "$elemMatch": {
                        "label": label,
                        "text": text
                    }
                }
            })
        elif label and not text:
            # Match any entity with this label
            clauses.append({
                "ner.entities.label": label
            })
        elif text and not label:
            # Match any entity with this text
            clauses.append({
                "ner.entities.text": text
            })
    
    # OR entities - at least one must be present
    if or_list:
        or_clauses = []
        for label, text in or_list:
            if label and text:
                or_clauses.append({
                    "ner.entities": {
                        "$elemMatch": {
                            "label": label,
                            "text": text
                        }
                    }
                })
            elif label and not text:
                or_clauses.append({
                    "ner.entities.label": label
                })
            elif text and not label:
                or_clauses.append({
                    "ner.entities.text": text
                })
        if or_clauses:
            clauses.append({"$or": or_clauses})
    
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def load_into_chroma(limit: int = None,
                     start_date: str = None,
                     end_date: str = None) -> int:
    """
    Embed every article that has `article` text **and no fetch_error**
    and add into Chroma. Only stores ID and embedding.
    Already-existing IDs are kept (not overwritten).
    Returns number of vectors stored.
    """
    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ]
    }

    # apply date window to Mongo query
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Get total count for tqdm progress bar
    print("Counting documents to load...")
    total_docs = mongo_coll.count_documents(q)
    if limit:
        total_docs = min(total_docs, limit)

    cursor = mongo_coll.find(q, {"_id": 1, "article": 1}).limit(limit) \
        if limit else mongo_coll.find(q, {"_id": 1, "article": 1})

    docs, ids = [], []
    stored = 0

    # Iterate with a progress bar that shows the total
    for doc in tqdm(cursor, total=total_docs, desc="Loading articles"):
        _id = str(doc["_id"])
        text = doc.get("article", "").strip()
        if not text:
            continue

        # Skip if ID already exists in Chroma
        if collection.get(ids=[_id])["ids"]:
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
        [{"id": <mongo _id>, "title": <title>, "published": <ISO date>, "source": <source>, "text": <article body>, "entities": <formatted entities>}, ...]
    
    Process:
    1. Get candidate IDs from ChromaDB (vector search or all IDs if no text)
    2. Apply all filters in MongoDB
    3. Return top N results
    
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    Optional entity filters:
    - and_entities: must have ALL of these entities
    - or_entities: must have AT LEAST ONE of these entities
    - show_entities: display these entities in results (empty list = show all)

    If text is blank and entity filters are specified, returns all matching documents
    without vector similarity scoring.
    """
    and_entities = and_entities or []
    or_entities = or_entities or []
    show_entities = show_entities or []

    # Step 1: Get candidate IDs from ChromaDB
    if text.strip():
        # Vector search - get more candidates than needed for post-filtering
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        emb = encoder.encode(query_text, convert_to_tensor=True).cpu().numpy().tolist()
        res = collection.query(
            query_embeddings=[emb],
            n_results=n * 10,  # Get extra candidates for filtering
            include=["documents"]
        )
        candidate_ids = res["ids"][0]
        # Store documents for later use
        candidate_docs = {_id: doc for _id, doc in zip(res["ids"][0], res["documents"][0])}
    else:
        # No text query - get all IDs from ChromaDB (for pure filter queries)
        res = collection.get(
            limit=10000,  # Get large batch for filtering
            include=["documents"]
        )
        candidate_ids = res["ids"]
        candidate_docs = {_id: doc for _id, doc in zip(res["ids"], res["documents"])}

    if not candidate_ids:
        return []

    # Step 2: Build MongoDB filter
    mongo_filter = {
        "_id": {"$in": [ObjectId(i) for i in candidate_ids]},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ]
    }

    # Add date filter
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        mongo_filter["published"] = date_filter

    # Add entity filter
    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        mongo_filter.update(entity_filter)

    # Step 3: Query MongoDB with filters
    mongo_docs = list(mongo_coll.find(
        mongo_filter,
        {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1}
    ))

    # Step 4: Build results maintaining ChromaDB ranking order
    out = []
    id_to_doc = {str(d["_id"]): d for d in mongo_docs}
    
    for _id in candidate_ids:
        if _id not in id_to_doc:
            continue  # Filtered out by MongoDB
        
        mdoc = id_to_doc[_id]
        
        # Get published date
        published_iso = ""
        published_dt = mdoc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            published_iso = published_dt.isoformat()

        # Get article text from ChromaDB result or MongoDB
        article_text = candidate_docs.get(_id, mdoc.get("article", ""))

        out.append({
            "id":        _id,
            "title":     mdoc.get("title", ""),
            "published": published_iso,
            "source":    mdoc.get("source", ""),
            "text":      article_text,
            "entities":  format_entities(extract_entities_from_doc(mdoc, show_entities))
                         if show_entities is not None else None
        })
        
        if len(out) >= n:
            break
    
    return out


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


def dump_entities(start_date: str = None,
                  end_date: str = None,
                  show_entities: List[Tuple[str, str]] = None) -> None:
    """
    Export tab-delimited list of unique entities sorted by entity type.
    Optional date window:
    - ISO-8601 strings (e.g., '2025-09-06T08:00:58+00:00')
    - Negative integers for relative days (e.g., '-7' = 7 days ago)
    Optional entity filter:
    - show_entities: only show these entities (empty list = show all)
    
    Output format: ENTITY_TYPE\tENTITY_TEXT\tCOUNT
    """
    # Build the query filter
    q = {
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ],
        "ner.entities": {"$exists": True}
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Fetch documents with NER data
    cursor = mongo_coll.find(q, {"ner": 1})

    # Collect entities with counts
    entity_counts = defaultdict(lambda: defaultdict(int))
    
    for doc in cursor:
        entities_in_doc = extract_entities_from_doc(doc, show_entities if show_entities is not None else [])
        
        for entity_type, entity_texts in entities_in_doc.items():
            for entity_text in entity_texts:
                entity_counts[entity_type][entity_text] += 1

    # Sort by entity type, then by entity text
    for entity_type in sorted(entity_counts.keys()):
        for entity_text in sorted(entity_counts[entity_type].keys()):
            count = entity_counts[entity_type][entity_text]
            print(f"{entity_type}\t{entity_text}\t{count}")


def export_articles(start_date: str = None,
                   end_date: str = None,
                   and_entities: List[Tuple[str, str]] = None,
                   or_entities: List[Tuple[str, str]] = None,
                   show_entities: List[Tuple[str, str]] = None) -> None:
    """
    Export articles matching entity filters, formatted like query results.
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

    # Build the query filter
    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}}
        ]
    }

    # Add date filter
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    # Add entity filter
    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        q.update(entity_filter)

    # Query MongoDB
    cursor = mongo_coll.find(
        q,
        {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1}
    )

    # Output results in same format as query command
    for doc in cursor:
        _id = str(doc["_id"])
        title = doc.get("title", "")
        source = doc.get("source", "")
        article_text = doc.get("article", "")
        
        # Get published date
        published_iso = ""
        published_dt = doc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            published_iso = published_dt.isoformat()

        print("---")
        print(f"ID: {_id}")
        print(f"Title: {title}")
        print(f"Published: {published_iso}")
        print(f"Source: {source}")
        
        # Show entities if requested
        if show_entities is not None:
            entities = format_entities(extract_entities_from_doc(doc, show_entities))
            print(entities)
        
        print(f"Text: {article_text}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="MongoDB â†' Chroma vector loader / querier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="Embed all articles into Chroma")
    p_load.add_argument("-l", "--limit", type=int, help="Only process N articles")
    p_load.add_argument("--start-date", help="Load articles published on/after this date (ISO or -N)")
    p_load.add_argument("--end-date",   help="Load articles published on/before this date (ISO or -N)")

    p_query = sub.add_parser("query", help="Search articles by text")
    p_query.add_argument("text", nargs='?', default='', help="Query string (optional if using entity filters)")
    p_query.add_argument("-n", "--top", type=int, default=13, help="How many results to return")
    p_query.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_query.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")
    p_query.add_argument("--andentity", help="Comma-separated entities (all required). Format: [LABEL/]TEXT")
    p_query.add_argument("--orentity", help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT")
    p_query.add_argument("--showentity", nargs='?', const='', help="Display entities. Provide comma-separated list ([LABEL/]TEXT or LABEL/ for all of type) or use flag alone to show all entities")

    p_title = sub.add_parser("title", help="Export tab-delimited MongoDB ID and article title")
    p_title.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_title.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")

    p_entity = sub.add_parser("dumpentity", help="Export tab-delimited list of unique entities")
    p_entity.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_entity.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")
    p_entity.add_argument("--showentity", nargs='?', const='', help="Filter entities. Provide comma-separated list ([LABEL/]TEXT or LABEL/ for all of type) or use flag alone to show all entities")

    p_article = sub.add_parser("article", help="Export articles matching entity filters")
    p_article.add_argument("--start-date", help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)")
    p_article.add_argument("--end-date",   help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)")
    p_article.add_argument("--andentity", help="Comma-separated entities (all required). Format: [LABEL/]TEXT")
    p_article.add_argument("--orentity", help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT")
    p_article.add_argument("--showentity", nargs='?', const='', help="Display entities. Provide comma-separated list ([LABEL/]TEXT or LABEL/ for all of type) or use flag alone to show all entities")

    args = parser.parse_args(argv)

    if args.cmd == "load":
        count = load_into_chroma(limit=args.limit,
                                start_date=args.start_date,
                                end_date=args.end_date)
        print(f"âœ…  Stored {count} new vectors in Chroma")
        return

    if args.cmd == "query":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []
        
        # Handle --showentity flag
        if args.showentity is not None:
            # Flag was provided
            if args.showentity == '':
                # No value provided, show all entities
                show_entities = []
            else:
                # Value provided, parse it
                show_entities = parse_entity_list(args.showentity)
        else:
            # Flag not provided, don't show entities
            show_entities = None

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
            print(f"Title: {h['title']}")
            print(f"Published: {h['published']}")
            print(f"Source: {h['source']}")
            if 'entities' in h and h['entities']:
                print(h['entities'])
            print(f"Text: {h['text']}")
        return

    if args.cmd == "title":
        export_titles(start_date=args.start_date,
                     end_date=args.end_date)
        return

    if args.cmd == "dumpentity":
        # Handle --showentity flag
        if args.showentity is not None:
            # Flag was provided
            if args.showentity == '':
                # No value provided, show all entities
                show_entities = []
            else:
                # Value provided, parse it
                show_entities = parse_entity_list(args.showentity)
        else:
            # Flag not provided, show all entities by default
            show_entities = []
        
        dump_entities(start_date=args.start_date,
                     end_date=args.end_date,
                     show_entities=show_entities)
        return

    if args.cmd == "article":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []
        
        # Handle --showentity flag
        if args.showentity is not None:
            # Flag was provided
            if args.showentity == '':
                # No value provided, show all entities
                show_entities = []
            else:
                # Value provided, parse it
                show_entities = parse_entity_list(args.showentity)
        else:
            # Flag not provided, don't show entities
            show_entities = None
        
        export_articles(start_date=args.start_date,
                       end_date=args.end_date,
                       and_entities=and_entities,
                       or_entities=or_entities,
                       show_entities=show_entities)
        return


if __name__ == "__main__":
    main()