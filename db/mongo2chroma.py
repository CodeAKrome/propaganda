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
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import pymongo
from bson import ObjectId

# ------------------------------------------------------------------
# Config — change if necessary
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

CHROMA_PATH = "./chroma"
CHROMA_COLL = "articles"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

# Lazy-loaded ChromaDB and encoder
_chroma_client = None
_collection = None
_encoder = None


def get_chroma_collection():
    """Lazy-load ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLL, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_encoder():
    """Lazy-load sentence transformer encoder."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(EMBED_MODEL)
    return _encoder


# ------------------------------------------------------------------
# Helper functions
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
    """
    Parse entity specification. Returns (label, text) tuple.
    If no slash, label is None and text is the full spec.
    """
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


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def load_into_chroma(
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    and_entities: Optional[List[Tuple[str | None, str]]] = None,
    or_entities: Optional[List[Tuple[str | None, str]]] = None,
    force: bool = False,
) -> int:
    """Embed every article and add into Chroma."""
    from tqdm import tqdm
    
    collection = get_chroma_collection()
    encoder = get_encoder()
    
    if force:
        print("Clearing existing Chroma collection...")
        try:
            _chroma_client.delete_collection(name=CHROMA_COLL)
            collection = _chroma_client.get_or_create_collection(
                name=CHROMA_COLL, metadata={"hnsw:space": "cosine"}
            )
            # Update the global reference
            global _collection
            _collection = collection
        except Exception as e:
            print(f"Note: Could not clear collection: {e}")
    
    and_entities = and_entities or []
    or_entities = or_entities or []

    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        q.update(entity_filter)

    print("Counting documents to load...")
    total_docs = mongo_coll.count_documents(q)
    if limit:
        total_docs = min(total_docs, limit)

    cursor = (
        mongo_coll.find(q, {"_id": 1, "article": 1}).sort("published", -1).limit(limit)
        if limit
        else mongo_coll.find(q, {"_id": 1, "article": 1}).sort("published", -1)
    )

    docs, ids = [], []
    stored = 0
    skipped = 0

    for doc in tqdm(cursor, total=total_docs, desc="Loading articles"):
        _id = str(doc["_id"])
        text = doc.get("article", "").strip()
        if not text:
            continue

        # Skip if already exists (unless force flag is used)
        if not force and collection.get(ids=[_id])["ids"]:
            skipped += 1
            continue

        docs.append(text)
        ids.append(_id)

        if len(docs) >= BATCH_SIZE:
            embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
            collection.add(documents=docs, embeddings=embs, ids=ids)
            stored += len(ids)
            docs, ids = [], []

    if docs:
        embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
        collection.add(documents=docs, embeddings=embs, ids=ids)
        stored += len(ids)

    if skipped > 0:
        print(f"Skipped {skipped} documents already in Chroma (use --force to reload)")

    return stored


def query_chroma(
    text: str,
    n: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    and_entities: Optional[List[Tuple[str | None, str]]] = None,
    or_entities: Optional[List[Tuple[str | None, str]]] = None,
    show_entities: Optional[List[Tuple[str | None, str]]] = None,
) -> List[Dict[str, str]]:
    """Return the `n` most similar articles."""
    collection = get_chroma_collection()
    encoder = get_encoder()
    
    and_entities = and_entities or []
    or_entities = or_entities or []
    show_entities = show_entities or []

    if text.strip():
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        emb = encoder.encode(query_text, convert_to_tensor=True).cpu().numpy().tolist()
        res = collection.query(
            query_embeddings=[emb],
            n_results=n * 10,
            include=["documents"],
        )
        candidate_ids = res["ids"][0]
        candidate_docs = {
            _id: doc for _id, doc in zip(res["ids"][0], res["documents"][0])
        }
    else:
        res = collection.get(limit=10000, include=["documents"])
        candidate_ids = res["ids"]
        candidate_docs = {_id: doc for _id, doc in zip(res["ids"], res["documents"])}

    if not candidate_ids:
        return []

    mongo_filter = {
        "_id": {"$in": [ObjectId(i) for i in candidate_ids]},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        mongo_filter["published"] = date_filter

    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        mongo_filter.update(entity_filter)

    mongo_docs = list(
        mongo_coll.find(
            mongo_filter,
            {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1},
        )
    )

    out = []
    id_to_doc = {str(d["_id"]): d for d in mongo_docs}

    for _id in candidate_ids:
        if _id not in id_to_doc:
            continue

        mdoc = id_to_doc[_id]
        published_iso = ""
        published_dt = mdoc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            published_iso = published_dt.isoformat()

        article_text = candidate_docs.get(_id, mdoc.get("article", ""))

        out.append(
            {
                "id": _id,
                "title": mdoc.get("title", ""),
                "published": published_iso,
                "source": mdoc.get("source", ""),
                "text": article_text,
                "entities": (
                    format_entities(extract_entities_from_doc(mdoc, show_entities))
                    if show_entities is not None
                    else None
                ),
            }
        )

        if len(out) >= n:
            break

    return out


def extract_entities_from_doc(
    doc: Dict, show_entities: List[Tuple[str | None, str]]
) -> Dict[str, Set[str]]:
    """Extract entities from document that match the show_entities filter."""
    result = defaultdict(set)

    if "ner" not in doc or "entities" not in doc["ner"]:
        return result

    for entity in doc["ner"]["entities"]:
        entity_text = entity.get("text", "")
        entity_label = entity.get("label", "")

        if not show_entities:
            result[entity_label].add(entity_text)
        else:
            for show_label, show_text in show_entities:
                if show_label is None:
                    if entity_text == show_text:
                        result[entity_label].add(entity_text)
                elif show_text == "":
                    if entity_label == show_label:
                        result[entity_label].add(entity_text)
                else:
                    if entity_label == show_label and entity_text == show_text:
                        result[entity_label].add(entity_text)

    return result


def format_entities(entities_by_type: Dict[str, Set[str]]) -> str:
    """Format entities for display."""
    if not entities_by_type:
        return "<entities>\n(none)\n</entities>"

    lines = ["<entities>"]
    for label in sorted(entities_by_type.keys()):
        entity_list = sorted(entities_by_type[label])
        lines.append(f"{label}: {', '.join(entity_list)}")
    lines.append("</entities>")

    return "\n".join(lines)


def export_titles(start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
    """Export tab-delimited published date, source, MongoDB ID, and article title."""
    q = {
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ]
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    cursor = mongo_coll.find(
        q, {"_id": 1, "title": 1, "published": 1, "source": 1}
    ).sort("published", -1)

    for doc in cursor:
        _id = str(doc["_id"])
        title = doc.get("title", "")
        source = doc.get("source", "")
        published = doc.get("published")

        if published and isinstance(published, datetime):
            published_str = published.strftime("%Y-%m-%d")
        else:
            published_str = ""

        title = title.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        source = source.replace("\t", " ").replace("\n", " ").replace("\r", " ")

        print(f"{published_str}\t{source}\t{_id}\t{title}")


def dump_entities(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_entities: Optional[List[Tuple[str | None, str]]] = None,
) -> None:
    """Export tab-delimited list of unique entities sorted by count."""
    q = {
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
        "ner.entities": {"$exists": True},
    }

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    cursor = mongo_coll.find(q, {"ner": 1}).sort("published", -1)

    InnerDict = lambda: defaultdict(int)
    entity_counts: Dict[str, Dict[str, int]] = defaultdict(InnerDict)

    for doc in cursor:
        entities_in_doc = extract_entities_from_doc(
            doc, show_entities if show_entities is not None else []
        )

        for entity_type, entity_texts in entities_in_doc.items():
            for entity_text in entity_texts:
                entity_counts[entity_type][entity_text] += 1

    sorted_entities = []
    for entity_type in entity_counts.keys():
        for entity_text, count in entity_counts[entity_type].items():
            sorted_entities.append((count, entity_type, entity_text))

    sorted_entities.sort(key=lambda x: (-x[0], x[1], x[2]))

    for count, entity_type, entity_text in sorted_entities:
        print(f"{count}\t{entity_type}\t{entity_text}")


def export_articles(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    and_entities: Optional[List[Tuple[str | None, str]]] = None,
    or_entities: Optional[List[Tuple[str | None, str]]] = None,
    show_entities: Optional[List[Tuple[str | None, str]]] = None,
    limit: Optional[int] = None,
) -> None:
    """Export articles matching entity filters."""
    id_list = id_list or []
    and_entities = and_entities or []
    or_entities = or_entities or []

    q = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

    if id_list:
        q["_id"] = {"$in": [ObjectId(i) for i in id_list]}

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        q["published"] = date_filter

    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        q.update(entity_filter)

    cursor = mongo_coll.find(
        q, {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1, "bias": 1}
    ).sort("published", -1)

    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        _id = str(doc["_id"])
        title = doc.get("title", "")
        source = doc.get("source", "")
        article_text = doc.get("article", "")

        published_iso = ""
        published_dt = doc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            published_iso = published_dt.isoformat()

        bias = doc.get("bias", "")

        print("---")
        print(f"ID: {_id}")
        print(f"Title: {title}")
        print(f"Published: {published_iso}")
        print(f"Source: {source}")
        print(f"Bias: {bias}")

        if show_entities is not None:
            entities = format_entities(extract_entities_from_doc(doc, show_entities))
            print(entities)

        print(f"Text: {article_text}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MongoDB → Chroma vector loader / querier"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="Embed all articles into Chroma")
    p_load.add_argument("-l", "--limit", type=int, help="Only process N articles")
    p_load.add_argument(
        "--start-date", help="Load articles published on/after this date (ISO or -N)"
    )
    p_load.add_argument(
        "--end-date", help="Load articles published on/before this date (ISO or -N)"
    )
    p_load.add_argument(
        "--andentity",
        help="Comma-separated entities (all required). Format: [LABEL/]TEXT",
    )
    p_load.add_argument(
        "--orentity",
        help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT",
    )
    p_load.add_argument(
        "--force",
        action="store_true",
        help="Clear existing collection and reload all articles",
    )

    p_query = sub.add_parser("query", help="Search articles by text")
    p_query.add_argument(
        "text",
        nargs="?",
        default="",
        help="Query string (optional if using entity filters)",
    )
    p_query.add_argument(
        "-n", "--top", type=int, default=13, help="How many results to return"
    )
    p_query.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)",
    )
    p_query.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)",
    )
    p_query.add_argument(
        "--andentity",
        help="Comma-separated entities (all required). Format: [LABEL/]TEXT",
    )
    p_query.add_argument(
        "--orentity",
        help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT",
    )
    p_query.add_argument(
        "--showentity",
        nargs="?",
        const="",
        help="Display entities. Provide comma-separated list or use flag alone to show all",
    )

    p_title = sub.add_parser(
        "title", help="Export tab-delimited MongoDB ID and article title"
    )
    p_title.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)",
    )
    p_title.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)",
    )

    p_entity = sub.add_parser(
        "dumpentity", help="Export tab-delimited list of unique entities"
    )
    p_entity.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)",
    )
    p_entity.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)",
    )
    p_entity.add_argument(
        "--showentity",
        nargs="?",
        const="",
        help="Filter entities. Provide comma-separated list or use flag alone to show all",
    )

    p_article = sub.add_parser(
        "article", help="Export articles matching entity filters"
    )
    p_article.add_argument(
        "-n",
        "--top",
        type=int,
        default=None,
        help="Maximum number of articles to return",
    )
    p_article.add_argument("--id", help="Comma-separated list of MongoDB _id strings")
    p_article.add_argument(
        "--idfile", help="File containing MongoDB _id strings, one per line"
    )
    p_article.add_argument(
        "--start-date",
        help="Start date: ISO format or negative days (e.g., '-7' for 7 days ago)",
    )
    p_article.add_argument(
        "--end-date",
        help="End date: ISO format or negative days (e.g., '-1' for 1 day ago)",
    )
    p_article.add_argument(
        "--andentity",
        help="Comma-separated entities (all required). Format: [LABEL/]TEXT",
    )
    p_article.add_argument(
        "--orentity",
        help="Comma-separated entities (at least one required). Format: [LABEL/]TEXT",
    )
    p_article.add_argument(
        "--showentity",
        nargs="?",
        const="",
        help="Display entities. Provide comma-separated list or use flag alone to show all",
    )

    args = parser.parse_args(argv)

    if args.cmd == "load":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []

        count = load_into_chroma(
            limit=int(args.limit) if args.limit is not None else None,
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
            and_entities=and_entities,
            or_entities=or_entities,
            force=args.force,
        )
        print(f"✅  Stored {count} new vectors in Chroma")
        return

    if args.cmd == "query":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []

        if args.showentity is not None:
            if args.showentity == "":
                show_entities = []
            else:
                show_entities = parse_entity_list(args.showentity)
        else:
            show_entities = None

        hits = query_chroma(
            args.text,
            n=int(args.top),
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
            and_entities=and_entities,
            or_entities=or_entities,
            show_entities=show_entities,
        )
        for h in hits:
            print("---")
            print(f"ID: {h['id']}")
            print(f"Title: {h['title']}")
            print(f"Published: {h['published']}")
            print(f"Source: {h['source']}")
            if "entities" in h and h["entities"]:
                print(h["entities"])
            print(f"Text: {h['text']}")
        return

    if args.cmd == "title":
        export_titles(
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
        )
        return

    if args.cmd == "dumpentity":
        if args.showentity is not None:
            if args.showentity == "":
                show_entities = []
            else:
                show_entities = parse_entity_list(args.showentity)
        else:
            show_entities = []

        dump_entities(
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
            show_entities=show_entities,
        )
        return

    if args.cmd == "article":
        id_list = []
        if args.id:
            id_list = [i.strip() for i in args.id.split(",")]
        if args.idfile:
            id_list.extend(parse_id_file(args.idfile))
        
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []

        if args.showentity is not None:
            if args.showentity == "":
                show_entities = []
            else:
                show_entities = parse_entity_list(args.showentity)
        else:
            show_entities = None

        export_articles(
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
            id_list=id_list,
            and_entities=and_entities,
            or_entities=or_entities,
            show_entities=show_entities,
            limit=int(args.top) if args.top is not None else None,
        )
        return


if __name__ == "__main__":
    main()