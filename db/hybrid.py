#!/usr/bin/env python3
"""
hybrid.py
Hybrid search:  MongoDB filter  →  temporary Chroma collection  →  vector search
Uses the same DB/collection credentials as mongo2chroma.py
"""

import os, uuid
import sys
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import pymongo
from bson import ObjectId
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Re-use the exact same configuration section from mongo2chroma.py
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

CHROMA_PATH = "./chroma"  # same persistent path

# HYBRID_COLL = "hybrid_tmp"  # temporary collection – wiped each run
HYBRID_COLL = f"hybrid_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
)

encoder = SentenceTransformer(EMBED_MODEL)


# ----------  re-use helper code from mongo2chroma.py  --------------
def parse_date_arg(date_str: str) -> datetime:
    if date_str.startswith("-") and date_str[1:].isdigit():
        return datetime.now() + timedelta(days=int(date_str))
    return datetime.fromisoformat(date_str)


def parse_entity_spec(spec: str) -> Tuple[Optional[str], str]:
    if "/" in spec:
        label, text = spec.split("/", 1)
        return label, text
    return None, spec


def parse_entity_list(entity_str: Optional[str]) -> List[Tuple[Optional[str], str]]:
    if not entity_str:
        return []
    return [parse_entity_spec(e.strip()) for e in entity_str.split(",")]


def build_mongo_entity_filter(
    and_list: List[Tuple[Optional[str], str]], or_list: List[Tuple[Optional[str], str]]
) -> Dict:
    clauses = []
    for label, text in and_list:
        if label and text:
            clauses.append(
                {"ner.entities": {"$elemMatch": {"label": label, "text": text}}}
            )
        elif label:
            clauses.append({"ner.entities.label": label})
        else:
            clauses.append({"ner.entities.text": text})
    if or_list:
        or_clauses = []
        for label, text in or_list:
            if label and text:
                or_clauses.append(
                    {"ner.entities": {"$elemMatch": {"label": label, "text": text}}}
                )
            elif label:
                or_clauses.append({"ner.entities.label": label})
            else:
                or_clauses.append({"ner.entities.text": text})
        clauses.append({"$or": or_clauses})
    if not clauses:
        return {}
    # ---  always wrap in a dict  ---
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


# --------------  debug helpers  ------------------------------------
def debug(msg: str):
    print(f"{msg}", file=sys.stderr)


# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Hybrid Mongo→Chroma vector search")
    parser.add_argument("text", help="Query string for vector search")
    parser.add_argument(
        "--andentity",
        help="Comma-separated entities (all required). Format: [LABEL/]TEXT",
    )
    parser.add_argument(
        "--orentity",
        help="Comma-separated entities (at least one). Format: [LABEL/]TEXT",
    )
    parser.add_argument("--start-date", help="Start date: ISO or -N")
    parser.add_argument("--end-date", help="End date: ISO or -N")
    parser.add_argument(
        "--fulltext", help="Full text search for MongoDB. Comma-separated for OR."
    )
    parser.add_argument(
        "--showentity",
        nargs="?",
        const="",
        help="Display entities. Provide list or use flag alone for all",
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=10,
        help="How many results to return (default 10)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Use entity search results as the candidate pool for full-text search.",
    )
    args = parser.parse_args(argv)

    and_entities = parse_entity_list(args.andentity)
    or_entities = parse_entity_list(args.orentity)

    # ---  process fulltext argument ---
    fulltext_search_string = None
    if args.fulltext:
        # For MongoDB $text search:
        # To search for an exact phrase, it must be wrapped in escaped quotes.
        # e.g., "\"toxic masculinity\""
        # The user can provide a comma-separated list of phrases for an OR search.
        # e.g., "toxic masculinity, gray rock" -> "\"toxic masculinity\" \"gray rock\""
        phrases = [term.strip() for term in args.fulltext.split(",")]
        # Wrap each phrase in quotes for an exact phrase search.
        fulltext_search_string = " ".join([f'"{phrase}"' for phrase in phrases])

    # 1. Build Mongo filter
    mongo_filter = {
        "article": {"$exists": True, "$ne": None},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }
    if args.start_date or args.end_date:
        dr = {}
        if args.start_date:
            dr["$gte"] = parse_date_arg(args.start_date)
        if args.end_date:
            dr["$lte"] = parse_date_arg(args.end_date)
        mongo_filter["published"] = dr
    entity_filter = build_mongo_entity_filter(and_entities, or_entities)
    if entity_filter:
        mongo_filter.update(entity_filter)

    # 2. Fetch candidates based on filter logic
    if args.filter and fulltext_search_string:
        # --filter: Entity search provides candidates for full-text search (2-stage)
        debug("Filter mode: Using entity search results for full-text search.")
        # Stage 1: Get IDs from entity/date filter
        initial_candidates = list(mongo_coll.find(mongo_filter, {"_id": 1}))
        candidate_ids = [c["_id"] for c in initial_candidates]
        debug(f"Entity filter matched: {len(candidate_ids)} records")

        if not candidate_ids:
            candidates = []
        else:
            # Stage 2: Run full-text search only on those IDs
            fulltext_filter = {
                "_id": {"$in": candidate_ids},
                "$text": {"$search": fulltext_search_string},
            }
            candidates = list(
                mongo_coll.find(
                    fulltext_filter,
                    {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1},
                )
            )
    else:
        # Default behavior (no --filter): Union of entity and full-text results
        debug("Default mode: Combining entity and full-text search results.")
        # Start with the entity/date filtered list
        candidates = list(
            mongo_coll.find(
                mongo_filter,
                {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1},
            ).sort("published", -1)
        )
        debug(f"Mongo filter matched: {len(candidates)} records")

        # If full-text search is specified, run it and combine results
        if fulltext_search_string:
            text_filter = {"$text": {"$search": fulltext_search_string}}
            if "published" in mongo_filter:
                text_filter["published"] = mongo_filter["published"]
            text_candidates = list(
                mongo_coll.find(
                    text_filter,
                    {"_id": 1, "title": 1, "source": 1, "published": 1, "ner": 1, "article": 1},
                )
            )
            debug(f"Full-text search matched: {len(text_candidates)} records")
            # Combine and de-duplicate
            candidate_dict = {str(c["_id"]): c for c in candidates}
            for c in text_candidates:
                candidate_dict[str(c["_id"])] = c
            candidates = list(candidate_dict.values())

    if not candidates:
        debug("No candidates found.")
        print("No articles match the filter.")
        return

    candidate_ids = [str(c["_id"]) for c in candidates]
    debug(f"Total unique candidates for vector search: {len(candidates)}")
    # debug("Mongo _ids: " + ",".join(candidate_ids))

    # 3. Wipe / create temporary Chroma collection
    try:
        chroma_client.delete_collection(HYBRID_COLL)
    except Exception:
        pass
    tmp_coll = chroma_client.create_collection(
        name=HYBRID_COLL, metadata={"hnsw:space": "cosine"}
    )

    # 4. Embed + insert candidates in batches
    batch_size = 4096
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        ids = [str(c["_id"]) for c in batch]
        docs = [c.get("article", "").strip() for c in batch]

        debug(
            f"Processing batch {i//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}..."
        )

        embeddings = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
        tmp_coll.add(documents=docs, embeddings=embeddings, ids=ids)

    # 5. Vector search
    query_emb = (
        encoder.encode(
            f"Represent this sentence for searching relevant passages: {fulltext_search_string if fulltext_search_string else args.text}",
            convert_to_tensor=True,
        )
        .cpu()
        .numpy()
        .tolist()
    )
    res = tmp_coll.query(
        query_embeddings=[query_emb], n_results=args.top, include=["documents"]
    )
    hit_ids = res["ids"][0]

    debug(f"Vector search returned: {len(hit_ids)} hits")
    # debug("Vector _ids: " + ",".join(hit_ids))

    # 6. Build id→doc map and print
    id_to_doc = {str(c["_id"]): c for c in candidates}
    show_entities = (
        parse_entity_list(args.showentity) if args.showentity is not None else None
    )

    for _id in hit_ids:
        doc = id_to_doc.get(_id)
        if not doc:
            continue
        published_iso = ""
        if doc.get("published") and isinstance(doc["published"], datetime):
            published_iso = doc["published"].isoformat()
        print("---")
        print(f"ID: {_id}")
        print(f"Title: {doc.get('title', '')}")
        print(f"Published: {published_iso}")
        print(f"Source: {doc.get('source', '')}")
        if show_entities is not None:
            from mongo2chroma import extract_entities_from_doc, format_entities

            # The logic in extract_entities_from_doc handles Optional[str] for the label,
            # so we can safely ignore the linter warning here.
            print(format_entities(extract_entities_from_doc(doc, show_entities)))  # type: ignore
        print(f"Text: {doc.get('article', '')}")


if __name__ == "__main__":
    main()
