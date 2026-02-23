#!/usr/bin/env python3
"""
hybrid.py
Hybrid search:  MongoDB filter  →  temporary Chroma collection  →  vector search
Uses the same DB/collection credentials as mongo2chroma.py
"""

import os
import uuid
import sys
import json
import argparse
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import pymongo
from bson import ObjectId
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Try to import rank_bm25 for reranking
try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

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


def format_bias(bias: Dict | str | None) -> str:
    """
    Format bias for display. Handles both object and legacy string formats.
    
    Args:
        bias: Bias data - either a dict with dir/deg/reason, or a JSON string
        
    Returns:
        Formatted string for display
    """
    if not bias:
        return "(none)"
    
    # Handle legacy string format
    if isinstance(bias, str):
        try:
            bias = json.loads(bias)
        except (json.JSONDecodeError, ValueError):
            return bias  # Return as-is if not valid JSON
    
    if not isinstance(bias, dict):
        return str(bias)
    
    # Format as object
    lines = []
    if "dir" in bias:
        dir_data = bias["dir"]
        if isinstance(dir_data, dict):
            dir_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(dir_data.items()))
            lines.append(f"Direction: {dir_str}")
    
    if "deg" in bias:
        deg_data = bias["deg"]
        if isinstance(deg_data, dict):
            deg_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(deg_data.items()))
            lines.append(f"Degree: {deg_str}")
    
    if "reason" in bias:
        lines.append(f"Reason: {bias['reason']}")
    
    return "\n".join(lines) if lines else str(bias)


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
    parser.add_argument(
        "--ids",
        help="File to write the MongoDB IDs of matched records to.",
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="Enable BM25 reranking on vector search results.",
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
                    {
                        "_id": 1,
                        "title": 1,
                        "source": 1,
                        "published": 1,
                        "ner": 1,
                        "article": 1,
                        "bias": 1,
                    },
                )
            )
    else:
        # Default behavior (no --filter): Union of entity and full-text results
        debug("Default mode: Combining entity and full-text search results.")
        # Start with the entity/date filtered list
        candidates = list(
            mongo_coll.find(
                mongo_filter,
                {
                    "_id": 1,
                    "title": 1,
                    "source": 1,
                    "published": 1,
                    "ner": 1,
                    "article": 1,
                    "bias": 1,
                },
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
                    {
                        "_id": 1,
                        "title": 1,
                        "source": 1,
                        "published": 1,
                        "ner": 1,
                        "article": 1,
                        "bias": 1,
                    },
                )
            )
            debug(f"Full-text search matched: {len(text_candidates)} records")
            # Combine and de-duplicate
            candidate_dict = {str(c["_id"]): c for c in candidates}
            for c in text_candidates:
                candidate_dict[str(c["_id"])] = c
            candidates = list(candidate_dict.values())

    if args.ids:
        command_line_args = " ".join(sys.argv)
        candidate_ids_to_write = [str(c["_id"]) for c in candidates]
        with open(args.ids, "a") as f:
            f.write(f"# {' '.join(sys.argv)}\n")
            for mongo_id in candidate_ids_to_write:
                f.write(f"{mongo_id}\n")
        debug(f"Appended {len(candidate_ids_to_write)} MongoDB IDs to {args.ids}")

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

    # If reranking, fetch a larger candidate pool (e.g., 10x the requested top N)
    # This gives BM25 enough material to re-order meaningfully.
    k_results = args.top * 10 if args.bm25 else args.top

    # Ensure we don't request more results than we have candidates
    k_results = min(k_results, len(candidates))

    search_text = fulltext_search_string if fulltext_search_string else args.text

    query_emb = (
        encoder.encode(
            f"Represent this sentence for searching relevant passages: {search_text}",
            convert_to_tensor=True,
        )
        .cpu()
        .numpy()
        .tolist()
    )
    res = tmp_coll.query(
        query_embeddings=[query_emb], n_results=k_results, include=["documents"]
    )
    hit_ids = res["ids"][0]
    hit_docs = res["documents"][0]

    debug(f"Vector search returned: {len(hit_ids)} hits")

    # --- BM25 Reranking Logic ---
    if args.bm25:
        if not HAS_BM25:
            debug("WARNING: rank_bm25 not installed. Skipping BM25 reranking.")
            # Fallback to slicing the original list if we fetched extra
            hit_ids = hit_ids[: args.top]
        else:
            debug("Applying BM25 reranking...")
            # Tokenize corpus and query (simple whitespace tokenization)
            tokenized_corpus = [doc.lower().split() for doc in hit_docs]
            tokenized_query = search_text.lower().split()

            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)

            # Combine ids and scores, then sort
            # hit_ids and doc_scores are index-aligned
            scored_results = list(zip(hit_ids, doc_scores))

            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Keep top N
            hit_ids = [item[0] for item in scored_results[: args.top]]
            debug("BM25 reranking complete.")

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
        print(f"Bias: {format_bias(doc.get('bias', ''))}")
        if show_entities is not None:
            from mongo2chroma import extract_entities_from_doc, format_entities

            # The logic in extract_entities_from_doc handles Optional[str] for the label,
            # so we can safely ignore the linter warning here.
            print(format_entities(extract_entities_from_doc(doc, show_entities)))  # type: ignore
        print(f"Text: {doc.get('article', '')}")


if __name__ == "__main__":
    main()
