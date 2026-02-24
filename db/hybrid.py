#!/usr/bin/env python3
"""
hybrid.py
Hybrid search using ChromaDB with metadata filtering.
Uses the same ChromaDB collection as mongo2chroma.py with metadata for date and entity filtering.
"""

import os
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

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")  # same persistent path
CHROMA_COLL = "articles"  # use the same collection as mongo2chroma.py

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
    parser = argparse.ArgumentParser(description="Hybrid ChromaDB vector search with metadata filtering")
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

    # 1. Get or create the main ChromaDB collection
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLL)
        debug(f"Using existing ChromaDB collection: {CHROMA_COLL}")
    except Exception:
        debug(f"Collection {CHROMA_COLL} not found. Please run mongo2chroma.py load first.")
        print("No articles found. Run 'python db/mongo2chroma.py load' first.")
        return

    # 2. Build ChromaDB where filter for date range
    where_filter = None
    if args.start_date or args.end_date:
        date_conditions = []
        if args.start_date:
            start_dt = parse_date_arg(args.start_date)
            date_conditions.append({"published": {"$gte": start_dt.isoformat()}})
        if args.end_date:
            end_dt = parse_date_arg(args.end_date)
            date_conditions.append({"published": {"$lte": end_dt.isoformat()}})
        
        if len(date_conditions) == 1:
            where_filter = date_conditions[0]
        else:
            where_filter = {"$and": date_conditions}

    # 3. Handle fulltext search mode (MongoDB text search first, then ChromaDB)
    if fulltext_search_string:
        debug(f"Full-text search mode: {fulltext_search_string}")
        
        # Build MongoDB filter
        mongo_filter = {
            "article": {"$exists": True, "$ne": None},
            "$or": [
                {"fetch_error": {"$exists": False}},
                {"fetch_error": {"$in": [None, ""]}},
            ],
            "$text": {"$search": fulltext_search_string},
        }
        
        if args.start_date or args.end_date:
            dr = {}
            if args.start_date:
                dr["$gte"] = parse_date_arg(args.start_date)
            if args.end_date:
                dr["$lte"] = parse_date_arg(args.end_date)
            mongo_filter["published"] = dr
        
        # Get matching IDs from MongoDB
        mongo_docs = list(mongo_coll.find(mongo_filter, {"_id": 1}).sort("published", -1))
        candidate_ids = [str(d["_id"]) for d in mongo_docs]
        debug(f"MongoDB full-text search matched: {len(candidate_ids)} records")
        
        if not candidate_ids:
            debug("No candidates found.")
            print("No articles match the filter.")
            return
        
        # Get documents from ChromaDB for these IDs
        chroma_res = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        id_to_doc = {_id: doc for _id, doc in zip(chroma_res["ids"], chroma_res["documents"])}
        id_to_meta = {_id: meta for _id, meta in zip(chroma_res["ids"], chroma_res["metadatas"])}
        
        # Filter by entities using metadata
        if and_entities or or_entities:
            filtered_ids = []
            for _id in candidate_ids:
                if _id not in id_to_meta:
                    continue
                meta = id_to_meta[_id]
                entities_str = meta.get("entities", "[]")
                try:
                    entities = json.loads(entities_str)
                except (json.JSONDecodeError, TypeError):
                    entities = []
                
                # Check AND entities - all must be present
                and_match = True
                for label, text_val in and_entities:
                    if text_val not in entities:
                        and_match = False
                        break
                
                # Check OR entities - at least one must be present
                or_match = True
                if or_entities:
                    or_match = False
                    for label, text_val in or_entities:
                        if text_val in entities:
                            or_match = True
                            break
                
                if and_match and or_match:
                    filtered_ids.append(_id)
            
            candidate_ids = filtered_ids
            debug(f"After entity filtering: {len(candidate_ids)} records")
        
        # For fulltext mode, we already have the results, just need to display them
        hit_ids = candidate_ids[:args.top * 10] if args.bm25 else candidate_ids[:args.top]
        hit_docs = [id_to_doc.get(_id, "") for _id in hit_ids]
        
    else:
        # 4. Vector search mode using ChromaDB
        search_text = args.text
        
        # Build query embedding
        query_emb = (
            encoder.encode(
                f"Represent this sentence for searching relevant passages: {search_text}",
                convert_to_tensor=True,
            )
            .cpu()
            .numpy()
            .tolist()
        )
        
        # Query ChromaDB with metadata filter
        k_results = args.top * 10 if args.bm25 else args.top
        res = collection.query(
            query_embeddings=[query_emb],
            n_results=k_results,
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        hit_ids = res["ids"][0]
        hit_docs = res["documents"][0]
        hit_metas = res["metadatas"][0]
        
        debug(f"ChromaDB vector search returned: {len(hit_ids)} hits")
        
        # Filter by entities using metadata
        if and_entities or or_entities:
            filtered_ids = []
            filtered_docs = []
            for _id, doc, meta in zip(hit_ids, hit_docs, hit_metas):
                entities_str = meta.get("entities", "[]")
                try:
                    entities = json.loads(entities_str)
                except (json.JSONDecodeError, TypeError):
                    entities = []
                
                # Check AND entities - all must be present
                and_match = True
                for label, text_val in and_entities:
                    if text_val not in entities:
                        and_match = False
                        break
                
                # Check OR entities - at least one must be present
                or_match = True
                if or_entities:
                    or_match = False
                    for label, text_val in or_entities:
                        if text_val in entities:
                            or_match = True
                            break
                
                if and_match and or_match:
                    filtered_ids.append(_id)
                    filtered_docs.append(doc)
            
            hit_ids = filtered_ids
            hit_docs = filtered_docs
            debug(f"After entity filtering: {len(hit_ids)} records")

    # --- BM25 Reranking Logic ---
    if args.bm25:
        if not HAS_BM25:
            debug("WARNING: rank_bm25 not installed. Skipping BM25 reranking.")
            hit_ids = hit_ids[: args.top]
        else:
            debug("Applying BM25 reranking...")
            search_text = fulltext_search_string if fulltext_search_string else args.text
            # Tokenize corpus and query (simple whitespace tokenization)
            tokenized_corpus = [doc.lower().split() for doc in hit_docs]
            tokenized_query = search_text.lower().split()

            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)

            # Combine ids and scores, then sort
            scored_results = list(zip(hit_ids, doc_scores))

            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Keep top N
            hit_ids = [item[0] for item in scored_results[: args.top]]
            debug("BM25 reranking complete.")

    # Write IDs to file if requested
    if args.ids:
        with open(args.ids, "a") as f:
            f.write(f"# {' '.join(sys.argv)}\n")
            for mongo_id in hit_ids:
                f.write(f"{mongo_id}\n")
        debug(f"Appended {len(hit_ids)} MongoDB IDs to {args.ids}")

    if not hit_ids:
        debug("No results found.")
        print("No articles match the query.")
        return

    # Fetch additional data from MongoDB for display
    mongo_filter = {
        "_id": {"$in": [ObjectId(i) for i in hit_ids]},
    }
    mongo_docs = list(
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
        )
    )
    id_to_doc = {str(d["_id"]): d for d in mongo_docs}
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

            print(format_entities(extract_entities_from_doc(doc, show_entities)))  # type: ignore
        print(f"Text: {doc.get('article', '')}")


if __name__ == "__main__":
    main()
