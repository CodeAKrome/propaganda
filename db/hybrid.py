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

# Embedding model defaults - can be overridden via CLI
DEFAULT_EMBED_TYPE = "flair-pooled"  # Options: flair-pooled, bge-large, sentence-transformer
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
FLAIR_MODEL = "news-forward"
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
)

# Lazy-loaded encoder
_encoder = None
_embed_type = None


class FlairPooledEncoder:
    """Wrapper for Flair PooledFlairEmbeddings to match sentence-transformers API."""
    
    def __init__(self, flair_embedding):
        self._flair = flair_embedding
    
    def encode(self, texts, convert_to_tensor=True, **kwargs):
        """Encode texts to embeddings. Returns tensor-like object with .cpu().numpy() method."""
        from flair.data import Sentence
        import numpy as np
        
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = []
        for text in texts:
            sent = Sentence(text)
            self._flair.embed(sent)
            # Get the embedding from the first token (pooled representation)
            if len(sent) > 0:
                emb = sent[0].embedding.detach().cpu().numpy()
            else:
                emb = np.zeros(self._flair.embedding_length)
            embeddings.append(emb)
        
        result = np.array(embeddings)
        
        # For single input, squeeze to 1D to match sentence-transformers behavior
        if single_input:
            result = result[0]
        
        # Create a tensor-like wrapper that matches sentence-transformers output
        class TensorWrapper:
            def __init__(self, arr):
                self._arr = arr
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self._arr
        
        return TensorWrapper(result)


def get_encoder(embed_type: str = "flair-pooled", embed_model: str | None = None):
    """
    Lazy-load encoder based on type.
    
    Args:
        embed_type: Type of embedding - 'flair-pooled', 'bge-large', or 'sentence-transformer'
        embed_model: Model name/path (for sentence-transformer type)
    
    Returns:
        Encoder object with .encode() method
    """
    global _encoder, _embed_type
    
    # Determine if we need to (re)load the encoder
    if _encoder is None or _embed_type != embed_type:
        if embed_type == "flair-pooled":
            from flair.embeddings import PooledFlairEmbeddings
            import numpy as np
            
            model_name = embed_model or FLAIR_MODEL
            _encoder = FlairPooledEncoder(PooledFlairEmbeddings(model_name))
        elif embed_type == "bge-large":
            _encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        else:  # sentence-transformer
            model = embed_model or EMBED_MODEL
            _encoder = SentenceTransformer(model)
        
        _embed_type = embed_type
    
    return _encoder


# Default encoder for backwards compatibility
encoder = get_encoder()


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
        Formatted string for display wrapped in <bias> tags
    """
    if not bias:
        return "<bias>\n(none)\n</bias>"
    
    # Handle legacy string format
    if isinstance(bias, str):
        try:
            bias = json.loads(bias)
        except (json.JSONDecodeError, ValueError):
            return f"<bias>\n{bias}\n</bias>"  # Return as-is if not valid JSON
    
    if not isinstance(bias, dict):
        return f"<bias>\n{str(bias)}\n</bias>"
    
    # Format as object
    lines = ["<bias>"]
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
    
    lines.append("</bias>")
    return "\n".join(lines)


# --------------  debug helpers  ------------------------------------
def debug(msg: str):
    print(f"{msg}", file=sys.stderr)


# ------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Hybrid ChromaDB vector search with metadata filtering")
    parser.add_argument("text", nargs="?", default="", help="Query string for vector search")
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
    parser.add_argument(
        "--flair-pooled",
        action="store_true",
        help="Use Flair PooledFlairEmbeddings (news-forward) as embedding model (default)",
    )
    parser.add_argument(
        "--bge-large",
        action="store_true",
        help="Use BAAI/bge-large-en-v1.5 sentence transformer as embedding model",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Use a custom sentence transformer model as embedding model",
    )
    args = parser.parse_args(argv)

    # Determine embedding type and model from CLI args
    if hasattr(args, 'embedding') and args.embedding:
        embed_type, embed_model = "sentence-transformer", args.embedding
    elif hasattr(args, 'bge_large') and args.bge_large:
        embed_type, embed_model = "bge-large", None
    else:
        # Default to flair-pooled
        embed_type, embed_model = "flair-pooled", None
    
    # Get the appropriate encoder
    encoder = get_encoder(embed_type, embed_model)

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
    # Note: ChromaDB only supports numeric comparisons for $gte/$lte, not strings
    # Date filtering will be done post-query via MongoDB
    where_filter = None

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
        
        # Special case: If no text query but entity filtering is requested,
        # query MongoDB directly for entities instead of doing meaningless vector search
        if not search_text.strip() and (and_entities or or_entities):
            debug("Entity-only search mode: querying MongoDB for entities")
            
            # Build MongoDB filter for entities
            mongo_filter = {
                "article": {"$exists": True, "$ne": None},
                "$or": [
                    {"fetch_error": {"$exists": False}},
                    {"fetch_error": {"$in": [None, ""]}},
                ],
            }
            
            # Add date filter if specified
            if args.start_date or args.end_date:
                dr = {}
                if args.start_date:
                    dr["$gte"] = parse_date_arg(args.start_date)
                if args.end_date:
                    dr["$lte"] = parse_date_arg(args.end_date)
                mongo_filter["published"] = dr
            
            # Add entity filters - query MongoDB's ner.entities.text field
            entity_conditions = []
            
            # For OR entities: at least one must match
            if or_entities:
                for label, text_val in or_entities:
                    entity_conditions.append({"ner.entities.text": text_val})
                mongo_filter["$or"] = entity_conditions
            
            # For AND entities: all must match (use $and)
            if and_entities:
                and_conditions = []
                for label, text_val in and_entities:
                    and_conditions.append({"ner.entities.text": text_val})
                if "$or" in mongo_filter:
                    # Combine: (OR conditions) AND (AND conditions)
                    mongo_filter = {
                        "$and": [
                            {"article": {"$exists": True, "$ne": None}},
                            {"$or": [{"fetch_error": {"$exists": False}}, {"fetch_error": {"$in": [None, ""]}}]},
                            {"$or": entity_conditions} if or_entities else {},
                            *and_conditions
                        ]
                    }
                    # Re-add date filter if present
                    if args.start_date or args.end_date:
                        dr = {}
                        if args.start_date:
                            dr["$gte"] = parse_date_arg(args.start_date)
                        if args.end_date:
                            dr["$lte"] = parse_date_arg(args.end_date)
                        mongo_filter["$and"].append({"published": dr})
                else:
                    # Only AND entities, no OR
                    for cond in and_conditions:
                        mongo_filter.update(cond)
            
            # Get matching IDs from MongoDB
            mongo_docs = list(mongo_coll.find(mongo_filter, {"_id": 1}).sort("published", -1).limit(args.top * 10))
            candidate_ids = [str(d["_id"]) for d in mongo_docs]
            debug(f"MongoDB entity search matched: {len(candidate_ids)} records")
            
            if not candidate_ids:
                debug("No candidates found.")
                print("No articles match the filter.")
                return
            
            # Get documents from ChromaDB for these IDs
            chroma_res = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
            id_to_doc = {_id: doc for _id, doc in zip(chroma_res["ids"], chroma_res["documents"])}
            id_to_meta = {_id: meta for _id, meta in zip(chroma_res["ids"], chroma_res["metadatas"])}
            
            hit_ids = candidate_ids[:args.top]
            hit_docs = [id_to_doc.get(_id, "") for _id in hit_ids]
            hit_metas = [id_to_meta.get(_id, {}) for _id in hit_ids]
            
        else:
            # Standard vector search mode
            # Step 1: If date filtering is needed, first get matching IDs from MongoDB
            date_filtered_ids = None
            if args.start_date or args.end_date:
                mongo_date_filter = {
                    "article": {"$exists": True, "$ne": None},
                    "$or": [
                        {"fetch_error": {"$exists": False}},
                        {"fetch_error": {"$in": [None, ""]}},
                    ],
                }
                date_filter = {}
                if args.start_date:
                    date_filter["$gte"] = parse_date_arg(args.start_date)
                if args.end_date:
                    date_filter["$lte"] = parse_date_arg(args.end_date)
                mongo_date_filter["published"] = date_filter
                
                # Get IDs from MongoDB that match date range
                date_filtered_ids = set(str(d["_id"]) for d in mongo_coll.find(mongo_date_filter, {"_id": 1}))
                debug(f"MongoDB date filter matched: {len(date_filtered_ids)} records")
            
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
            
            # Query ChromaDB
            # Fetch more results to account for date filtering
            k_results = args.top * 20 if date_filtered_ids is not None else (args.top * 10 if args.bm25 else args.top)
            res = collection.query(
                query_embeddings=[query_emb],
                n_results=k_results,
                include=["documents", "metadatas"]
            )
            
            hit_ids = res["ids"][0]
            hit_docs = res["documents"][0]
            hit_metas = res["metadatas"][0]
            
            debug(f"ChromaDB vector search returned: {len(hit_ids)} hits")
            
            # Step 2: Filter by date if we have a date filter
            if date_filtered_ids is not None:
                filtered_ids = []
                filtered_docs = []
                filtered_metas = []
                for _id, doc, meta in zip(hit_ids, hit_docs, hit_metas):
                    if _id in date_filtered_ids:
                        filtered_ids.append(_id)
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                hit_ids = filtered_ids
                hit_docs = filtered_docs
                hit_metas = filtered_metas
                debug(f"After date filtering: {len(hit_ids)} records")
            
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
        elif not hit_docs:
            debug("No documents for BM25 reranking.")
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

    if not hit_ids:
        debug("No results found.")
        print("No articles match the query.")
        return

    # Write IDs to file if requested
    if args.ids:
        with open(args.ids, "a") as f:
            f.write(f"# {' '.join(sys.argv)}\n")
            for mongo_id in hit_ids:
                f.write(f"{mongo_id}\n")
        debug(f"Appended {len(hit_ids)} MongoDB IDs to {args.ids}")

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
        print(f"{format_bias(doc.get('bias', ''))}")
        if show_entities is not None:
            from mongo2chroma import extract_entities_from_doc, format_entities

            print(format_entities(extract_entities_from_doc(doc, show_entities)))  # type: ignore
        print(f"Text: {doc.get('article', '')}")


if __name__ == "__main__":
    main()
