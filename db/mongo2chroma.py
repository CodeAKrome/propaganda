#!/usr/bin/env python3
"""
mongo2chroma.py
Load cleaned articles from MongoDB into Chroma vector DB
and query them by text similarity.
ChromaDB stores only vectors and IDs. All filtering and data lookup via MongoDB.
"""

import os
import sys
import json
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

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLL = "articles"
BATCH_SIZE = 32

# Embedding model defaults - can be overridden via CLI
DEFAULT_EMBED_TYPE = "flair-pooled"  # Options: flair-pooled, bge-large, sentence-transformer
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
FLAIR_MODEL = "news-forward"
# ------------------------------------------------------------------

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_coll = mongo_client[MONGO_DB][MONGO_COLL]

# Lazy-loaded ChromaDB and encoder
_chroma_client = None
_collection = None
_encoder = None
_embed_type = None  # Track which encoder type is loaded


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


def get_encoder(embed_type: str = "flair-pooled", embed_model: str = None):
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
            from flair.data import Sentence
            import numpy as np
            
            model_name = embed_model or FLAIR_MODEL
            _encoder = FlairPooledEncoder(PooledFlairEmbeddings(model_name))
        elif embed_type == "bge-large":
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        else:  # sentence-transformer
            from sentence_transformers import SentenceTransformer
            model = embed_model or EMBED_MODEL
            _encoder = SentenceTransformer(model)
        
        _embed_type = embed_type
    
    return _encoder


class FlairPooledEncoder:
    """Wrapper for Flair PooledFlairEmbeddings to match sentence-transformers API."""
    
    def __init__(self, flair_embedding):
        self._flair = flair_embedding
    
    def encode(self, texts, convert_to_tensor=True, **kwargs):
        """Encode texts to embeddings. Returns tensor-like object with .cpu().numpy() method."""
        from flair.data import Sentence
        import numpy as np
        import torch
        
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = []
        for text in texts:
            sent = Sentence(text)
            self._flair.embed(sent)
            # Get the embedding from the first token (pooled representation)
            # PooledFlairEmbeddings provides a single embedding per sentence
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
    embed_type: str = "flair-pooled",
    embed_model: Optional[str] = None,
) -> int:
    """Embed every article and add into Chroma."""
    from tqdm import tqdm

    collection = get_chroma_collection()
    encoder = get_encoder(embed_type, embed_model)

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
        mongo_coll.find(q, {"_id": 1, "article": 1, "published": 1, "ner": 1}).sort("published", -1).limit(limit)
        if limit
        else mongo_coll.find(q, {"_id": 1, "article": 1, "published": 1, "ner": 1}).sort("published", -1)
    )

    docs, ids, metadatas = [], [], []
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

        # Build metadata
        metadata = {}
        
        # Add publication date as ISO string for filtering
        published_dt = doc.get("published")
        if published_dt and isinstance(published_dt, datetime):
            metadata["published"] = published_dt.isoformat()
        
        # Add entities as JSON string for filtering
        ner = doc.get("ner")
        if ner and "entities" in ner:
            # Store entity texts as a JSON string for filtering
            entity_texts = [e.get("text", "") for e in ner["entities"] if e.get("text")]
            if entity_texts:
                metadata["entities"] = json.dumps(entity_texts)
        
        docs.append(text)
        ids.append(_id)
        metadatas.append(metadata)

        if len(docs) >= BATCH_SIZE:
            embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
            collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metadatas)
            stored += len(ids)
            docs, ids, metadatas = [], [], []

    if docs:
        embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metadatas)
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
    embed_type: str = "flair-pooled",
    embed_model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return the `n` most similar articles using ChromaDB metadata filtering for dates."""
    collection = get_chroma_collection()
    encoder = get_encoder(embed_type, embed_model)

    and_entities = and_entities or []
    or_entities = or_entities or []
    show_entities = show_entities or []

    # Build ChromaDB where filter for date range
    where_filter = None
    if start_date or end_date:
        date_conditions = []
        if start_date:
            start_dt = parse_date_arg(start_date)
            date_conditions.append({"published": {"$gte": start_dt.isoformat()}})
        if end_date:
            end_dt = parse_date_arg(end_date)
            date_conditions.append({"published": {"$lte": end_dt.isoformat()}})
        
        if len(date_conditions) == 1:
            where_filter = date_conditions[0]
        else:
            where_filter = {"$and": date_conditions}

    if text.strip():
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        emb = encoder.encode(query_text, convert_to_tensor=True).cpu().numpy().tolist()
        res = collection.query(
            query_embeddings=[emb],
            n_results=n * 10,
            where=where_filter,
            include=["documents", "metadatas"],
        )
        candidate_ids = res["ids"][0]
        candidate_docs = {
            _id: doc for _id, doc in zip(res["ids"][0], res["documents"][0])
        }
        candidate_metas = {
            _id: meta for _id, meta in zip(res["ids"][0], res["metadatas"][0])
        }
    else:
        res = collection.get(where=where_filter, limit=10000, include=["documents", "metadatas"])
        candidate_ids = res["ids"]
        candidate_docs = {_id: doc for _id, doc in zip(res["ids"], res["documents"])}
        candidate_metas = {_id: meta for _id, meta in zip(res["ids"], res["metadatas"])}

    if not candidate_ids:
        return []

    # Filter by entities using metadata (entities stored as JSON string)
    if and_entities or or_entities:
        filtered_ids = []
        for _id in candidate_ids:
            meta = candidate_metas.get(_id, {})
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

    # Fetch additional data from MongoDB for display
    mongo_filter = {
        "_id": {"$in": [ObjectId(i) for i in candidate_ids]},
        "$or": [
            {"fetch_error": {"$exists": False}},
            {"fetch_error": {"$in": [None, ""]}},
        ],
    }

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


def export_titles(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> None:
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
        q,
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
        print(f"Bias: {format_bias(bias)}")

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
    p_load.add_argument(
        "--flair-pooled",
        action="store_true",
        help="Use Flair PooledFlairEmbeddings (news-forward) as embedding model (default)",
    )
    p_load.add_argument(
        "--bge-large",
        action="store_true",
        help="Use BAAI/bge-large-en-v1.5 sentence transformer as embedding model",
    )
    p_load.add_argument(
        "--embedding",
        type=str,
        help="Use a custom sentence transformer model as embedding model",
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
    p_query.add_argument(
        "--flair-pooled",
        action="store_true",
        help="Use Flair PooledFlairEmbeddings (news-forward) as embedding model (default)",
    )
    p_query.add_argument(
        "--bge-large",
        action="store_true",
        help="Use BAAI/bge-large-en-v1.5 sentence transformer as embedding model",
    )
    p_query.add_argument(
        "--embedding",
        type=str,
        help="Use a custom sentence transformer model as embedding model",
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

    # Helper to determine embed_type and embed_model from args
    def get_embed_args(args):
        """Determine embedding type and model from CLI args."""
        if hasattr(args, 'embedding') and args.embedding:
            return "sentence-transformer", args.embedding
        elif hasattr(args, 'bge_large') and args.bge_large:
            return "bge-large", None
        else:
            # Default to flair-pooled
            return "flair-pooled", None

    if args.cmd == "load":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []
        embed_type, embed_model = get_embed_args(args)

        count = load_into_chroma(
            limit=int(args.limit) if args.limit is not None else None,
            start_date=str(args.start_date) if args.start_date is not None else None,
            end_date=str(args.end_date) if args.end_date is not None else None,
            and_entities=and_entities,
            or_entities=or_entities,
            force=args.force,
            embed_type=embed_type,
            embed_model=embed_model,
        )
        print(f"✅  Stored {count} new vectors in Chroma")
        return

    if args.cmd == "query":
        and_entities = parse_entity_list(args.andentity) if args.andentity else []
        or_entities = parse_entity_list(args.orentity) if args.orentity else []
        embed_type, embed_model = get_embed_args(args)

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
            embed_type=embed_type,
            embed_model=embed_model,
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
