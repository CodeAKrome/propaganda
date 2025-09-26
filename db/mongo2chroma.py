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
# Public API
# ------------------------------------------------------------------
def load_into_chroma(limit: int = None) -> int:
    """
    Embed every article that has `article` text and upsert into Chroma.
    Already-existing IDs are overwritten (idempotent).
    Returns number of vectors stored.
    """
    q = {"article": {"$exists": True, "$ne": None}}
    if limit:
        cursor = mongo_coll.find(q, {"_id": 1, "article": 1}).limit(limit)
    else:
        cursor = mongo_coll.find(q, {"_id": 1, "article": 1})

    docs, ids = [], []

    for doc in tqdm(cursor, desc="Loading articles"):
        _id  = str(doc["_id"])
        text = doc["article"]
        if not text.strip():
            continue

        docs.append(text)
        ids.append(_id)

        # Batch-encode for speed
        if len(docs) >= BATCH_SIZE:
            embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
            collection.upsert(
                documents=docs,
                embeddings=embs,
                ids=ids
            )
            docs, ids = [], []

    # Final leftovers
    if docs:
        embs = encoder.encode(docs, convert_to_tensor=True).cpu().numpy().tolist()
        collection.upsert(documents=docs, embeddings=embs, ids=ids)

    return collection.count()


def query_chroma(text: str, n: int = 5) -> List[Dict[str, str]]:
    """
    Return the `n` most similar articles as:
        [{"id": <mongo _id>, "text": <article body>}, ...]
    """
    emb = encoder.encode(text, convert_to_tensor=True).cpu().numpy().tolist()
    res = collection.query(
        query_embeddings=[emb],
        n_results=n,
        include=["documents", "distances"]
    )

    # Build a list of {id, text} dicts
    return [
        {"id": _id, "text": doc}
        for _id, doc in zip(res["ids"][0], res["documents"][0])
    ]


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
    p_query.add_argument("-n", "--top", type=int, default=10, help="How many results to return")

    args = parser.parse_args(argv)

    if args.cmd == "load":
        count = load_into_chroma(limit=args.limit)
        print(f"✅  Stored {count} vectors in Chroma")
        return

    if args.cmd == "query":
        hits = query_chroma(args.text, n=args.top)
        for h in hits:
            print("---")
            print(f"ID: {h['id']}")
            print(f"Text: {h['text']}")
        return


if __name__ == "__main__":
    main()
