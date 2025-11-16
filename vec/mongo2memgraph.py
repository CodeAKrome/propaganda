#!/usr/bin/env python3
"""
mongo2memgraph.py  --mode {load|query}
Load:  Mongo → vectors → Memgraph  (uses the "article" field)
Query: natural-language question → vector → top-k Articles  (+ mongo_id)
"""

import argparse
import os
import sys
from itertools import islice
from typing import Iterable, List

import numpy as np
from neo4j import GraphDatabase
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# CONFIG – now explicitly uses the "article" field
# ------------------------------------------------------------------
MONGO_URI    = os.getenv("MONGO_URI",    "mongodb://root:example@localhost:27017/rssnews")
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
BATCH_SIZE   = 512
MODEL_NAME   = "all-MiniLM-L6-v2"
TEXT_FIELD   = "article"          # <--  cleaned article text
VECTOR_DIM   = 384
INDEX_NAME   = "article_vector_idx"
# ------------------------------------------------------------------

mongo = MongoClient(MONGO_URI)
mg    = GraphDatabase.driver(MEMGRAPH_URI)
model = SentenceTransformer(MODEL_NAME)

# ---------- shared helpers ----------
def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def encode(texts: List[str]) -> List[List[float]]:
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype(np.float32).tolist()

# ---------- loading mode ----------
def iter_articles(coll) -> Iterable[dict]:
    for doc in coll.find({TEXT_FIELD: {"$exists": True, "$ne": None}},
                         projection=["title", TEXT_FIELD]):
        yield {
            "mongo_id": str(doc["_id"]),
            "title"   : doc.get("title", "") or "",
            "text"    : doc[TEXT_FIELD][:10_000]
        }

def create_vector_index(session):
    try:
        session.run(f"CALL vec.index.drop('{INDEX_NAME}')")
    except Exception:
        pass
    session.run(f"CALL vec.index.create('{INDEX_NAME}', 'Article', 'vector', {VECTOR_DIM})")

def upload_batch(session, batch: List[dict]):
    session.run(
        """
        UNWIND $batch AS row
        CREATE (a:Article {
            mongo_id: row.mongo_id,
            title   : row.title,
            vector  : row.vector
        })
        """,
        batch=batch
    )

def load_vectors():
    coll = mongo.get_database().articles
    with mg.session() as ses:
        create_vector_index(ses)
        for b in batched(iter_articles(coll), BATCH_SIZE):
            vectors = encode([x["text"] for x in b])
            for item, vec in zip(b, vectors):
                item["vector"] = vec
            upload_batch(ses, b)
            print(f"inserted {len(b)} vectors", file=sys.stderr)
    print("✅  Load complete – vector index ready.", file=sys.stderr)

# ---------- query mode ----------
def query_vectors(question: str, topk: int):
    query_vec = encode([question])[0]
    with mg.session() as ses:
        result = ses.run(
            f"""
            CALL vec.search('{INDEX_NAME}', $vec, $k) YIELD node, score
            RETURN node.mongo_id AS mongo_id,
                   node.title    AS title,
                   score
            ORDER BY score DESC
            """,
            vec=query_vec,
            k=topk
        )
        print(f"Top-{topk} matches for: {question}\n")
        for record in result:
            print(f"{record['score']:.3f}  {record['mongo_id']}  {record['title']}")

# ---------- CLI ----------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mongo → Memgraph vector store / searcher")
    sp = p.add_subparsers(dest="mode", required=True)

    sp.add_parser("load", help="Embed Mongo article text and store vectors in Memgraph")

    q = sp.add_parser("query", help="Natural-language vector search")
    q.add_argument("--question", required=True, help="Question / sentence to search for")
    q.add_argument("--topk", type=int, default=5, help="How many results (default 5)")
    return p

def main():
    args = build_cli().parse_args()
    if args.mode == "load":
        load_vectors()
    else:
        query_vectors(args.question, args.topk)

if __name__ == "__main__":
    main()
