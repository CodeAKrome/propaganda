#!/usr/bin/env python
"""
Semantic search, pass things that match query.
"""
import sys
import argparse
import torch
from sentence_transformers import SentenceTransformer, util

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Semantic Text Filter')
    parser.add_argument('query', type=str, help='The search query string')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum similarity score (0-1)')
    parser.add_argument('--limit', type=int, default=None, help='Max number of results to output')
    args = parser.parse_args()

    # 2. Load the SBERT model
    # "all-mpnet-base-v2" is excellent for general semantic search
    try:
        model = SentenceTransformer("all-mpnet-base-v2")
    except Exception as e:
        sys.stderr.write(f"Error loading model: {e}\n")
        sys.exit(1)

    # 3. Read input from Stdin
    # Assumes input is line-separated text (e.g., news headlines)
    lines = [line.strip() for line in sys.stdin if line.strip()]
    
    if not lines:
        sys.stderr.write("No input data provided on stdin.\n")
        return

    # 4. Compute Embeddings
    # We encode both the query and the corpus (lines)
    query_embedding = model.encode(args.query, convert_to_tensor=True)
    corpus_embeddings = model.encode(lines, convert_to_tensor=True)

    # 5. Calculate Cosine Similarities
    # util.cos_sim returns a matrix, we take [0] for the single query
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 6. Pair lines with their scores
    results = []
    for i in range(len(lines)):
        score = cos_scores[i].item()
        if score >= args.threshold:
            results.append((lines[i], score))

    # 7. Sort by score (Descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # 8. Output Results
    count = 0
    for line, score in results:
        if args.limit and count >= args.limit:
            break
        # Print score and text, tab-separated
        print(f"{score:.4f}\t{line}")
        count += 1

if __name__ == "__main__":
    main()

