#!/usr/bin/env python3
"""
News Article Semantic Clustering using ChromaDB Vector Search

This script groups news articles into semantic categories by:
1. Embedding article titles with Sentence-Transformers (all-MiniLM-L6-v2)
2. Storing embeddings in ChromaDB with cosine distance
3. Performing similarity-based connected-components clustering via Chroma's ANN search
4. Using Union-Find to merge similar articles (cosine similarity ≥ threshold)

Usage:
    python group_news_chroma.py input.tsv [--threshold 0.78] [--n_neighbors 200] 
                                           [--min_size 2] [--output output.json]

Input format: Tab-delimited file with no header
    Column 1: Article ID (string)
    Column 2: Article title (string)

Output: JSON with clusters sorted by size, or pretty-printed console output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union by rank.
    Used to efficiently merge articles into connected components.
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if merged, False if already in same set."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def get_clusters(self) -> Dict[int, List[int]]:
        """Group indices by their root parent."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


def read_articles(filepath: str) -> List[Tuple[str, str]]:
    """
    Read tab-delimited file with article ID and title.
    
    Args:
        filepath: Path to TSV file (no header, col1=ID, col2=title)
    
    Returns:
        List of (article_id, title) tuples
    """
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                articles.append((parts[0], parts[1]))
    return articles


def embed_articles(articles: List[Tuple[str, str]], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed article titles using Sentence-Transformers.
    
    Args:
        articles: List of (article_id, title) tuples
        model_name: SentenceTransformer model name
    
    Returns:
        NumPy array of shape (n_articles, embedding_dim)
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    titles = [title for _, title in articles]
    print(f"Embedding {len(titles)} article titles...")
    embeddings = model.encode(titles, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def create_chroma_collection(n_articles: int) -> Tuple[chromadb.Client, chromadb.Collection]:
    """
    Create ChromaDB client and collection with cosine distance.
    Uses in-memory for ≤50k articles, persistent otherwise.
    
    Args:
        n_articles: Number of articles to store
    
    Returns:
        Tuple of (client, collection)
    """
    if n_articles <= 50000:
        print("Using in-memory ChromaDB client")
        client = chromadb.Client()
    else:
        print("Using persistent ChromaDB at ./chroma_db")
        client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete collection if exists
    try:
        client.delete_collection("news_articles")
    except:
        pass
    
    # Create with cosine distance
    collection = client.create_collection(
        name="news_articles",
        metadata={"hnsw:space": "cosine"}
    )
    
    return client, collection


def cluster_with_chroma(
    articles: List[Tuple[str, str]],
    embeddings: np.ndarray,
    threshold: float = 0.78,
    n_neighbors: int = 200,
    min_size: int = 2
) -> Tuple[List[List[int]], List[int]]:
    """
    Perform similarity-based clustering using ChromaDB ANN search.
    
    Args:
        articles: List of (article_id, title) tuples
        embeddings: NumPy array of embeddings
        threshold: Cosine similarity threshold (default 0.78)
        n_neighbors: Number of neighbors to retrieve per query (default 200)
        min_size: Minimum cluster size to keep (default 2)
    
    Returns:
        Tuple of (clusters, noise_indices) where clusters is a list of index lists
    """
    n_articles = len(articles)
    
    # Create ChromaDB collection
    client, collection = create_chroma_collection(n_articles)
    
    # Add embeddings to Chroma (provide embeddings explicitly)
    print("Adding embeddings to ChromaDB...")
    ids = [str(i) for i in range(n_articles)]
    documents = [title for _, title in articles]
    
    # Batch add for efficiency
    batch_size = 5000
    for i in range(0, n_articles, batch_size):
        end_idx = min(i + batch_size, n_articles)
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            documents=documents[i:end_idx]
        )
    
    # Initialize Union-Find
    uf = UnionFind(n_articles)
    
    # Query each article and union similar ones
    print(f"Querying ChromaDB for similar articles (threshold={threshold}, n_neighbors={n_neighbors})...")
    distance_threshold = 1.0 - threshold  # Convert cosine similarity to distance
    
    for i in tqdm(range(n_articles)):
        # Query for nearest neighbors
        results = collection.query(
            query_embeddings=[embeddings[i].tolist()],
            n_results=min(n_neighbors, n_articles)
        )
        
        # Process results
        result_ids = results['ids'][0]
        result_distances = results['distances'][0]
        
        for result_id, distance in zip(result_ids, result_distances):
            j = int(result_id)
            
            # Skip self-match
            if i == j:
                continue
            
            # Union if similar enough (distance ≤ threshold means similarity ≥ threshold)
            if distance <= distance_threshold:
                uf.union(i, j)
    
    # Optional: Exact refinement for smaller datasets
    if n_articles <= 8000:
        print("Performing exact refinement pass for small dataset...")
        clusters_dict = uf.get_clusters()
        
        # Only refine clusters with size > 1
        for root, indices in clusters_dict.items():
            if len(indices) <= 1:
                continue
            
            # Compute exact pairwise similarities within cluster
            cluster_embeddings = embeddings[indices]
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)
            
            # Split if any pair falls below threshold
            # (Re-run union-find on this subgraph)
            sub_uf = UnionFind(len(indices))
            for idx_i in range(len(indices)):
                for idx_j in range(idx_i + 1, len(indices)):
                    if similarities[idx_i, idx_j] >= threshold:
                        sub_uf.union(idx_i, idx_j)
            
            # Update main union-find if splits detected
            sub_clusters = sub_uf.get_clusters()
            if len(sub_clusters) > 1:
                # Reset these nodes in main UF and re-union based on exact similarities
                for sub_indices in sub_clusters.values():
                    if len(sub_indices) > 1:
                        global_indices = [indices[i] for i in sub_indices]
                        for k in range(1, len(global_indices)):
                            uf.union(global_indices[0], global_indices[k])
    
    # Extract final clusters
    clusters_dict = uf.get_clusters()
    
    # Separate into main clusters and noise
    clusters = []
    noise_indices = []
    
    for indices in clusters_dict.values():
        if len(indices) >= min_size:
            clusters.append(sorted(indices))
        else:
            noise_indices.extend(indices)
    
    # Sort clusters by size (descending)
    clusters.sort(key=lambda x: len(x), reverse=True)
    
    return clusters, noise_indices


def format_output(
    articles: List[Tuple[str, str]],
    clusters: List[List[int]],
    noise_indices: List[int],
    threshold: float,
    n_neighbors: int,
    min_size: int,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Format clustering results as JSON structure.
    
    Args:
        articles: Original articles list
        clusters: List of cluster index lists
        noise_indices: List of noise/singleton indices
        threshold: Cosine similarity threshold used
        n_neighbors: Number of neighbors queried
        min_size: Minimum cluster size
        model_name: Embedding model name
    
    Returns:
        Dictionary with parameters, clusters, and noise
    """
    output = {
        "parameters": {
            "threshold": threshold,
            "n_neighbors": n_neighbors,
            "min_size": min_size,
            "total_articles": len(articles),
            "embedding_model": model_name
        },
        "clusters": [],
        "noise": []
    }
    
    # Add clusters
    for cluster_id, indices in enumerate(clusters):
        cluster_articles = [
            {"id": articles[i][0], "title": articles[i][1]}
            for i in indices
        ]
        output["clusters"].append({
            "cluster_id": cluster_id,
            "size": len(indices),
            "articles": cluster_articles
        })
    
    # Add noise
    for i in sorted(noise_indices):
        output["noise"].append({
            "id": articles[i][0],
            "title": articles[i][1]
        })
    
    return output


def print_summary(output: Dict[str, Any]):
    """Print a human-readable summary to console."""
    params = output["parameters"]
    clusters = output["clusters"]
    noise = output["noise"]
    
    print("\n" + "="*80)
    print("CLUSTERING RESULTS")
    print("="*80)
    print(f"Total articles: {params['total_articles']}")
    print(f"Embedding model: {params['embedding_model']}")
    print(f"Threshold (cosine similarity): {params['threshold']}")
    print(f"Neighbors queried: {params['n_neighbors']}")
    print(f"Min cluster size: {params['min_size']}")
    print(f"\nFound {len(clusters)} clusters")
    print(f"Noise/singletons: {len(noise)} articles")
    print("="*80)
    
    # Show top 10 largest clusters
    print(f"\nTop {min(10, len(clusters))} largest clusters:\n")
    for cluster in clusters[:10]:
        print(f"Cluster {cluster['cluster_id']} ({cluster['size']} articles):")
        for article in cluster['articles'][:5]:
            title = article['title']
            if len(title) > 100:
                title = title[:97] + "..."
            print(f"  • {article['id']}: {title}")
        if len(cluster['articles']) > 5:
            print(f"  ... and {len(cluster['articles']) - 5} more")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster news articles using ChromaDB vector search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input_file",
        help="Path to tab-delimited file (ID<tab>Title, no header)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold (default: 0.78)"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=200,
        help="Number of neighbors to retrieve per query (default: 200)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=2,
        help="Minimum cluster size to keep (default: 2)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (if not specified, prints to console)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Read articles
    print(f"Reading articles from {args.input_file}...")
    articles = read_articles(args.input_file)
    
    if not articles:
        print("Error: No articles found in input file", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(articles)} articles")
    
    # Embed articles
    embeddings = embed_articles(articles)
    
    # Cluster articles
    clusters, noise_indices = cluster_with_chroma(
        articles,
        embeddings,
        threshold=args.threshold,
        n_neighbors=args.n_neighbors,
        min_size=args.min_size
    )
    
    # Format output
    output = format_output(
        articles,
        clusters,
        noise_indices,
        args.threshold,
        args.n_neighbors,
        args.min_size
    )
    
    # Write or print results
    if args.output:
        output_path = Path(args.output)
        print(f"\nWriting results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
        print_summary(output)
    else:
        print_summary(output)


if __name__ == "__main__":
    main()
