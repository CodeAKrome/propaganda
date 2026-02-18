#!/usr/bin/env python3
"""
Cluster news articles or documents by title using sentence embeddings.
Reads tab-delimited id<TAB>title from stdin, outputs id, title, and cluster label.
"""

import sys
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import warnings
warnings.filterwarnings('ignore')

def read_input():
    """Read tab-delimited id and title from stdin."""
    ids = []
    titles = []
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)  # Split on first tab only
        if len(parts) == 2:
            ids.append(parts[0])
            titles.append(parts[1])
        else:
            # If no tab, treat whole line as title with auto-generated ID
            ids.append(str(len(ids)))
            titles.append(parts[0])
    
    return ids, titles

def cluster_hdbscan(embeddings, min_cluster_size=5, min_samples=2):
    """
    HDBSCAN clustering - good for noise handling, no fixed cluster count needed.
    Returns -1 for noise/outliers.
    """
    # HDBSCAN works better with cosine distance, so we convert to distance matrix
    # or use metric='cosine' directly if available
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Euclidean works fine on normalized embeddings
        cluster_selection_method='eom'  # Excess of Mass (more conservative)
    )
    
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer.probabilities_

def cluster_agglomerative(embeddings, n_clusters=None, distance_threshold=None):
    """
    Agglomerative clustering - good for hierarchical/nested topics.
    Either specify n_clusters OR distance_threshold (for automatic cluster detection).
    """
    if n_clusters is None and distance_threshold is None:
        # Default: let it decide based on distance
        distance_threshold = 0.5
        n_clusters = None
    elif n_clusters is not None:
        distance_threshold = None
    
    # Cosine similarity approach: convert to distance (1 - similarity)
    # Or use metric='cosine' directly (sklearn >= 0.24)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average',
        compute_distances=True
    )
    
    labels = clustering.fit_predict(embeddings)
    return labels, clustering

def main():
    parser = argparse.ArgumentParser(
        description='Cluster documents by title using sentence embeddings'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['hdbscan', 'agglomerative'],
        default='hdbscan',
        help='Clustering method (default: hdbscan)'
    )
    parser.add_argument(
        '--model', '-M',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--min-cluster-size', '-c',
        type=int,
        default=3,
        help='HDBSCAN: minimum cluster size (default: 3)'
    )
    parser.add_argument(
        '--n-clusters', '-n',
        type=int,
        default=None,
        help='Agglomerative: exact number of clusters (optional)'
    )
    parser.add_argument(
        '--distance-threshold', '-d',
        type=float,
        default=None,
        help='Agglomerative: distance threshold to cut tree (optional)'
    )
    parser.add_argument(
        '--output-probabilities', '-p',
        action='store_true',
        help='Output cluster probabilities (HDBSCAN only)'
    )
    
    args = parser.parse_args()
    
    # Read input
    try:
        ids, titles = read_input()
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not titles:
        print("No input received. Provide tab-delimited id<TAB>title via stdin.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(titles)} documents", file=sys.stderr)
    
    # Generate embeddings
    print(f"Generating embeddings using {args.model}...", file=sys.stderr)
    model = SentenceTransformer(args.model)
    embeddings = model.encode(titles, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity clustering
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Cluster
    print(f"Clustering using {args.method}...", file=sys.stderr)
    
    if args.method == 'hdbscan':
        labels, probabilities = cluster_hdbscan(
            embeddings, 
            min_cluster_size=args.min_cluster_size
        )
        
        # Output results
        header = "id\ttitle\tcluster"
        if args.output_probabilities:
            header += "\tprobability"
        print(header)
        
        for idx, (id_, title, label, prob) in enumerate(zip(ids, titles, labels, probabilities)):
            line = f"{id_}\t{title}\t{label}"
            if args.output_probabilities:
                line += f"\t{prob:.3f}"
            print(line)
            
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"\nClusters found: {n_clusters}, Noise points: {n_noise}", file=sys.stderr)
        
    else:  # agglomerative
        labels, clusterer = cluster_agglomerative(
            embeddings,
            n_clusters=args.n_clusters,
            distance_threshold=args.distance_threshold
        )
        
        # Output results
        print("id\ttitle\tcluster")
        for id_, title, label in zip(ids, titles, labels):
            print(f"{id_}\t{title}\t{label}")
            
        n_clusters = len(set(labels))
        print(f"\nClusters found: {n_clusters}", file=sys.stderr)
        
        # If hierarchical output is needed, we could export linkage matrix here
        if hasattr(clusterer, 'distances_'):
            print(f"Distance range: {min(clusterer.distances_):.3f} to {max(clusterer.distances_):.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
