import logging
import numpy as np
import pandas as pd
from typing import Dict
from typing import List
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


def cluster_articles(
    embeddings: np.ndarray,
    method: str = "dbscan",
    min_samples: int = 2,
    eps: float = 0.25
) -> np.ndarray:
    """
    Cluster articles based on embedding similarity.
    
    Args:
        embeddings: Normalized embedding vectors
        method: Clustering method ("dbscan" currently supported)
        min_samples: Minimum samples for DBSCAN core point
        eps: Maximum distance for DBSCAN neighborhood
        
    Returns:
        Array of cluster labels (-1 for outliers/uncategorized)
        
    Raises:
        ValueError: If unsupported clustering method specified
    """
    if method.lower() != "dbscan":
        raise ValueError(f"Unsupported clustering method: {method}")
    
    try:
        logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        
        # DBSCAN expects distance metric, we have cosine similarity
        # Use precomputed distance matrix
        # Distance = 1 - similarity for normalized vectors
        
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters and {n_noise} outliers")
        
        return cluster_labels
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise


def generate_category_names(
    articles: pd.DataFrame,
    cluster_labels: np.ndarray,
    min_cluster_size: int
) -> Dict[int, Dict]:
    """
    Generate human-readable category names for each cluster.
    
    Args:
        articles: DataFrame with article data
        cluster_labels: Array of cluster assignments
        min_cluster_size: Minimum articles to form valid category
        
    Returns:
        Dictionary mapping cluster_id to category information
    """
    categories = {}
    
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Remove outlier label
    
    for cluster_id in unique_labels:
        # Get articles in this cluster
        mask = cluster_labels == cluster_id
        cluster_articles = articles[mask]
        
        if len(cluster_articles) < min_cluster_size:
            continue
        
        # Generate category name using keyword extraction
        titles = cluster_articles['title'].tolist()
        category_name = extract_category_name(titles)
        
        categories[cluster_id] = {
            'category_name': category_name,
            'article_indices': np.where(mask)[0].tolist()
        }
    
    return categories


def extract_category_name(titles: List[str]) -> str:
    """
    Extract a descriptive category name from article titles.
    
    Args:
        titles: List of article titles in the cluster
        
    Returns:
        Generated category name
    """
    try:
        # Use TF-IDF to find important keywords
        vectorizer = TfidfVectorizer(
            max_features=5,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(titles)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-3:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        # Create category name
        category_name = " / ".join([
            word.title() for word in top_keywords[:2]
        ])
        
        return category_name if category_name else "General"
        
    except:
        return "General"


def format_results(
    categories: Dict[int, Dict],
    articles: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict:
    """
    Format clustering results for output.
    
    Args:
        categories: Dictionary of category information
        articles: DataFrame with article data
        cluster_labels: Array of cluster assignments
        
    Returns:
        Formatted results dictionary
    """
    results = {
        'categories': {},
        'uncategorized': []
    }
    
    # Add categorized articles
    for cat_id, cat_info in categories.items():
        article_list = []
        for idx in cat_info['article_indices']:
            article_list.append({
                'article_id': str(articles.iloc[idx]['article_id']),
                'title': articles.iloc[idx]['title']
            })
        
        # Convert cat_id to int() to ensure it's a Python int, not numpy.int64
        display_id = int(cat_id) + 1  # 1-indexed for display
        
        results['categories'][display_id] = {
            'category_id': display_id,
            'category_name': cat_info['category_name'],
            'article_count': len(article_list),
            'articles': article_list
        }
    
    # Add uncategorized articles
    uncategorized_mask = cluster_labels == -1
    uncategorized_articles = articles[uncategorized_mask]
    
    for _, row in uncategorized_articles.iterrows():
        results['uncategorized'].append({
            'article_id': str(row['article_id']),
            'title': row['title']
        })
    
    return results
    
def format_results000(
    categories: Dict[int, Dict],
    articles: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict:
    """
    Format clustering results for output.
    
    Args:
        categories: Dictionary of category information
        articles: DataFrame with article data
        cluster_labels: Array of cluster assignments
        
    Returns:
        Formatted results dictionary
    """
    results = {
        'categories': {},
        'uncategorized': []
    }
    
    # Add categorized articles
    for cat_id, cat_info in categories.items():
        article_list = []
        for idx in cat_info['article_indices']:
            article_list.append({
                'article_id': str(articles.iloc[idx]['article_id']),
                'title': articles.iloc[idx]['title']
            })
        
        results['categories'][cat_id + 1] = {  # 1-indexed for display
            'category_id': cat_id + 1,
            'category_name': cat_info['category_name'],
            'article_count': len(article_list),
            'articles': article_list
        }
    
    # Add uncategorized articles
    uncategorized_mask = cluster_labels == -1
    uncategorized_articles = articles[uncategorized_mask]
    
    for _, row in uncategorized_articles.iterrows():
        results['uncategorized'].append({
            'article_id': str(row['article_id']),
            'title': row['title']
        })
    
    return results
