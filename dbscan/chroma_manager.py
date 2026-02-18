"""
ChromaDB operations for storing and querying article embeddings.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


def initialize_chroma(
    collection_name: str,
    persist_dir: str
) -> chromadb.Collection:
    """
    Initialize ChromaDB client and collection.
    
    Args:
        collection_name: Name of the collection to create/load
        persist_dir: Directory for persistent storage
        
    Returns:
        ChromaDB collection object
        
    Raises:
        RuntimeError: If ChromaDB initialization fails
    """
    try:
        logger.info(f"Creating ChromaDB client at {persist_dir}")
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Delete existing collection if it exists (for fresh start)
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, which is fine
            logger.info(f"No existing collection to delete")
        
        # Create new collection with cosine similarity
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Created collection: {collection_name}")
        return collection
        
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")


def add_articles_to_chroma(
    collection: chromadb.Collection,
    articles: pd.DataFrame,
    embeddings: np.ndarray
) -> None:
    """
    Batch insert articles with embeddings into ChromaDB.
    
    Args:
        collection: ChromaDB collection object
        articles: DataFrame with article_id and title columns
        embeddings: Numpy array of embedding vectors
        
    Raises:
        ValueError: If data dimensions don't match
    """
    if len(articles) != len(embeddings):
        raise ValueError(
            f"Article count ({len(articles)}) doesn't match "
            f"embedding count ({len(embeddings)})"
        )
    
    try:
        # Prepare data for insertion
        ids = [str(aid) for aid in articles['article_id'].tolist()]
        documents = articles['title'].tolist()
        metadatas = [
            {
                "article_id": str(row['article_id']),
                "title": row['title']
            }
            for _, row in articles.iterrows()
        ]
        
        # Batch insert
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(articles)} articles to ChromaDB")
        
    except Exception as e:
        logger.error(f"Failed to add articles to ChromaDB: {e}")
        raise


def query_similar_articles(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    n_results: int = 10,
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Query ChromaDB for similar articles.
    
    Args:
        collection: ChromaDB collection object
        query_embedding: Query embedding vector
        n_results: Maximum number of results to return
        similarity_threshold: Minimum cosine similarity threshold
        
    Returns:
        Dictionary containing query results
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    # Filter by similarity threshold
    filtered_results = {
        'ids': [],
        'distances': [],
        'metadatas': []
    }
    
    for i, distance in enumerate(results['distances'][0]):
        similarity = 1 - distance  # Convert distance to similarity
        if similarity >= similarity_threshold:
            filtered_results['ids'].append(results['ids'][0][i])
            filtered_results['distances'].append(distance)
            filtered_results['metadatas'].append(results['metadatas'][0][i])
    
    return filtered_results