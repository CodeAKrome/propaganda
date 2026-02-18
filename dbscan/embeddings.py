"""
Embedding generation using sentence transformers.
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)


def generate_embeddings(titles: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of article titles.
    
    Args:
        titles: List of article title strings
        
    Returns:
        Normalized embedding vectors as numpy array of shape (n_titles, embedding_dim)
        
    Raises:
        RuntimeError: If model loading or embedding generation fails
    """
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        logger.info(f"Generating embeddings for {len(titles)} titles")
        embeddings = model.encode(
            titles,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return np.array(embeddings)
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Normalized embedding vectors
        
    Returns:
        Similarity matrix of shape (n, n)
    """
    # Since embeddings are normalized, dot product gives cosine similarity
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix
