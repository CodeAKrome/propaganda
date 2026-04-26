#!/usr/bin/env python3
"""
Embedding generation using sentence transformers.

Optimizations:
- Batch processing for memory efficiency
- Progress reporting
- Error handling
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)


def generate_embeddings(
    titles: List[str], batch_size: Optional[int] = None, show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of article titles.

    Args:
        titles: List of article title strings
        batch_size: Batch size for processing (default from config)
        show_progress: Show progress bar

    Returns:
        Normalized embedding vectors as numpy array of shape (n_titles, embedding_dim)

    Raises:
        RuntimeError: If model loading or embedding generation fails
    """
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)

        logger.info(
            f"Generating embeddings for {len(titles)} titles (batch_size={batch_size})"
        )

        # Process in batches for memory efficiency
        embeddings = model.encode(
            titles,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )

        return np.array(embeddings, dtype=np.float32)

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
    # Use optimized BLAS operations
    similarity_matrix = embeddings @ embeddings.T
    return similarity_matrix
