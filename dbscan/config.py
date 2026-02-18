"""
Configuration constants for the news categorization system.
"""

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB settings
COLLECTION_NAME = "news_articles"
DEFAULT_PERSIST_DIR = "./chroma_db"

# Clustering parameters
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_DBSCAN_EPS = 0.25  # Distance threshold (1 - similarity)
DEFAULT_DBSCAN_MIN_SAMPLES = 2

# File paths
DEFAULT_OUTPUT_PATH = "categories.json"
