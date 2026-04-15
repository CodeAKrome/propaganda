# DBSCAN Clustering — News Article Categorization

Cluster and categorize news articles using DBSCAN and semantic embeddings.

## Components

| File | Description |
|------|-------------|
| `dbscan/main.py` | Main CLI entry point |
| `dbscan/embeddings.py` | Embedding generation |
| `dbscan/chroma_manager.py` | ChromaDB management |
| `dbscan/categorizer.py` | Clustering logic |
| `dbscan/file_handler.py` | File I/O |

---

## main.py

CLI entry point for article categorization.

### Usage

```bash
python dbscan/main.py [options]
```

### Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Input TSV file | Required |
| `--output FILE` | Output JSON file | `categorized.json` |
| `--min-size N` | Min cluster size | 5 |
| `--threshold N` | Similarity threshold | 0.85 |
| `--persist-dir DIR` | ChromaDB directory | `./chroma_dbscan` |

### Examples

```bash
python dbscan/main.py articles.tsv
python dbscan/main.py articles.tsv --min-size 10 --threshold 0.9
```

---

## How It Works

1. **Load Articles**: Read from TSV file
2. **Embed**: Generate semantic embeddings for titles
3. **Store**: Add to ChromaDB collection
4. **Cluster**: Apply DBSCAN to find categories
5. **Name**: Generate category names from top terms

### Input Format (TSV)

```
id	title	source	published
1	Breaking: Ukraine peace talks	cnn	2025-04-15
2	Markets rally on Fed news	bbc	2025-04-14
```

### Output Format (JSON)

```json
{
  "categorization": [
    {
      "category": "War & Conflict",
      "articles": [
        {"id": "1", "title": "Breaking: Ukraine peace talks", "source": "cnn", "score": 0.92},
        {"id": "5", "title": "Israeli strikes continue", "source": "bbc", "score": 0.88}
      ]
    },
    {
      "category": "Economy & Markets",
      "articles": [...]
    }
  ],
  "unclustered": [
    {"id": "10", "title": "Misc article", "source": "reuters"}
  ]
}
```

---

## categorizer.py

Core clustering logic.

### Python API

```python
from categorizer import cluster_articles, generate_category_names

# Cluster using embeddings
clusters = cluster_articles(
    embeddings,
    min_cluster_size=5,
    similarity_threshold=0.85
)

# Generate category names
category_names = generate_category_names(
    cluster_articles,
    article_titles
)
```

### DBSCAN Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `min_cluster_size` | Min articles to form cluster | 5-10 |
| `similarity_threshold` | Cosine similarity | 0.8-0.95 |

---

## chroma_manager.py

ChromaDB management for embeddings storage.

### Functions

```python
from chroma_manager import initialize_chroma, add_articles_to_chroma

# Initialize collection
collection = initialize_chroma(
    collection_name="news_articles",
    persist_dir="./chroma_dbscan"
)

# Add articles
add_articles_to_chroma(collection, articles_df, embeddings)
```

---

## embeddings.py

Embedding generation using sentence transformers.

### Models

- Default: `sentence-transformers/all-MiniLM-L6-v2`
- Alternative: `BAAI/bge-large-en-v1.5`

### Python API

```python
from embeddings import generate_embeddings

titles = ["Article title 1", "Article title 2"]
embeddings = generate_embeddings(titles)
print(embeddings.shape)  # (2, 384)
```

---

## Environment Variables

```bash
# MongoDB (for loading articles)
MONGO_URI=mongodb://root:pass@localhost:27017

# ChromaDB
CHROMA_PATH=./chroma_dbscan
```

## Makefile Targets

```makefile
# Run DBSCAN categorization
dbscan:
	python dbscan/main.py input.tsv --min-size 5

# Category visualization (future)
dbscan-viz:
	python dbscan/visualize.py categorized.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (no input, clustering failure) |

## Troubleshooting

### "No clusters found"

- Reduce `--min-size` (try 3)
- Reduce `--threshold` (try 0.7)

### "All articles unclustered"

- Articles may be too diverse
- Try lower similarity threshold