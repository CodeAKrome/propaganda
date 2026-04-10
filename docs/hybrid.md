# hybrid.py

Hybrid search combining ChromaDB vector similarity with MongoDB filtering and optional BM25 reranking.

## Overview

Performs semantic search using vector embeddings, then filters and optionally reranks results using BM25. Supports entity filtering, date ranges, and full-text search.

## Usage

```bash
python db/hybrid.py [query] [options]
```

## Arguments

### Query

| Position | Description | Example |
|----------|-------------|---------|
| `text` | Semantic search query (optional) | `"Ukraine war latest"` |

### Entity Filtering

| Option | Description | Example |
|--------|-------------|---------|
| `--andentity TEXT` | Entities ALL must be present | `--andentity "PERSON/Trump,ORG/NATO"` |
| `--orentity TEXT` | At least ONE entity must match | `--orentity "GPE/Iran,GPE/Israel"` |
| `--substr` | Enable substring matching | `--substr` |

### Date Filtering

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date (ISO or -N) | `--start-date -7` |
| `--end-date DATE` | End date (ISO or -N) | `--end-date -1` |

### Text Filtering

| Option | Description | Example |
|--------|-------------|---------|
| `--fulltext TEXT` | MongoDB full-text search (OR) | `--fulltext "climate,environment"` |
| `--search TERMS` | All terms must appear (AND) | `--search "war,Ukraine,peace"` |
| `--orsearch TERMS` | At least one term (OR) | `--orsearch "meeting,summit"` |

### Results

| Option | Description | Example |
|--------|-------------|---------|
| `-n, --top N` | Number of results (default: 10) | `-n 20` |
| `--ids FILE` | Write matched IDs to file | `--ids output_ids.txt` |
| `--showentity` | Show entities in output | `--showentity` or `--showentity "PERSON,ORG"` |

### BM25 Reranking

| Option | Description | Example |
|--------|-------------|---------|
| `--bm25` | Enable BM25 reranking | `--bm25` |
| `--bm25-query TEXT` | Separate BM25 query | `--bm25-query "peace talks"` |

### Embedding Model

| Option | Description | Example |
|--------|-------------|---------|
| `--flair-pooled` | Use Flair news-forward (default) | `--flair-pooled` |
| `--bge-large` | Use BGE-large | `--bge-large` |
| `--embedding MODEL` | Custom sentence-transformer | `--embedding intfloat/e5-base-v2` |

## Examples

### Basic Semantic Search

```bash
python db/hybrid.py "climate change policy"
```

### Search with Entity Filter

```bash
python db/hybrid.py "military operations" --andentity "GPE/Ukraine" -n 15
```

### Search with Date Range

```bash
python db/hybrid.py "election results" --start-date -7 --end-date -1
```

### Vector + Full-text Hybrid

```bash
python db/hybrid.py "diplomatic talks" --fulltext "summit,meeting" -n 10
```

### BM25 Reranking

```bash
python db/hybrid.py "peace negotiations" --bm25 --bm25-query "ceasefire agreement" -n 5
```

### Save IDs to File

```bash
python db/hybrid.py "breaking news" --start-date -3 --ids matched_ids.txt
```

### OR Entity Search

```bash
python db/hybrid.py "" --orentity "PERSON/Netanyahu,PERSON/Gantz" -n 10
```

### Show Entities in Results

```bash
python db/hybrid.py "conflict update" --showentity
```

### Substring Matching

```bash
python db/hybrid.py "war" --orentity "GPE/Ukr" --substr
```

### Multiple Filters Combined

```bash
python db/hybrid.py " humanitarian crisis" \
  --start-date -14 \
  --end-date -1 \
  --orentity "GPE/Gaza,GPE/Palestine" \
  --search "aid,relief" \
  --bm25 \
  -n 20
```

## Output Format

```
---
ID: 678f3a2b1c4d...
Title: Article Title
Published: 2025-04-02
Source: cnn
---
Text: Article content...
```

With `--showentity`:
```
---
Entities:
PERSON: Donald Trump, Joe Biden
ORG: NATO, EU
GPE: Ukraine, Russia
---
```

## Architecture

1. **Vector Search**: Query ChromaDB for similar documents
2. **Date Filter**: Filter by publication date in MongoDB
3. **Entity Filter**: Filter by NER entities in MongoDB
4. **Text Filter**: Apply AND/OR text filters (--search, --orsearch, --fulltext)
5. **BM25 Rerank**: Optional reranking of vector results by BM25 score

## Environment Variables

```bash
# MongoDB
MONGO_URI=mongodb://user:pass@host:27017
MONGO_USER=root
MONGO_PASS=your_password_here

# ChromaDB
CHROMA_PATH=./chroma_db

# NER Service
NER_URL=http://localhost:8100/extract
```

For full list of environment variables, see `.env.example`.