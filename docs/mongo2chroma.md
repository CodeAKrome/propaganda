# mongo2chroma.py

MongoDB → Chroma vector database loader and semantic search engine.

## Overview

Loads articles from MongoDB into ChromaDB for vector similarity search. Supports filtering by date range, entities, and multiple embedding models.

## Commands

### load — Embed articles into Chroma

```bash
python db/mongo2chroma.py load [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `-l, --limit N` | Only process N articles | `--limit 1000` |
| `--start-date DATE` | Articles published on/after date | `--start-date -7` or `--start-date 2025-01-01` |
| `--end-date DATE` | Articles published on/before date | `--end-date -1` |
| `--andentity TEXT` | Entities ALL must match | `--andentity "PERSON/Trump,ORG/UN"` |
| `--orentity TEXT` | Entities AT LEAST ONE must match | `--orentity "GPE/Iran,GPE/Israel"` |
| `--force` | Clear collection and reload | `--force` |
| `--flair-pooled` | Use Flair news-forward (default) | `--flair-pooled` |
| `--bge-large` | Use BAAI/bge-large-en-v1.5 | `--bge-large` |
| `--embedding MODEL` | Custom sentence-transformer model | `--embedding intfloat/e5-base-v2` |

#### Examples

```bash
# Load last 7 days of articles
python db/mongo2chroma.py load --start-date -7 --limit 500

# Load articles about Iran with forced rebuild
python db/mongo2chroma.py load --orentity "GPE/Iran" --force

# Load with BGE embeddings
python db/mongo2chroma.py load --bge-large --limit 1000
```

---

### query — Semantic search

```bash
python db/mongo2chroma.py query "search text" [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `text` | Query string (optional) | `"climate change policy"` |
| `-n, --top N` | Number of results (default: 13) | `-n 5` |
| `--start-date DATE` | Filter by start date | `--start-date -30` |
| `--end-date DATE` | Filter by end date | `--end-date -1` |
| `--andentity TEXT` | Entities ALL must match | `--andentity "PERSON/Biden"` |
| `--orentity TEXT` | Entities at least one must match | `--orentity "ORG/NATO,ORG/EU"` |
| `--showentity` | Show entities in results | `--showentity` or `--showentity "PERSON,ORG"` |
| `--flair-pooled` | Use Flair embeddings | `--flair-pooled` |
| `--bge-large` | Use BGE-large embeddings | `--bge-large` |
| `--embedding MODEL` | Custom model | `--embedding sentence-transformers/all-MiniLM-L6-v2` |

#### Examples

```bash
# Basic semantic search
python db/mongo2chroma.py query "Russian invasion of Ukraine" -n 10

# Search with date filter
python db/mongo2chroma.py query "election results" --start-date -7 --end-date -1

# Search with entity filtering
python db/mongo2chroma.py query "military action" --andentity "GPE/Ukraine" --showentity

# Search without text (entity-only)
python db/mongo2chroma.py query "" --orentity "PERSON/Netanyahu" -n 5
```

---

### title — Export article titles

```bash
python db/mongo2chroma.py title [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date filter | `--start-date -30` |
| `--end-date DATE` | End date filter | `--end-date -1` |

#### Example

```bash
python db/mongo2chroma.py title --start-date -7
# Output: 2025-04-02	cnn	678f3a2b1c4d...	Article Title Here
```

---

### dumpentity — Export entity counts

```bash
python db/mongo2chroma.py dumpentity [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date filter | `--start-date -30` |
| `--end-date DATE` | End date filter | `--end-date -1` |
| `--showentity` | Filter to specific entity types | `--showentity "PERSON,ORG"` |

#### Example

```bash
python db/mongo2chroma.py dumpentity --start-date -7
# Output: 42	PERSON	Donald Trump
#         38	ORG	Hamas
#         15	GPE	Israel
```

---

### article — Export full articles

```bash
python db/mongo2chroma.py article [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `-n, --top N` | Maximum results | `-n 10` |
| `--id IDS` | Comma-separated MongoDB IDs | `--id 678f3a2b...,679abc12...` |
| `--idfile FILE` | File with IDs (one per line) | `--idfile ids.txt` |
| `--start-date DATE` | Start date filter | `--start-date -30` |
| `--end-date DATE` | End date filter | `--end-date -1` |
| `--andentity TEXT` | All entities must match | `--andentity "PERSON/Biden"` |
| `--orentity TEXT` | At least one entity | `--orentity "GPE/Russia,GPE/Ukraine"` |
| `--showentity` | Show entities in output | `--showentity` |

#### Example

```bash
# Export articles by ID file
python db/mongo2chroma.py article --idfile ids.txt -n 5

# Export recent articles with entities
python db/mongo2chroma.py article --start-date -3 --showentity
```

## Date Format

Supports two formats:
- **Negative days**: `-7` = 7 days ago, `-1` = yesterday
- **ISO-8601**: `2025-09-06` or `2025-09-06T08:00:00+00:00`

## Entity Format

Entities can be specified with optional label prefix:
- `TEXT` — Match any entity with this text
- `LABEL/TEXT` — Match entity with specific label

Examples: `PERSON/Trump`, `ORG/UN`, `GPE/Iran`, `Israel`

## Environment Variables

```bash
# MongoDB Configuration
MONGO_URI=mongodb://user:pass@host:27017
MONGO_USER=root
MONGO_PASS=your_password_here
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=rssnews          # default
MONGO_COLL=articles       # default

# ChromaDB
CHROMA_PATH=./chroma_db   # default
```

For full list of environment variables, see `.env.example`.

## Architecture

- **MongoDB**: Source of article data (raw text, metadata, NER entities)
- **ChromaDB**: Vector store for semantic similarity search
- **Embedding Models**: Flair, BGE-large, or custom sentence-transformers