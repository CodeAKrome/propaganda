# Slack Backfill for Vector Loading

Load articles into ChromaDB with intelligent backfilling - if fewer articles exist in the date range than the target, progressively load older articles to reach the target count.

## Overview

The `--slack` option in `mongo2chroma.py load` enables intelligent backfilling:

1. **First pass**: Load all articles in the date range (e.g., last 2 days) not already in ChromaDB
2. **Check**: Compare processed count to slack target
3. **Backfill**: If processed < slack, fetch older articles from MongoDB until target is reached
4. **Stop**: Continue until `slack` articles are processed

## Quick Start

### Using Makefile (Default: 3333 articles)

```bash
make vector
```

This is equivalent to:
```bash
python db/mongo2chroma.py load --start-date -2 --slack 3333
```

### Custom Slack Value

```bash
python db/mongo2chroma.py load --start-date -2 --slack 1000
```

### Without Slack (Original Behavior)

```bash
python db/mongo2chroma.py load --start-date -2
```

---

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--start-date DATE` | Date filter for initial load (ISO or -N days) | Required |
| `--slack N` | Target number of articles to process | None (original behavior) |
| `--limit N` | Hard limit on total articles | None |
| `--force` | Clear ChromaDB and reload all | False |

---

## How It Works

### Flow Diagram

```
┌─────────────────────────────────────────┐
│  mongo2chroma.py load --slack 3333     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  1. Load articles from --start-date     │
│     (last 2 days by default)            │
│     Skip if already in ChromaDB          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  2. Count articles stored: 108         │
│     Slack target: 3333                  │
│     Need backfill: 3225                 │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. Backfill:                           │
│     - Get existing ChromaDB IDs         │
│     - Query MongoDB for articles         │
│       NOT in ChromaDB (any date)         │
│     - Sort by published DESC             │
│     - Load until slack reached           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  4. Total stored: 3333                  │
│     Done!                                │
└─────────────────────────────────────────┘
```

---

## Example Outputs

### Standard Run with Slack

```bash
$ python db/mongo2chroma.py load --start-date -2 --slack 3333

Counting documents to load...
Vectorizing: 100%|████████████████████████████████| 2356/2356 [05:51<00:00,  6.71it/s]
Skipped 2248 documents already in Chroma (use --force to reload)
Slack target: 3333, processed: 108, need 3225 more - backfilling older articles
Found 3225 additional articles to backfill
Backfilling: 100%|████████████████████████████| 3225/3225 [03:12<00:00, 16.81it/s]
Backfill complete. Total stored: 3333
✅  Stored 3333 new vectors in Chroma
```

### Run Without Slack (Original Behavior)

```bash
$ python db/mongo2chroma.py load --start-date -7

Counting documents to load...
Vectorizing: 100%|████████████████████████████████| 1523/1523 [02:15<00:00, 11.25it/s]
✅  Stored 1523 new vectors in Chroma
```

### Run with Slack (All Date Range Already Loaded)

```bash
$ python db/mongo2chroma.py load --start-date -2 --slack 3333

Counting documents to load...
Vectorizing: 100%|████████████████████████████████| 3333/3333 [04:30<00:00, 12.30it/s]
Slack target: 3333, processed: 3333
✅  Stored 3333 new vectors in Chroma
```

### Run with Slack (No New Articles in Date Range)

```bash
$ python db/mongo2chroma.py load --start-date -2 --slack 1000

Counting documents to load...
Vectorizing: 100%|████████████████████████████████| 5/5 [00:05<00:00,  1.01it/s]
Skipped 4995 documents already in Chroma (use --force to reload)
Slack target: 1000, processed: 5, need 995 more - backfilling older articles
Found 234387 additional articles to backfill
Backfilling: 100%|████████████████████████████| 995/995 [01:12<00:00, 13.74it/s]
Backfill complete. Total stored: 1000
✅  Stored 1000 new vectors in Chroma
```

---

## Makefile Configuration

The `SLACK_VECTOR` constant in Makefile controls the default slack value:

```makefile
# Makefile line 10
SLACK_VECTOR = 3333
```

To change the default, edit this line in Makefile or override at runtime:

```bash
make vector SLACK_VECTOR=5000
```

---

## Error Handling

### ChromaDB ID Fetch Error (Fixed)

**Previous Error:**
```
ValueError: Expected include item to be one of documents, embeddings, metadatas, distances, got ids in get.
```

**Solution:** The code now uses the correct ChromaDB API:
```python
# Fixed: Use limit parameter instead of include=["ids"]
result = collection.get(limit=fetch_count, include=["ids"])
existing_ids = set(result.get("ids", []))
```

If the ID fetch fails, a warning is printed and backfill continues without deduplication.

### Memory Limits

To prevent memory issues with very large ChromaDB collections, the existing ID fetch is limited to 5000 IDs:

```python
fetch_count = min(total_count, 5000)
if total_count > fetch_count:
    print(f"Warning: ChromaDB has {total_count} IDs, only checking first {fetch_count}")
```

---

## Use Cases

### 1. Gradual Buildup of Vector Database

Instead of loading all 230K+ articles at once (which causes OOM), use slack to gradually accumulate:

```bash
# Day 1: Load 3333
make vector

# Day 2: Load next 3333 (plus any new from last 2 days)
make vector

# Day 3: ...
make vector
```

Over time, ChromaDB accumulates all articles through incremental runs.

### 2. Focused Backfill

If a specific topic needs more coverage:

```bash
python db/mongo2chroma.py load \
  --orentity "GPE/Ukraine" \
  --slack 500 \
  --start-date -30
```

### 3. Force Rebuild with Accumulation

To rebuild while keeping existing vectors:

```bash
# Don't use --force - it clears everything
# Instead, use normal slack loading which skips existing
python db/mongo2chroma.py load --slack 3333
```

---

## Comparison: With vs Without Slack

| Behavior | Without `--slack` | With `--slack` |
|----------|-------------------|-----------------|
| Date range | Uses `--start-date` | Uses `--start-date` (initial) |
| Article count | All in date range | Up to `slack` (backfills older if needed |
| Duplicate handling | Skips existing | Skips existing |
| Memory usage | Lower | Moderate (more articles) |
| Use case | Quick sync | Full accumulation |

---

## Files

| File | Description |
|------|-------------|
| `db/mongo2chroma.py` | Main script with slack logic |
| `Makefile` | Contains `SLACK_VECTOR = 3333` constant |
| `docs/slack_backfill.md` | This documentation |
| `docs/diagrams/slack_flow.svg` | Flow diagram |