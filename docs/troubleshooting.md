# Pipeline Troubleshooting Guide

Common issues and solutions for the propaganda pipeline.

## Quick Diagnostics

```bash
# Check all pipeline status
make check-timestamp        # Validate timestamp
ls db/output/*.vec | wc -l  # Count vec files
ls db/output/*.md | wc -l  # Count report files
```

## Common Issues

### 1. Empty vec Files

**Symptom**: vec files contain "No articles match the query"

**Causes**:
- Timestamp is in the future or too old
- Date parsing bug (see fix below)
- MongoDB has no articles in the date range

**Solutions**:
```bash
# Regenerate fresh timestamp
make timestamp

# Or manually set to 2 days ago
echo "2026-04-15T00:00:00Z" > db/timestamp.txt
```

### 2. Date Parsing Bug (FIXED)

**Symptom**: `--start-date -2` returns 0 articles (looking into future)

**Cause**: The `parse_date_arg` function was using `datetime.now() + timedelta(days=-N)` which adds instead of subtracts.

**Fix Applied** (hybrid.py line 153):
```python
# Before (WRONG):
return datetime.now() + timedelta(days=int(remainder))

# After (CORRECT):
return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=int(remainder))
```

### 3. runreport Takes Too Long

**Symptom**: Sequential processing of 62 articles takes hours

**Solution**: Use parallel processing
```bash
python db/runreport.py hybrid_batch.tsv --parallel 4
```

Recommended parallelism: 2-4 for typical workloads.

### 4. DBSCAN Fails with Empty Results

**Symptom**: `make dbscan` produces no clusters

**Cause**: `entity` target must run first to create `output/titles.tsv`

**Solution**:
```bash
make entity
make dbscan
# Or run the full categorize pipeline
make categorize
```

### 5. Missing Titles File

**Symptom**: `output/titles.tsv` not found

**Cause**: Run from wrong directory or entity not run

**Solution**:
```bash
# Run from project root
cd /Users/kyle/hub/propaganda
make entity
```

### 6. vec Files Have "No articles match the filter"

**Symptom**: vec files created but contain no data

**Cause**: Date filter in hybrid.py looking into future (see issue #2)

**Solution**: Apply the fix in hybrid.py as noted above

### 7. ChromaDB ID Fetch Error with Slack Backfill

**Symptom**:
```
ValueError: Expected include item to be one of documents, embeddings, metadatas, distances, got ids in get.
make: *** [vector] Error 1
```

**Cause**: The `--slack` option in `mongo2chroma.py load` uses an outdated ChromaDB API.

**Solution**: Update mongo2chroma.py to use correct API:

```python
# Old (broken):
existing_ids = set(collection.get(include=["ids"])["ids"])

# New (fixed):
total_count = collection.count()
fetch_count = min(total_count, 5000)
result = collection.get(limit=fetch_count, include=["ids"])
existing_ids = set(result.get("ids", []))
```

This fix is already applied in the codebase. If you're using an older version, update to the latest.

### 8. Out of Memory When Loading All Articles

**Symptom**: Vector loading crashes with OOM when loading all ~230K articles

**Cause**: Attempting to load all articles without date filter or limit

**Solution**: Use the `--slack` option with a date range:

```bash
# Load last 2 days with up to 3333 articles (default)
make vector

# Or manually:
python db/mongo2chroma.py load --start-date -2 --slack 3333
```

This processes articles incrementally, backfilling with older articles as needed. Each run accumulates more articles over time.

## Makefile Targets Reference

| Target | Purpose |
|--------|---------|
| `make timestamp` | Generate fresh timestamp |
| `make check-timestamp` | Validate timestamp is recent |
| `make load` | Load RSS feeds to MongoDB |
| `make ner` | Run NER on articles |
| `make t5bias` | Run bias detection |
| `make vector` | Load to ChromaDB |
| `make entity` | Extract titles & entities |
| `make runhybrid` | Generate vec files for batch |
| `make runreport` | Generate markdown reports |
| `make dbscan` | Cluster titles |
| `make mp3small` | Generate TTS audio |

## Environment Variables

```bash
# Required
MONGO_URI=mongodb://root:example@localhost:27017

# Optional
OLLAMA_HOST=localhost:11434
GEMINI_API_KEY=your_key_here
```

For full list, see `.env_example`.