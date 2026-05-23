# Timestamp Mechanism

The timestamp system controls the date range for all pipeline operations. It ensures consistency across MongoDB queries, vector generation, and report processing.

## Overview

The pipeline uses a single timestamp file (`db/timestamp.txt`) that all Makefile targets read to determine which articles to process. This ensures all targets operate on the same date range.

## Files

- **Location**: `db/timestamp.txt`
- **Format**: ISO 8601 with Z suffix (e.g., `2026-04-16T14:30:00Z`)
- **Contents**: Single line with UTC timestamp

## How It Works

### 1. Generation

The timestamp is created by `db/mktimestamp.py`:

```bash
# Default: 3 days ago
python db/mktimestamp.py

# Custom: N days ago
python db/mktimestamp.py 7
```

### 2. Makefile Integration

The Makefile reads the timestamp to set `NUMDAYS`:

```makefile
NUMDAYS := $(shell cat db/timestamp.txt 2>/dev/null | cut -d'T' -f1)
NUMDAYS ?= $(shell date +%F)  # fallback to today if file missing
```

This is used by all targets:
- `make load` - Load articles since NUMDAYS
- `make ner` - Process NER for articles since NUMDAYS
- `make t5bias` - Run bias detection since NUMDAYS
- `make vector` - Load to ChromaDB since NUMDAYS
- `make entity` - Extract entities since NUMDAYS
- `make runhybrid` - Generate vectors since NUMDAYS

### 3. Automatic Update

The `timestamp` target runs automatically at the start of `testrun`:

```makefile
testrun: timestamp load ner t5bias vector entity ...
```

This ensures the timestamp is always fresh before processing.

## Commands

### Check Current Timestamp

```bash
cat db/timestamp.txt
```

### Manually Set Timestamp

```bash
# 3 days ago (default)
make timestamp

# Custom days
python db/mktimestamp.py 7
```

### Validate Timestamp

```bash
# Check if timestamp is recent (within 24 hours)
make check-timestamp
```

## Troubleshooting

### Problem: Empty vec files ("No articles match the query")

**Cause**: Timestamp is too old - MongoDB has no articles in that date range.

**Solution**:
```bash
make timestamp  # Regenerate timestamp
make mkvec      # Regenerate vec files
```

### Problem: Inconsistent results across targets

**Cause**: Timestamp file modified during pipeline run.

**Solution**:
```bash
# Run timestamp first to ensure consistency
make timestamp && make testrun
```

### Problem: Fallback to today's date

**Cause**: `db/timestamp.txt` missing or unreadable.

**Solution**:
```bash
make timestamp  # Recreate timestamp file
```

## Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESTAMP_OFFSET` | `3` | Days ago for default timestamp |
| `NUMDAYS` | From timestamp file | Date range for processing |

## Examples

### View current timestamp

```bash
$ cat db/timestamp.txt
2026-04-13T21:11:41Z
```

### Force today's date

```bash
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > db/timestamp.txt
```

### Check if timestamp needs refresh

```bash
# Compare timestamp date to current date
current=$(date -u +%Y-%m-%d)
ts=$(cut -d'T' -f1 db/timestamp.txt)
if [[ "$current" != "$ts" ]]; then
    echo "Timestamp is stale - run 'make timestamp'"
fi
```