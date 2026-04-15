# dedupe.py — Deduplication Utility

Removes repeated lines (headers/footers) from RSS articles that appear across multiple articles from the same source.

## Overview

Finds patterns of repeated text that appear in multiple articles from the same source and removes them. Useful for cleaning up RSS feed artifacts like navigation menus, author bios, or boilerplate text that get included repeatedly.

## Usage

```bash
python db/dedupe.py [options]
```

## Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--source NAME` | Process specific source | All sources |
| `--safe` | Safe mode (no writes) | False |

## Examples

### Safe Mode (Preview Changes)

```bash
python db/dedupe.py --safe
```

### Safe Mode for Specific Source

```bash
python db/dedupe.py --source cnn --safe
```

### Apply Changes

```bash
python db/dedupe.py
python db/dedupe.py --source bbc
```

## How It Works

1. **Normalize Lines**: Remove all non-alphanumeric characters for comparison
2. **Find Patterns**: Identify lines appearing in 2+ articles from same source
3. **Filter**: Skip lines < 10 chars, normalized < 5 chars
4. **Remove**: Strip matching repeated lines from articles

## Output Format

```
================================================================================
RSS Article Header/Footer Removal Tool
================================================================================
Mode: LIVE MODE (will modify database)
Target source: ALL_SOURCES
✓ Connected to MongoDB: rssnews.articles
Found 15 unique sources in database

================================================================================
Processing source: cnn
================================================================================
Found 150 articles from cnn
Processed 150 articles with 'article' field content
Found 45 unique line patterns
Found 12 repeated line patterns

Sample of repeated lines (first 10):
  1. 'Sign up for the CNN Newsletter' (in 45 articles)
  2. 'Subscribe to CNN's newsletter' (in 38 articles)
  3. 'Follow CNN on Twitter' (in 25 articles)
  ...

Processing articles...
Articles with 'article' field: 150
Articles without 'article' field: 0
Articles modified: 45/150
Total lines removed: 234

================================================================================
FINAL REPORT
================================================================================

Summary:
  Sources processed: 15
  Total articles: 5000
  Articles modified: 234
  Total lines removed: 1543

Breakdown by source:
Source                                   Articles     Modified     Lines Removed   Repeated Patterns
----------------------------------------------------------------------------------------------------
cnn                                      150          45           234           12
bbc                                       180          67           345           15
reuters                                   200          89           456           18
...

✓ Changes have been applied to the database
```

## Environment Variables

```bash
MONGO_URI=mongodb://root:pass@localhost:27017
MONGO_DB=rssnews
MONGO_COLL=articles
```

## Key Algorithm

```python
# Normalization removes all non-alphanumeric
normalized = re.sub(r"[^a-zA-Z0-9]", "", line.lower())

# Pattern must appear in MORE than one article
repeated_lines = {norm for norm, ids in line_to_articles.items() if len(ids) > 1}
```

## Filter Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Min line length | 10 chars | Avoid false positives |
| Min normalized | 5 chars | Skip very short patterns |
| Min occurrences | 2+ articles | Must be truly repeated |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Connection error |

## Makefile Integration

```makefile
dedupe:
	python db/dedupe.py
```