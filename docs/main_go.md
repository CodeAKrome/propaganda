# main.go

RSS feed aggregator with user-agent rotation, curl fallback, and MongoDB storage.

## Overview

Fetches RSS feeds from configurable sources, applies text cleaning rules, and stores articles in MongoDB. Supports parallel fetching, retry logic, and user-agent success tracking.

## Usage

```bash
go run main.go <feeds.tsv> <clean-rules.tsv>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `feeds.tsv` | Tab-separated file with feed sources |
| `clean-rules.tsv` | Tab-separated regex rules for text cleaning |

## Feed Config Format (`feeds.tsv`)

Tab-separated with columns: `source_name` | `feed_url`

```
cnn	http://rss.cnn.com/rss/edition.rss
bbc	http://feeds.bbci.co.uk/news/rss.xml
reuters	https://www.reutersagency.com/feed/
```

## Clean Rules Format (`clean-rules.tsv`)

Tab-separated with columns: `source_pattern` | `regex_pattern` | `replacement`

```
cnn	^By .*\n	 
bbc	\s+	 
```

## Examples

### Basic Run

```bash
go run main.go config/big.tsv config/kill.tsv
```

### Test Run (smaller feed)

```bash
go run main.go config/test.tsv config/kill.tsv
```

## Configuration

### Environment Variables

```bash
MONGO_USER=root
MONGO_PASS=example
MONGO_HOST=localhost
MONGO_PORT=27017
```

### Constants (in code)

| Constant | Default | Description |
|----------|---------|-------------|
| `workerCount` | 8 | Parallel fetch workers |
| `requestTimeout` | 15s | HTTP request timeout |
| `maxRetries` | 3 | Max retries per feed |
| `initialBackoff` | 2s | Initial retry backoff |
| `MINLINE` | 128 | Min article length |

### User-Agent Rotation

The system tracks success/failure rates per user-agent and sorts them:

```go
// Scores: success - failure
// Sorted descending for best performer first
```

User-agent stats are persisted in `user_agent_stats.txt`.

### Fallback to curl

If the Go HTTP client fails, it falls back to `curl`:

```bash
curl -s -A "$userAgent" -m 15 "$url"
```

## MongoDB Schema

### Articles Collection

```json
{
  "source": "cnn",
  "title": "Article Title",
  "description": "Article description",
  "link": "https://...",
  "published": "2025-04-02T12:00:00Z",
  "raw": "raw HTML",
  "article": "cleaned article text",
  "fetch_error": null,
  "tags": ["tag1", "tag2"]
}
```

### Stats Collection

```json
{
  "_id": "stats",
  "source_counts": {"cnn": 150, "bbc": 120},
  "updated": "2025-04-02T12:00:00Z"
}
```

## Makefile Integration

```makefile
load:
	go run main.go config/big.tsv config/kill.tsv
```

## Features

- **Parallel fetching**: 8 concurrent workers by default
- **Retry with backoff**: Exponential backoff on failures
- **User-agent rotation**: Best performer used first
- **Curl fallback**: If Go HTTP fails, uses curl
- **Regex cleaning**: Per-source text cleaning rules
- **Backfill**: Fills missing raw/article text from existing content