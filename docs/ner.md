# ner-hub/main.go — NER Processor

Named Entity Recognition processor that extracts entities from news articles using multiple NER service endpoints with parallel processing and automatic failover.

## Overview

Processes articles from MongoDB that are missing the `ner` field, extracts named entities using external NER services, and stores results. Supports multiple endpoints, retry logic, graceful shutdown, and detailed statistics.

## Usage

```bash
cd ner-hub && go run main.go [options] <endpoints.tsv>
```

## Arguments

| Position | Description | Example |
|----------|-------------|---------|
| `endpoints.tsv` | NER service endpoints (required) | `endpoints.tsv` |

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config FILE` | Config file path (alt to positional) | `--config endpoints.tsv` |
| `--start-date DATE` | Start date (YYYY-MM-DD or -N) | `--start-date -7` |
| `--end-date DATE` | End date (YYYY-MM-DD or -N) | `--end-date -1` |

## Endpoints Config Format (`endpoints.tsv`)

Tab-separated file with columns: `name` | `url`

```
flair	http://localhost:8100/extract
nerd	http://localhost:8200/extract
```

## Examples

### Basic Run

```bash
cd ner-hub && go run . endpoints.tsv
```

### With Date Range

```bash
cd ner-hub && go run . --start-date -7 --end-date -1 endpoints.tsv
```

### Parallel Processing (Multiple Endpoints)

```bash
cd ner-hub && go run . --start-date -3 multi_endpoints.tsv
```

## Features

### Multi-Endpoint Load Balancing

- Single shared job queue
- Workers from all endpoints consume from same queue
- Automatic failover between endpoints
- Each endpoint has dedicated stats tracking

### Retry Logic

- Up to 3 retry attempts per request
- Exponential backoff on failure
- Handles connection reset, EOF, broken pipe errors
- 500 errors trigger retry

### Graceful Shutdown

- SIGINT/SIGTERM handling
- Stops article fetching immediately
- Allows workers to complete current article
- Saves progress and prints statistics

### Progress Tracking

- Color-coded progress bar
- Real-time stats: processed, errors, skipped
- Latency tracking per endpoint

### Text Length Filtering

- Skips articles > 60,000 characters
- Marks as `ner_skipped: "text_too_long"`

## Output Format

```
======================================================================
NER PROCESSOR - SHUTDOWN STATISTICS REPORT
======================================================================
Shutdown Time:        2025-04-15 14:30:22
Total Runtime:        00:05:43
----------------------------------------------------------------------

flair:
  Newly processed:          150
  Already processed:        45
  Errors:                   5
  Skipped (too long):       10
  Requests sent:          150
  Latency (min/avg/max): 45ms / 120ms / 890ms

----------------------------------------------------------------------
OVERALL TOTALS:
  Newly processed:    150
  Already processed:  45
  Errors:             5
  Skipped (too long): 10
  Total articles:    210

  Total requests:     150
  Global latency (min/avg/max): 45ms / 120ms / 890ms
  Processing rate:    0.46 articles/second

----------------------------------------------------------------------
ENTITY TYPE DISTRIBUTION:
  PERSON               1,234 entities (35.22%)
  ORG                   876 entities (25.00%)
  GPE                   654 entities (18.66%)
  DATE                  432 entities (12.33%)
  EVENT                 198 entities ( 5.65%)
  FAC                    96 entities ( 2.74%)
  PRODUCT               134 entities ( 0.40%)

  Total entities:     3,624
======================================================================
```

## MongoDB Schema

### Input Collection

```json
{
  "_id": ObjectId,
  "title": "Article title",
  "article": "Article text content...",
  "source": "cnn",
  "published": ISODate
}
```

### Output (after NER)

```json
{
  "_id": ObjectId,
  ...
  "ner": {
    "entities": [
      {"text": "Biden", "label": "PERSON"},
      {"text": "White House", "label": "ORG"},
      {"text": "Washington", "label": "GPE"}
    ],
    "entity_counts": {
      "PERSON": 3,
      "ORG": 1,
      "GPE": 1
    },
    "total_entities": 5
  }
}
```

### Skipped (too long)

```json
{
  "ner_skipped": "text_too_long"
}
```

## Environment Variables

```bash
MONGO_USER=root
MONGO_PASS=your_password
```

## Constants (in code)

| Constant | Default | Description |
|----------|---------|-------------|
| `timeout` | 360s | HTTP request timeout |
| `workersPerEndpoint` | 2 | Worker goroutines per endpoint |
| `maxTextLength` | 60000 | Max article length |
| `maxRetries` | 3 | Retry attempts |
| `retryDelay` | 2s | Delay between retries |
| `sharedQueueSize` | 100 | Job queue buffer |

## Makefile Integration

```makefile
ner:
	cd ../ner-hub && go run . --start-date $(NUMDAYS) endpoints.tsv
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid args, connection failure) |

## Logging

- Errors logged to `ner.log`
- Both stderr and log file
- JSON format for machine parsing:
  ```json
  {"server":"flair","operation":"callNERService","error":"connection refused"}
  ```

## Entity Labels

Standard NER labels supported:
- `PERSON` — People
- `ORG` — Organizations
- `GPE` — Geopolitical entities (countries, cities)
- `EVENT` — Events
- `DATE` — Dates
- `FAC` — Facilities
- `PRODUCT` — Products
- `NORP` — Nationalities/religious/political groups
- `MONEY` — Monetary values
- `CARDINAL` — Numbers

## Troubleshooting

### "No articles to process"

All articles already have `ner` field. Use `--start-date` to process new articles.

### High error rate

- Check NER service is running: `curl http://localhost:8100/extract`
- Check network connectivity
- Review `ner.log` for error details

### Slow processing

- Add more endpoints to `endpoints.tsv`
- Workers scale horizontally with endpoints