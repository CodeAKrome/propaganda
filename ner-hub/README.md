# NER Hub (Go)

High-performance NER (Named Entity Recognition) processor written in Go. Processes articles from MongoDB and extracts entities using multiple NER service endpoints in parallel.

## Quick Reference

| Task | Command |
|------|---------|
| Run NER processing | `./RUNME.sh [start-date]` |
| Check endpoints | `./check_endpoints.sh` |
| Build binary | `go build` |
| Run directly | `./multiflair --start-date -3 endpoints.tsv` |

## Shell Scripts

### RUNME.sh
Builds and runs the NER processor with configurable date range.

```bash
./RUNME.sh [start-date]

Arguments:
  start-date    Relative date (e.g., -3 for 3 days ago, -7 for last week)
                Default: -3

Example:
  ./RUNME.sh          # Process last 3 days
  ./RUNME.sh -7       # Process last week
  ./RUNME.sh 2024-01-15  # Process since specific date
```

**What it does:**
1. Builds the Go binary (`go build`)
2. Runs multiflair with the specified start date
3. Processes articles from MongoDB
4. Extracts entities using NER service endpoints
5. Stores results back in MongoDB

### check_endpoints.sh
Checks health status of configured NER endpoints.

```bash
./check_endpoints.sh
# Checks: http://chico.local:8100/health
# Checks: http://harpo.local:8100/health
# Output: Health status for each endpoint

# Add more endpoints by editing the script
```

## Go Application

### main.go
The main NER processor application.

```bash
go run . --start-date -3 endpoints.tsv
```

**Command-line flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--start-date` | -3 | Start date (YYYY-MM-DD or -N) |
| `--end-date` | today | End date (YYYY-MM-DD or -N) |

**Arguments:**
1. endpoints.tsv - Tab-separated file with NER endpoint URLs

**Example endpoints.tsv:**
```
chico	http://chico.local:8100/extract
harpo	http://harpo.local:8100/extract
```

### Endpoints Configuration

The `endpoints.tsv` file defines available NER service endpoints:

```bash
# Format: NAME<tab>URL
chico	http://chico.local:8100/extract
harpo	http://harpo.local:8100/extract
zeppo	http://zeppo.local:8100/extract
```

### Parallel Processing

The NER Hub automatically:
- Distributes work across all configured endpoints
- Load balances based on endpoint availability
- Retries failed requests on other endpoints
- Reports progress with statistics

## Output

### Log File
- `ner.log` - Error and status log (created in working directory)

### MongoDB
Results stored in articles collection:
```json
{
  "_id": "...",
  "ner": {
    "entities": [
      {"text": "President", "label": "PER"},
      {"text": "Washington", "label": "LOC"}
    ],
    "entity_counts": {"PER": 1, "LOC": 1},
    "total_entities": 2
  }
}
```

## Statistics Report

When processing completes, a summary is printed:

```
======================================================================
NER PROCESSOR - SHUTDOWN STATISTICS REPORT
======================================================================
Server: chicok8s:
  Processed:          150
  Already processed:  12
  Errors:             3
  Skipped (too long): 5
  Requests sent:      150
  Latency (min/avg/max): 45ms / 120ms / 450ms

======================================================================
OVERALL TOTALS:
  Newly processed:    150
  Already processed: 12
  Errors:            3
  Total articles:    165
  Processing rate:   12.50 articles/second

ENTITY TYPE DISTRIBUTION:
  PER              523 entities (32.45%)
  ORG              412 entities (25.55%)
  LOC              356 entities (22.07%)
  DATE             145 entities ( 8.99%)
  ...
======================================================================
```

## Requirements

### Go
```bash
go version  # Requires Go 1.21+
```

### MongoDB
```bash
export MONGO_URI="mongodb://localhost:27017"
# Or use MONGO_USER/MONGO_PASS for auth
```

### NER Services
Running NER endpoints (see ../ner/):
```bash
# Start NER service on each worker
docker run -d -p 8100:8100 ner-app
```

## Configuration

### Environment Variables
```bash
# MongoDB (optional)
export MONGO_URI="mongodb://user:pass@localhost:27017"
export MONGO_USER="root"
export MONGO_PASS="example"

# Logging
# Log output goes to ner.log in current directory
```

### Worker Configuration
Each worker machine runs the NER service:
```bash
# On each worker:
cd /path/to/ner
docker build -t ner-app .
docker run -d -p 8100:8100 ner-app

# Verify:
curl http://localhost:8100/health
```

## Troubleshooting

### "connection refused"
- Verify NER service is running on endpoint
- Check firewall rules between machines
- Run `./check_endpoints.sh` to diagnose

### "no articles to process"
- Check MongoDB has articles in date range
- Verify articles have `article` field populated
- Check for `fetch_error` field

### Out of memory
- Reduce worker count in Go code
- Process smaller date ranges
- Increase MongoDB batch size

## Performance

### Typical Throughput
- 10-15 articles/second per endpoint
- Linear scaling with endpoint count
- Latency: 50-500ms per article

### Optimization Tips
1. Add more NER endpoints for parallel processing
2. Use local network for endpoint communication
3. Ensure MongoDB is on fast storage
4. Adjust worker count based on endpoint capacity

## See Also

- Python NER Service: [../ner/README.md](../ner/README.md)
- Main Documentation: [../docs/ner.md](../docs/ner.md)
- MongoDB Articles: [../docs/mongo2chroma.md](../docs/mongo2chroma.md)