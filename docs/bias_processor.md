# bias_processor.py

LLM-based bias processing pipeline. Detects political bias in news articles using T5 server API and writes results to MongoDB.

## Overview

Processes articles from MongoDB that are missing the `bias` field, sends them to a T5 bias detection server, and stores the results. Supports graceful shutdown, failure recovery, and JSON normalization.

## Usage

```bash
python llm/bias_processor.py [options]
```

## Arguments

### Processing Control

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size N` | Limit number of articles to process | All |
| `--dry-run` | Don't write to database | False |
| `--max-failures N` | Stop after N consecutive failures | 3 |
| `--api-url URL` | T5 server URL | `http://localhost:1337` |
| `--output-file FILE` | Save processed IDs to file | `processed_articles_TIMESTAMP.json` |

### Date Filtering

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date (ISO or -N) | `-7` or `2025-01-01` |
| `--end-date DATE` | End date (ISO or -N) | `-1` |

### ID Filtering

| Option | Description | Example |
|--------|-------------|---------|
| `--id IDS` | Comma-separated MongoDB IDs | `--id 678f3a2b...,679abc12...` |
| `--idfile FILE` | File with MongoDB IDs | `--idfile ids.txt` |

## Examples

### Process Last 7 Days

```bash
python llm/bias_processor.py --start-date -7 --batch-size 100
```

### Process Specific IDs

```bash
python llm/bias_processor.py --idfile article_ids.txt
```

### Dry Run (No Writes)

```bash
python llm/bias_processor.py --dry-run --batch-size 10
```

### Custom API URL

```bash
python llm/bias_processor.py --api-url http://server:1337 --batch-size 50
```

## Output Format

```
Processing: 678f3a2b1c4d...
  Result: {
    "dir": {"L": 0.15, "C": 0.60, "R": 0.25},
    "deg": {"L": 0.10, "M": 0.80, "H": 0.10},
    "reason": "Article presents balanced coverage..."
  }
  SUCCESS - Updated bias field
Completed: 100 processed, 0 failed
Saved 100 processed article IDs to: processed_articles_20250415_143022.json
```

## Bias Result Format

Stored in MongoDB as:

```json
{
  "dir": {"L": 0.15, "C": 0.60, "R": 0.25},
  "deg": {"L": 0.10, "M": 0.80, "H": 0.10},
  "reason": "Article presents balanced coverage..."
}
```

Where:
- `dir`: Political direction scores (L=Left, C=Center, R=Right)
- `deg`: Bias degree (L=Low, M=Medium, H=High)
- `reason`: Explanation of the bias assessment

## Python API

```python
from llm.bias_processor import BiasProcessor, parse_date_arg, parse_id_file

# Initialize processor
processor = BiasProcessor(
    api_url="http://localhost:1337",
    output_file="processed.json"
)

# Count articles without bias
count = processor.count_articles_without_bias(
    start_date="-7",
    end_date="-1"
)
print(f"Found {count} articles to process")

# Process articles
processor.process_articles(
    batch_size=50,
    dry_run=False,
    max_failures=3,
    start_date="-7"
)

# Parse date arguments
from datetime import datetime
date = parse_date_arg("-7")  # 7 days ago
date = parse_date_arg("2025-01-01")  # specific date

# Parse ID file
ids = parse_id_file("ids.txt")

# Close connection
processor.close()
```

## JSON Repair

The processor automatically repairs malformed JSON from the LLM:

- Missing outer braces → added
- Missing quotes on keys → added
- Case variations (left→L, LEFT→L) → normalized
- Alternative field names → mapped to standard

## Environment Variables

```bash
MONGO_URI=mongodb://root:pass@localhost:27017
MONGO_USER=root
MONGO_PASS=your_password
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid args, connection failure) |

## Signal Handling

- SIGINT/SIGTERM: Graceful shutdown, saves processed IDs before exit
- Processed IDs saved to JSON file on completion or shutdown

## Makefile Integration

```makefile
bias:
	python llm/bias_processor.py --batch-size 100 --start-date -3
```