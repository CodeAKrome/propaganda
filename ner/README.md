# Named Entity Recognition (NER)

This directory contains the NER (Named Entity Recognition) service using Flair NLP, along with scripts for processing articles and managing the service.

## Quick Reference

| Task | Script |
|------|--------|
| Start NER service | `./run.sh` or `docker run -d -p 8100:8100 ner-app` |
| Run NER processing | `./RUNME.sh [date]` |
| Build Docker image | `./build.sh` |
| Distribute NER workload | `./distributener.sh` |

## Shell Scripts

### RUNME.sh
Main entry point for NER processing - called from Makefile.

```bash
./RUNME.sh [date]

Arguments:
  date    Optional date (YYYY-MM-DD). Defaults to today.

Example:
  ./RUNME.sh              # Process today's articles
  ./RUNME.sh 2024-01-15   # Process articles from specific date

Requirements:
  - Conda environment with flair installed
  - MongoDB running with articles collection
  - NER service running on port 8100
```

### run.sh
Starts the NER Docker container.

```bash
./run.sh
# Runs: docker run -d -p 8100:8100 ner-app
# Starts NER service on port 8100
```

### build.sh
Builds the NER Docker image.

```bash
./build.sh
# Builds Docker image from Dockerfile
# Creates: ner-app container
```

### distributener.sh
Distributes NER processing across multiple machines.

```bash
./distributener.sh
# Uses scp to copy scripts to worker nodes
# Requires SSH access to configured hosts
# See script for target hosts and configuration
```

## Python Scripts

### ner.py
FastAPI-based NER microservice using Flair NLP.

```bash
cd /Users/kyle/hub/propaganda/ner
python ner.py

# Runs on: http://localhost:8100
# Endpoint: POST /extract
```

**Features:**
- Named Entity Recognition using Flair
- Entity counting and statistics
- Service health monitoring
- Request/response tracking

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract` | POST | Extract entities from text |
| `/health` | GET | Service health check |
| `/stats` | GET | Service statistics |
| `/` | GET | Service info |

**Environment:**
```bash
pip install flair fastapi uvicorn
```

### processor.py
Article processor that queries MongoDB and calls NER service.

```bash
python -c "
import sys
sys.path.insert(0, '.')
from processor import process_date
process_date('2024-01-15')
"
```

## Docker

### Building the Image

```bash
cd /Users/kyle/hub/propaganda/ner
docker build -t ner-app .
```

### Running the Container

```bash
# Run with default settings
docker run -d -p 8100:8100 ner-app

# Run with custom port
docker run -d -p 8100:8100 -e PORT=8100 ner-app

# View logs
docker logs -f ner-app
```

### Docker Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| PORT | 8100 | Service port |
| MAX_TEXT_LENGTH | 50000 | Max characters per request |
| MAX_SENTENCE_LENGTH | 10000 | Flair sentence limit |

## API Reference

### Extract Entities

```bash
curl -X POST http://localhost:8100/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "President Obama visited Washington yesterday."}'
```

**Response:**
```json
{
  "entities": [
    {"text": "President", "label": "PER"},
    {"text": "Obama", "label": "PER"},
    {"text": "Washington", "label": "LOC"}
  ],
  "entity_counts": {"PER": 2, "LOC": 1},
  "total_entities": 3
}
```

### Health Check

```bash
curl http://localhost:8100/health
```

## Configuration

### Environment Variables

```bash
# MongoDB
export MONGO_URI="mongodb://localhost:27017"

# NER Service
export NER_URL="http://localhost:8100/extract"
```

### Supported Entity Types

Flair's `ner-ontonotes-fast` model provides:
- PER (Person)
- ORG (Organization)
- LOC (Location)
- DATE (Date)
- TIME (Time)
- MONEY (Money)
- PERCENT (Percent)

## Troubleshooting

### Service won't start
```bash
# Check if port is in use
lsof -i :8100

# Check Docker logs
docker logs ner-app
```

### No entities extracted
```bash
# Verify NER service is running
curl http://localhost:8100/health

# Check MongoDB connection
mongosh -u root -p example
```

### Out of memory
```bash
# Reduce batch size in processor.py
# Use smaller Flair model: ner-ontonotes-fast (vs ner-ontonotes)
```

## Output

Log file: `ner_service.log` (created in working directory)

## See Also

- Main README: [../README.md](../README.md)
- Documentation: [../docs/ner.md](../docs/ner.md)
- NER Hub (Go): [../ner-hub/README.md](../ner-hub/README.md)