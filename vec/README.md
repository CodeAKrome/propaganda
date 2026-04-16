# Vec - Vector Database & Graph Integration

Integration between MongoDB vector data and Memgraph graph database for knowledge graph queries.

## Quick Reference

| Task | Script/Command |
|------|----------------|
| Set environment | `source doenv.sh` |
| Install Memgraph | `./install_memgraph.sh` |
| Run Mongo to Memgraph | `python mongo2memgraph.py` |

## Shell Scripts

### doenv.sh
Sets environment variables for Memgraph connection.

```bash
source doenv.sh

# Sets:
# - MONGO_URI=mongodb://user:pass@localhost:27017/rssnews
# - MEMGRAPH_URI=bolt://localhost:7687

# Usage:
# source doenv.sh && python mongo2memgraph.py
```

### install_memgraph.sh
Installs Memgraph database.

```bash
./install_memgraph.sh

# Runs: curl -sSf "https://install.memgraph.com" | sh
# Follow prompts for your platform

# Alternative (Docker):
docker run -p 7687:7687 -p 3000:3000 memgraph/memgraph
```

## Python Scripts

### mongo2memgraph.py
Transfers vector data from MongoDB to Memgraph.

```bash
python mongo2memgraph.py [options]

Options:
  --limit <n>      Limit number of articles (default: all)
  --start-date     Filter by start date
  --end-date       Filter by end date
  --dry-run        Show what would be transferred

Example:
python mongo2memgraph.py --limit 100
python mongo2memgraph.py --start-date -7
```

## Configuration

### Environment Variables

```bash
# MongoDB
export MONGO_URI="mongodb://user:pass@localhost:27017/rssnews"

# Memgraph
export MEMGRAPH_URI="bolt://localhost:7687"
export MEMGRAPH_USER="memgraph"
export MEMGRAPH_PASS="memgraph"
```

### Memgraph Connection

| Setting | Default | Description |
|---------|---------|-------------|
| Host | localhost | Memgraph server |
| Port | 7687 | Bolt protocol port |
| Web Port | 3000 | Memgraph Lab web UI |

## Memgraph Usage

### Connect via CLI
```bash
memgraph
```

### Connect via Python
```bash
pip install gqlalchemy
```

```python
from gqlalchemy import Memgraph

memgraph = Memgraph(host="localhost", port=7687)

# Query
result = memgraph.execute_and_fetch("MATCH (n) RETURN n LIMIT 5")
```

### Web Interface
```bash
# Open Memgraph Lab
http://localhost:3000
```

## Knowledge Graph Schema

### Nodes
- `Article` - News article
- `Entity` - Named entity (PER, ORG, LOC)
- `Source` - News source

### Relationships
- `HAS_ENTITY` - Article contains entity
- `LINKED_TO` - Entities related
- `PUBLISHED_BY` - Article from source

## Example Queries

### Find articles with specific entity
```cypher
MATCH (a:Article)-[:HAS_ENTITY]->(e:Entity {text: "Putin"})
RETURN a.title, a.published
LIMIT 10
```

### Find related entities
```cypher
MATCH (e1:Entity)-[:LINKED_TO]->(e2:Entity)
WHERE e1.text = "Ukraine"
RETURN e2.text, count(*) as rel_count
ORDER BY rel_count DESC
LIMIT 20
```

### Entity co-occurrence
```cypher
MATCH (a:Article)-[:HAS_ENTITY]->(e1:Entity),
      (a)-[:HAS_ENTITY]->(e2:Entity)
WHERE e1.text = "Russia"
RETURN e2.text, count(*) as co_occur
ORDER BY co_occur DESC
LIMIT 15
```

## Troubleshooting

### Connection refused
```bash
# Check Memgraph is running
docker ps | grep memgraph

# Or start manually
memgraph
```

### No data transferred
- Check MongoDB has articles
- Verify date filters
- Run with --dry-run to debug

### Query slow
- Create indexes in Memgraph
- Use LIMIT in queries
- Check dataset size

## See Also

- Main README: [../README.md](../README.md)
- MongoDB: [../docs/mongo2chroma.md](../docs/mongo2chroma.md)
- Memgraph Docs: https://memgraph.com/docs