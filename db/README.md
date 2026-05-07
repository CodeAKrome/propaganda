# Database & Processing Scripts

This directory contains shell scripts for batch processing, vector generation, reporting, and data pipeline operations.

## Quick Reference

| Task | Script |
|------|--------|
| Generate vector file for single topic | `./mkvec.sh` |
| Generate full report for single topic | `./report.sh` |
| Run all topics (vector generation) | `./runmkvecbatch.sh` |
| Run all topics (report generation) | `./runentitybatch.sh` |
| Run bias detection on collected IDs | `./runbias.sh` |
| Run DBSCAN clustering | `./dbscan.sh` |
| Parallel execution | `./run_parallel.sh` |

## Shell Scripts

### Vector Generation

#### mkvec.sh
Generates vector files from article search results using hybrid search.

```bash
./mkvec.sh <date_offset> <filename> <entities> <query> [fulltext]

Arguments:
  date_offset  Relative date (e.g., -1 for yesterday, -7 for last week)
  filename     Output file base name (creates output/filename.vec)
  entities     Entity filter (e.g., "Israel", "Russia,China")
  query        Search query string
  fulltext     Optional: prefix with + for prefilter mode

Example:
  ./mkvec.sh -7 israel_news "Israel" "Summarize events in Israel" +Israel
  # Creates: output/israel_news.vec
```

#### runmkvecbatch.sh
Runs mkvec.sh for all predefined topics in batch mode.

```bash
./runmkvecbatch.sh
# Processes all topics: Israel, Sudan, Climate, Milei, Farage, Iran, etc.
# Creates vector files in output/ directory
```

### Report Generation

#### report.sh
Generates news reports from vector files using LLM analysis with failover support.

```bash
./report.sh <date_offset> <filename> <entities> <query> [fulltext]

Arguments:
  date_offset  Relative date (e.g., -1, -7)
  filename    Output file base name
  entities    Entity filter
  query       Search query
  fulltext    Optional fulltext filter

Output files:
  output/<filename>.vec        - Article vectors
  output/<filename>.cypher     - Extracted relationships
  output/<filename>.md         - Final report
  output/<filename>.reporter   - Reporter prompt

Models used (with failover):
  - Cypher: ollama gpt-oss:20b
  - Report: gemini models/gemini-3-flash-preview → gemini models/gemini-2.5-flash → ollama gpt-oss:20b

Example:
  ./report.sh -7 israel "Israel" "Summarize events in Israel"
```

#### runentitybatch.sh
Runs report.sh for all predefined topics in batch mode.

```bash
./runentitybatch.sh
# Processes ~55 topics and generates reports for each
```

#### batchquery.sh
Template script for running multiple queries with different parameters.

```bash
./batchquery.sh <command> <startdate>

Example:
  ./batchquery.sh ./report.sh -1
```

#### batchquerysmallest.sh
Simplified batch query with reduced topic set for quick testing.

```bash
./batchquerysmallest.sh <command> <startdate>
```

### Bias Detection

#### runbias.sh
Runs bias detection processing on collected article IDs.

```bash
./runbias.sh
# Reads: output/*.ids or output/ids.txt
# Writes: bias data to MongoDB via geminize.py

# Requires:
#  - MongoDB running with articles collection
#  - Ollama with gpt-oss:20b model
```

### Clustering & Analysis

#### dbscan.sh
Runs DBSCAN clustering on vector files.

```bash
./dbscan.sh <workdir> <filename>

Example:
  ./dbscan.sh cluster my_topic
  # Reads: cluster/my_topic.vec
  # Runs clustering and outputs results
```

#### clustervec2md.sh
Batch processes all .vec files in cluster/ directory with DBSCAN.

```bash
./clustervec2md.sh
```

### Testing Scripts

#### test_convert_to_vec.sh
Tests the convert_to_vec.py script with sample categories.

```bash
./test_convert_to_vec.sh
# Runs: convert_to_vec.py ../dbscan/categories.json cluster
```

#### test_dbscan.sh
Tests DBSCAN clustering functionality.

```bash
./test_dbscan.sh
```

#### test_cluster_vec.sh
Tests cluster vector functionality.

```bash
./test_cluster_vec.sh
```

#### test.sh
Basic test runner.

```bash
./test.sh
```

#### test_hybrid.sh
Tests hybrid search functionality.

```bash
./test_hybrid.sh
```

#### test_geminize.py
Tests geminize.py functionality.

```bash
python test_geminize.py
```

### Parallel Execution

#### run_parallel.sh
Runs multiple report tasks in parallel.

```bash
./run_parallel.sh
# Configure parallelism in script header
```

#### parallel.sh
Simple parallel execution runner.

```bash
./parallel.sh
```

#### paralell_runentitybatch.sh
Parallel version of entity batch processing.

```bash
./paralell_runentitybatch.sh
```

### Utility Scripts

#### titles.sh
Extracts article titles from vector files.

```bash
./titles.sh
```

#### triple.sh
Generates triples from data.

```bash
./triple.sh
```

#### quad.sh
Generates quads from data.

```bash
./quad.sh
```

#### q.sh
Quick query utility.

```bash
./q.sh
```

#### runllms.sh
Runs multiple LLM queries.

```bash
./runllms.sh
```

#### mkrunllms.sh
Generates LLM run configurations.

```bash
./mkrunllms.sh
```

#### demo_timeout.sh
Demonstrates timeout handling.

```bash
./demo_timeout.sh
```

### Topic-Specific Scripts

#### runbatch.sh, batch.sh, allbatch.sh
Batch processing scripts for various topics.

#### gaza.sh, g.sh
Gaza-related report generation.

#### portland.sh
Portland-specific reporting.

#### sept.sh
September-related processing.

#### cloud.sh
Cloud-related operations.

#### b.sh, bakbatch.sh, antichrist.sh
Various batch and topic scripts.

#### b.sh
Single-letter alias script.

## Python Scripts (Reference)

See individual documentation for these core Python scripts:

| Script | Documentation |
|--------|---------------|
| [mongo2chroma.py](../docs/mongo2chroma.md) | Vector DB loading |
| [hybrid.py](../docs/hybrid.md) | Hybrid search |
| [geminize.py](../docs/geminize.md) | LLM processing |
| [report.py](report.py) | Report generation (Python version) |
| [dedupe.py](dedupe.py) | Deduplication |

### Additional Python Scripts

#### filter_ansi.py
Filters ANSI escape codes and unwraps fixed-width text from LLM output.

```bash
# Process a single file
python filter_ansi.py < input.md > output.md

# Process all cluster markdown files
for f in db/cluster/*.md; do python filter_ansi.py < "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
```

**Features:**
- Removes ANSI escape codes from LLM output
- Unwraps line-wrapped text (fixes 70-char line breaks)
- Re-joins hyphenated words split across lines
- Handles non-breaking spaces and special characters

**Makefile targets:**
```bash
make cleanclusteransi   # Process db/cluster/*.md
make cleanoutputansi    # Process db/output/*.md  
make cleanmarkdownansi  # Process both
```

#### svo_backfill.py
Backfills Subject-Verb-Object (SVO) extraction for articles.

```bash
# Run SVO backfill (default: processes all articles without SVO)
python svo_backfill.py

# With limit
python svo_backfill.py --limit 1000

# With date filter
python svo_backfill.py --start-date -30
```

See [docs/slack_backfill.md](../docs/slack_backfill.md) for more on vector loading with backfill.

## Configuration

### Environment Variables

```bash
# MongoDB
export MONGO_URI="mongodb://localhost:27017"

# Ollama
export OLLAMA_HOST="localhost:11434"

# Chroma
export CHROMA_PATH="./chroma_db"
```

### Output Directory

All scripts output to the `output/` directory:
- `*.vec` - Vector files
- `*.md` - Generated reports
- `*.cypher` - Extracted relationships
- `*.ids` - Article ID lists

## Quick Start

```bash
# 1. Generate vectors for a topic
./mkvec.sh -7 my_topic "Topic Name" "Search query"

# 2. Generate report
./report.sh -7 my_topic "Topic Name" "Search query"

# 3. Run batch processing
./runmkvecbatch.sh    # Generate all vectors
./runentitybatch.sh   # Generate all reports

# 4. Run bias detection
./runbias.sh
```

## See Also

- Main README: [../README.md](../README.md)
- Documentation: [../docs/](../docs/)
- Makefile targets: [../docs/makefile.md](../docs/makefile.md)