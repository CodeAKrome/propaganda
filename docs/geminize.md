# geminize.py

Process MongoDB articles through LLM (Gemini, Ollama, or MLX) for analysis, summarization, or entity extraction.

## Overview

Reads articles from MongoDB, sends them to an LLM with a custom prompt, and writes the results back to MongoDB. Supports multiple backends and flexible filtering.

## Usage

```bash
python db/geminize.py [options]
```

## Arguments

### Backend Selection

| Option | Description | Example |
|--------|-------------|---------|
| `--model NAME` | Gemini model (default: gemini-2.0-flash) | `--model gemini-1.5-pro` |
| `--ollama` | Use Ollama instead of Gemini | `--ollama` |
| `--mlx` | Use MLX (Apple Silicon) instead of Gemini | `--mlx` |

### Data Source/Target

| Option | Description | Example |
|--------|-------------|---------|
| `--src FIELD` | Source field in MongoDB (default: article) | `--src raw_text` |
| `--dst FIELD` | Destination field to write (required) | `--dst summary` |
| `--id IDS` | Specific MongoDB IDs to process | `--id 678f3a2b...,679abc12...` |
| `--idsource FILE` | File with MongoDB IDs (one per line) | `--idsource ids.txt` |
| `--idfile FILE` | Output file for processed IDs | `--idfile processed.txt` |

### Date & Source Filters

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date (ISO or -N days) | `--start-date -7` |
| `--end-date DATE` | End date (ISO or -N days) | `--end-date -1` |
| `--news SOURCES` | Filter by news sources | `--news cnn,bbc,reuters` |

### Query/Prompt

| Option | Description | Example |
|--------|-------------|---------|
| `-q, --query TEXT` | Prompt for the LLM | `-q "Summarize this article"` |
| `--queryfile FILE` | Read prompt from file | `--queryfile prompt.txt` |
| `-n N` | Limit number of records | `-n 10` |

### Behavior

| Option | Description | Example |
|--------|-------------|---------|
| `--update` | Overwrite existing destination field | `--update` |
| `--dry-run` | Show what would be processed | `--dry-run` |
| `--json` | Extract JSON from LLM output | `--json` |
| `--prompt-cache-file PATH` | MLX prompt cache file | `--prompt-cache-file cache.safetensors` |

## Examples

### Basic Summarization with Gemini

```bash
python db/geminize.py \
  --dst summary \
  -q "Provide a 2-sentence summary of this news article" \
  --start-date -7
```

### Entity Extraction with Ollama

```bash
python db/geminize.py \
  --ollama \
  --dst entities \
  -q "Extract all people, organizations, and locations from this article as JSON" \
  --json \
  --start-date -3
```

### Bias Analysis with MLX

```bash
python db/geminize.py \
  --mlx \
  --dst bias \
  -q "Analyze the political bias of this article. Return JSON with direction (left/center/right) and degree (0-1)" \
  --json \
  -n 50
```

### Process Specific IDs

```bash
python db/geminize.py \
  --dst analysis \
  -q "What is the main topic?" \
  --id 678f3a2b1c4d5e6f,ghi789...
```

### Process from ID file

```bash
python db/geminize.py \
  --dst summary \
  -q "Summarize" \
  --idsource articles_to_process.txt \
  --idfile done.txt
```

### Read prompt from file

```bash
python db/geminize.py \
  --dst sentiment \
  --queryfile prompts/sentiment_analysis.txt \
  --start-date -1
```

### Dry run to test query

```bash
python db/geminize.py \
  --dst test \
  -q "What is this about?" \
  --dry-run \
  -n 3
```

## Prompt Templates

### Summarization

```
Summarize this news article in 2-3 sentences. Focus on the key facts and who/what/when/where.
```

### Entity Extraction

```
Extract all named entities from this article. Return as JSON:
{"persons": [], "organizations": [], "locations": [], "dates": []}
```

### Bias Analysis

```
Analyze the political bias of this article. Return JSON:
{"direction": "left|center|right", "degree": 0.0-1.0, "reason": "explanation"}
```

### Sentiment

```
Classify the sentiment of this article: positive, negative, or neutral.
Return JSON: {"sentiment": "...", "confidence": 0.0-1.0}
```

## Environment Variables

```bash
MONGO_URI=mongodb://user:pass@host:27017
MONGO_DB=rssnews
GEMINI_API_KEY=your_api_key   # for Gemini backend
OLLAMA_HOST=localhost:11434    # for Ollama backend
```

## Output

The script prints progress as it processes:

```
Processing 10 articles...
  [1/10] ID: 678f3a2b... ✓
  [2/10] ID: 679abc12... ✓
  [3/10] ID: 680def34... ✗ (empty article)
Done. 9 processed, 1 skipped.
IDs written to ids.txt
```

## Error Handling

- Skips articles where source field is empty
- Logs errors but continues processing
- Use `--dry-run` to test before processing many records