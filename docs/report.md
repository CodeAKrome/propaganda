# report.py

Generate news reports from article vectors using LLM analysis with multi-model failover.

## Overview

Takes vector search results (from `hybrid.py` or `mongo2chroma.py`), generates relationship graphs (SVO triples), and produces a TV-style news report using LLM failover (Gemini → Ollama).

## Usage

```bash
python db/report.py [arguments]
```

## Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `startdate` | Date offset (-N days) or ISO date | `-7` or `2025-04-01` |
| `filename` | Output base name | `iran_report` |
| `entity` | Entity to analyze | `Iran` or `Iran,Israel` |
| `query` | Report query/prompt | `"Summarize the latest developments"` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `svoprompt` | Path to SVO prompt file | `prompt/kgsvo.txt` |
| `workdir` | Working directory | `output` |

## Examples

### Basic Report

```bash
python db/report.py -7 iran_news "Iran nuclear developments" Iran
```

This generates:
- `output/iran_news.cypher` — Extracted relationships
- `output/iran_news.txt` — Final news report

### Multi-Entity Report

```bash
python db/report.py -14 middle_east "Middle East conflict analysis" "Iran,Israel,Gaza"
```

### Custom Prompt

```bash
python db/report.py -3 ukraine "Ukraine war update" Ukraine --svoprompt prompt/custom_svo.txt
```

### Custom Output Directory

```bash
python db/report.py -7 report_output "Climate policy analysis" "Climate,Environment" --workdir ./reports
```

## Workflow

1. **Read Vector File**: Reads `.vec` file from `hybrid.py` or `mongo2chroma.py` query output
2. **Generate SVO**: Extracts Subject-Verb-Object triples using LLM
3. **Generate Report**: Produces TV-style report with bias analysis
4. **Failover**: Tries Gemini models first, then Ollama if all fail

## Output Files

| File | Description |
|------|-------------|
| `{filename}.cypher` | Extracted relationship triples |
| `{filename}.txt` | Final news report |

## Prompt Files

### SVO Prompt (`prompt/kgsvo.txt`)

Template for extracting subject-verb-object relationships from articles:

```
Extract all relationships from the articles.
Format each as: ("subject","object","verb","description")
...
```

### Reporter Prompt

Built-in prompt that instructs the LLM to:
- Summarize major themes
- Analyze political bias (direction: L/C/R, degree: L/M/H)
- Speak in professional newscaster tone
- Not use markup or tables

## Environment Variables

```bash
MONGO_URI=mongodb://user:pass@host:27017
GEMINI_API_KEY=your_key
OLLAMA_HOST=localhost:11434
```

## Failover Chain

Default model failover order:
1. `gemini:models/gemini-3.1-pro-preview`
2. `gemini:models/gemini-3-flash-preview`
3. `gemini:models/gemini-pro-latest`
4. `gemini:models/gemini-flash-latest`
5. `ollama:gpt-oss:120b`

## Integration with Makefile

```makefile
runreport:
	source $(DB_ENV)/bin/activate && cd db && ./runreport.py hybrid_batch.tsv
```

The `runreport` target processes a batch file with format:
```
startdate,filename,entity,query
```

Example `hybrid_batch.tsv`:
```
-7,iran_news,Iran,Iran developments
-14,ukraine_war,Ukraine,Ukraine conflict
-3,climate,Climate,Climate policy
```