# Utility Scripts — Reference Guide

Miscellaneous utilities for data processing and pipeline operations.

## ID Extraction & Processing

### extract_ids.py

Extract MongoDB IDs from articles.

```bash
python db/extract_ids.py [options]
```

Options:
- `--start-date DATE` — Start date filter
- `--end-date DATE` — End date filter
- `--output FILE` — Output file (default: ids.txt)
- `--source SOURCE` — Filter by source

### dump_ids.py

Dump IDs to file for batch processing.

```bash
python db/dump_ids.py --ids 678f3a2b...,679abc12... --output batch.txt
```

---

## Date & Timestamp Utilities

### mktimestamp.py

Generate timestamp file for pipeline.

```bash
python db/mktimestamp.py [days_ago]
```

Output: writes ISO date to `db/timestamp.txt`

### test_timeout.py

Test MongoDB query timeouts.

```bash
python db/test_timeout.py --limit 1000
```

---

## Title & Text Processing

### titlesort.py

Sort articles by title.

```bash
python db/titlesort.py --input articles.json --output sorted.json
```

### clean_install.py

Clean and reinstall model cache.

```bash
python semantic/clean_install.py
```

---

## MongoDB Utilities

### mongo_pager.py

Paginate through MongoDB cursor.

```python
from mongo_pager import paginate, batch_fetch

# Paginate cursor
for batch in paginate(cursor, batch_size=1000):
    process(batch)

# Batch fetch
docs = batch_fetch(collection, ids, batch_size=100)
```

### mongo_rw.py

Read/write utilities.

```python
from mongo_rw import read_article, write_article, upsert_article
```

---

## Data Conversion

### convert_to_vec.py

Legacy vector conversion (replaced by mongo2chroma.py).

### dot2png.py

Graphviz DOT to PNG conversion.

```bash
python db/dot2png.py graph.dot output.png
```

### trans_cos.py

Translation cosine similarity.

```bash
python semantic/trans_cos.py text1 text2
```

---

## Report Generation

### runhybrid.py

Run hybrid search with report output.

```bash
python db/runhybrid.py "climate" --output report.txt
```

### runreport.py

Generate news report.

```bash
python db/runreport.py -7 climate_news "Climate developments" Climate
```

---

## Graph & Relationships

### cypher_to_graph.py

Extract SVO triples from articles with LLM.

```bash
python db/cypher_to_graph.py --input articles.json --output graph.json
```

### distribute.py

Distribute tasks across workers.

```bash
python db/distribute.py --workers 4 --input data.json
```

---

## API Clients

### groqai.py

Groq API integration.

```python
from groqai import GroqAI

client = GroqAI(model="mixtral-8x7b-32768")
response = client.chat("Your question")
```

### gemini.py

Gemini API integration.

```python
from gemini import GeminiAI

client = GeminiAI()
response = client.generate("Your prompt")
```

---

## Logging & Metrics

### llm_analysis.py

LLM output analysis.

```bash
python db/llm_analysis.py --input results.json
```

---

## Data Format Examples

### Article IDs (line-separated)

```
678f3a2b1c4d4567890abcdef01
679abc1234d567890abcdef02
67a0bcd1234e567890abcdef03
```

### JSONL Input

```json
{"id": "1", "text": "Article text..."}
{"id": "2", "text": "Another article..."}
```

---

## Quick Reference

| Need... | Use |
|--------|-----|
| Extract IDs | `db/extract_ids.py` |
| Generate timestamp | `db/mktimestamp.py` |
| Convert vectors | `db/mongo2chroma.py load` |
| Test timeout | `db/test_timeout.py` |
| Run report | `db/runreport.py` |