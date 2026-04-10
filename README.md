<img width="1728" height="960" alt="Image" src="https://github.com/user-attachments/assets/e13fccab-e70c-4b80-b4ca-395e5737ccf0" />

# Propaganda — News Analysis Pipeline

A complete local-first news aggregation, analysis, and reporting pipeline.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  RSS Feeds  │───▶│  MongoDB    │───▶│  NER        │
│  (main.go)  │    │  (Articles) │    │  (Flair)    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                   ┌──────────────────────────┘
                   ▼
              ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
              │  ChromaDB   │◀──▶│  Hybrid     │───▶│  Report     │
              │  (Vectors)  │    │  Search     │    │  Generation │
              └─────────────┘    └─────────────┘    └─────────────┘
                                              │
                   ┌──────────────────────────┘
                   ▼
              ┌─────────────┐    ┌─────────────┐
              │  T5 Bias    │    │  Video Gen  │
              │  Detection  │    │  (MGM)      │
              └─────────────┘    └─────────────┘
```

## Quick Start

```bash
# 1. Load RSS feeds
go run main.go config/big.tsv config/kill.tsv

# 2. Run NER
python ner/main.py

# 3. Generate vectors
python db/mongo2chroma.py load --limit 100

# 4. Search articles
python db/hybrid.py "climate change" -n 10

# 5. Generate report
python db/report.py -7 climate_news "Climate developments" Climate

# 6. Generate video
python mgm/mgm.py article.txt output.mp4
```

Or use the Makefile:

```bash
make testrun    # Full pipeline
make smallthingsthatgo  # Quick test
```

## Components

### Data Ingestion

| File | Description |
|------|-------------|
| [main.go](docs/main_go.md) | RSS feed aggregator with user-agent rotation |
| [Makefile](Makefile) | Pipeline orchestration |

### Database & Search

| File | Description |
|------|-------------|
| [mongo2chroma.py](docs/mongo2chroma.md) | MongoDB → ChromaDB vector loader |
| [hybrid.py](docs/hybrid.md) | Hybrid vector + BM25 search |
| [geminize.py](docs/geminize.md) | LLM processing pipeline |
| [report.py](docs/report.md) | News report generation |

### AI Services

| File | Description |
|------|-------------|
| [ner/main.py](ner/main.py) | Named Entity Recognition (Flair) |
| [ollamaai.py](docs/ollamaai.md) | Ollama LLM client |
| [mgm/mgm.py](docs/mgm.md) | Video generation (SD Turbo + Kokoro) |

### Bias Detection

| File | Description |
|------|-------------|
| [t5/bias_detector/](t5/bias_detector/) | T5+LoRA bias detection |
| [llm/bias_processor.py](llm/bias_processor.py) | LLM-based bias processing |

## Documentation

- [mongo2chroma.md](docs/mongo2chroma.md) — Vector loading & search
- [hybrid.md](docs/hybrid.md) — Hybrid search with BM25
- [geminize.md](docs/geminize.md) — LLM article processing
- [report.md](docs/report.md) — News report generation
- [ollamaai.md](docs/ollamaai.md) — Ollama CLI client
- [main_go.md](docs/main_go.md) — RSS feed aggregator
- [mgm.md](docs/mgm.md) — Video generator

## Environment Variables

```bash
MONGO_URI=mongodb://user:pass@host:27017
MONGO_USER=root
MONGO_PASS=example
GEMINI_API_KEY=your_key
OLLAMA_HOST=localhost:11434
```

## Directory Structure

```
propaganda/
├── main.go           # RSS aggregator
├── Makefile          # Pipeline tasks
├── config/           # Feed configs
├── db/               # Database scripts
│   ├── mongo2chroma.py
│   ├── hybrid.py
│   ├── geminize.py
│   └── report.py
├── ner/              # Named Entity Recognition
├── llm/              # LLM processing
├── t5/               # T5 bias detection
├── mgm/              # Video generation
├── front/            # React web UI
├── back/             # Express API
├── dashboard/        # Streamlit dashboard
└── docs/             # Documentation
```

## Frontend

```bash
cd front && npm install && npm start
```

API server:
```bash
cd back && node server.js
```

## Dashboard

```bash
cd dashboard && streamlit run app.py
```

## License

MIT — 100% local, no API keys required (except Gemini optional).