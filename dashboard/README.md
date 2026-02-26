# RSS News Dashboard

A Streamlit dashboard for visualizing MongoDB RSS news data.

## Requirements

```bash
pip install -r requirements.txt
```

Or use the venv in the parent directory:
```bash
cd ..
source db/.venv/bin/activate
pip install streamlit pandas pymongo
```

## Running

```bash
streamlit run app.py
```

Or with the parent venv:
```bash
cd ..
source db/.venv/bin/activate
streamlit run dashboard/app.py
```

## Features

- **Total Records**: Shows the total count of articles in MongoDB
- **Records by Source**: Bar chart and table of article counts per source
- **Records Over Time**: Bar chart showing article counts over time with date range
- **Records by Source Over Time**: Line chart comparing multiple sources over time with multi-select

## Configuration

The dashboard uses the same MongoDB connection settings as `mongo2chroma.py`:
- `MONGO_URI`: MongoDB connection string (default: `mongodb://root:example@localhost:27017`)
- Database: `rssnews`
- Collection: `articles`

Override with environment variable:
```bash
MONGO_URI="mongodb://user:pass@host:27017" streamlit run app.py
```
