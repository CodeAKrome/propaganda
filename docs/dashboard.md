# Dashboard — Streamlit Data Visualization

Real-time dashboard for monitoring RSS news data, bias analysis, and training telemetry.

## Quick Start

```bash
make dashboard
```

Or manually:

```bash
cd dashboard
source .venv/bin/activate
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Features

### Data Overview
- **Total Records** — Count of all articles in MongoDB
- **Records by Source** — Bar chart of articles per RSS feed
- **Records Over Time** — Line chart of article volume by date
- **Source × Date Heatmap** — Cross-reference sources and dates

### Bias Analysis

#### Overall Bias Distribution
- Pie chart showing Left / Center / Right bias proportions
- Bar chart showing Low / Medium / High bias degree
- Overall assessment badge (e.g., 🔵 LEFT BIASED)

#### Per-Source Bias
- Filter by specific news source
- Time-series of bias direction over time
- Heatmap of bias by source and date

#### Bias Details Table
- Sortable table with source, title, direction, degree
- Click to view full article text

### Training Telemetry

- Training run history with timestamps
- Loss curves per epoch/step
- Model performance metrics
- Device used (MPS/CUDA/CPU)

---

## Screenshots

### Main Dashboard View

```
┌─────────────────────────────────────────────────────────────────────┐
│  📰 RSS News Dashboard                                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ 📊 Total Records     │  │ 🎯 Overall Bias       │             │
│  │ 42,531               │  │ Analyzed: 15,234      │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                     │
│  Overall Bias Distribution        Bias Degree (Strength)          │
│  ┌────────────────────────┐      ┌────────────────────────┐          │
│  │    ████   ████████    │      │ Low ████████ 0.32     │          │
│  │   █████ ███████████   │      │ Med ██████████ 0.48  │          │
│  │  ████████████████████  │      │ High███████████ 0.20 │          │
│  └────────────────────────┘      └────────────────────────┘          │
│                                                                     │
│  🔵 LEFT BIASED (threshold ≥0.5)                                     │
│                                                                     │
│  📋 Sample Bias Records (Raw)                                       │
│  ┌──────────┬──────────────────────────┬─────────┬────────┐        │
│  │ Source   │ Title                   │ Dir     │ Deg    │        │
│  ├──────────┼──────────────────────────┼─────────┼────────┤        │
│  │ CNN      │ Breaking: ...           │ Left    │ High   │        │
│  │ Fox News │ Exclusive: ...          │ Right   │ Medium │        │
│  └──────────┴──────────────────────────┴─────────┴────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### Records by Source

```
┌─────────────────────────────────────────────────────────────────────┐
│  📊 Records by Source                                               │
│                                                                     │
│  ████████████████████████████████ 28,431  CNN                       │
│  ████████████████████████        22,156  BBC                      │
│  ██████████████████               18,234  Reuters                  │
│  ████████████████                 12,451  Fox News                 │
│  ████████████                      8,923  MSNBC                   │
│  ...                                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Records Over Time

```
┌─────────────────────────────────────────────────────────────────────┐
│  📈 Records Over Time                                               │
│                                                                     │
│      800 │       ╭─╮           ╭───╮                                │
│      600 │     ╭─╯ ╰─╮       ╭─╯   ╰─╮     ╭─╮                      │
│      400 │   ╭─╯       ╰───╮╭╯       ╰─────╯ ╰─╮                   │
│      200 │───╯                                    ───               │
│        0 └─────────────────────────────────────────────────         │
│          2025-01   2025-02   2025-03   2025-04   2025-05          │
└─────────────────────────────────────────────────────────────────────┘
```

### Source × Date Heatmap

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔥 Source × Date Heatmap                                           │
│                                                                     │
│         Jan  Jan  Feb  Feb  Mar  Mar  Apr  Apr                     │
│          15   28   12   26   08   22   05   19                     │
│  CNN     ████ ████ ████ ████ ████ ████ ████ ████                    │
│  BBC     ████ ████ ████ ████ ████ ████ ████ ████                    │
│  Fox     ████ ████ ████ ████ ████ ████ ████ ████                    │
│  MSNBC   ████ ████ ████ ████ ████ ████ ████ ████                    │
│                                                                     │
│  Legend: Darker = More articles                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Per-Source Bias Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│  📊 Per-Source Bias                                                 │
│                                                                     │
│  Select Source: [All Sources ▼]                                    │
│                                                                     │
│  ┌──────────────────────────────────────┐                          │
│  │ Source    │ Avg Left │ Avg Center │ Avg Right │                 │
│  ├──────────┼──────────┼────────────┼───────────┤                  │
│  │ CNN      │   0.45   │   0.35    │   0.20   │                  │
│  │ Fox News │   0.15   │   0.25    │   0.60   │                  │
│  │ BBC      │   0.30   │   0.45    │   0.25   │                  │
│  │ Reuters  │   0.25   │   0.50    │   0.25   │                  │
│  └──────────────────────────────────────┘                          │
│                                                                     │
│  Bias Direction Over Time (Line Chart)                             │
│  ─────────────────────────────────────────                          │
│      0.8 │                            ╭─── Right                    │
│      0.6 │                   ╭────────╯                             │
│      0.4 │         ╭────────╯        Center                        │
│      0.2 │────────╯                  Left                           │
│        0 └────────────────────────────────────                      │
│            Jan    Feb    Mar    Apr    May                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Bias Details Table

```
┌─────────────────────────────────────────────────────────────────────┐
│  📋 Bias Details                                                    │
│                                                                     │
│  Search: [________________]  Filter: [Direction ▼] [Degree ▼]       │
│                                                                     │
│  ┌──────────┬─────────────────────────────┬─────────┬────────┐    │
│  │ Source   │ Title                        │ Dir     │ Deg    │    │
│  ├──────────┼─────────────────────────────┼─────────┼────────┤    │
│  │ CNN      │ Breaking: Biden announces... │ Left    │ Medium │    │
│  │ Fox      │ Exclusive: Hunter Biden...    │ Right   │ High   │    │
│  │ BBC      │ World: Global markets...     │ Center  │ Low    │    │
│  │ MSNBC    │ Analysis: Economic policy... │ Left    │ High   │    │
│  │ Reuters │ Update: Fed interest rates   │ Center  │ Medium │    │
│  └──────────┴─────────────────────────────┴─────────┴────────┘    │
│                                                                     │
│  Showing 1-10 of 15,234  [< Prev] [1] [2] ... [1524] [Next >]     │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Telemetry

```
┌─────────────────────────────────────────────────────────────────────┐
│  📡 Training Telemetry                                              │
│                                                                     │
│  ┌──────────────────────────────────────┐                          │
│  │ Run ID      │ Date       │ Status    │                         │
│  ├─────────────┼────────────┼───────────┤                          │
│  │ lora_llama  │ 2025-04-09 │ Complete  │                         │
│  │ t5_v2       │ 2025-04-08 │ Complete  │                         │
│  │ t5_v1       │ 2025-04-07 │ Failed    │                         │
│  └─────────────┴────────────┴───────────┘                          │
│                                                                     │
│  Selected Run: lora_llama                                           │
│                                                                     │
│  Loss Curve                                                         │
│  ─────────                                                         │
│      2.5 │                                                    ╲     │
│        2 │                                                 ╱  ╲    │
│      1.5 │                                              ╱     ╲   │
│        1 │                                           ╱        ╲   │
│      0.5 │─────────────────────────────────────────           │
│        0 └────────────────────────────────────────────────       │
│            0    100   200   300   400   500                     │
│                           Steps                                    │
│                                                                     │
│  Metrics: Epochs: 3 | Final Loss: 0.42 | Device: MPS              │
└─────────────────────────────────────────────────────────────────────┘
```

### Sidebar Options

```
┌──────────────┐
│ Options      │
│ [Refresh]    │
│──────────────│
│ Connected to:│
│ mongodb://   │
│ root:****@   │
│ localhost:   │
│ 27017        │
│              │
│ Database:    │
│ rssnews      │
│              │
│ Collection: │
│ articles     │
└──────────────┘
```

---

## Configuration

### Environment Variables

```bash
MONGO_URI=mongodb://root:example@localhost:27017
MONGO_DB=rssnews
MONGO_COLL=articles
```

### Data Refresh

- Data cached for 300 seconds (5 minutes)
- Click "Refresh Data" in sidebar to reload

### Dependencies

```
streamlit
pandas
pymongo
altair
```

Install via:

```bash
cd dashboard
pip install -r requirements.txt
```

---

## Troubleshooting

### "Connection Refused"
- Ensure MongoDB is running: `mongod`
- Check `MONGO_URI` in environment

### "No Data Displayed"
- Verify articles exist: `mongosh --eval "db.articles.countDocuments({})"`
- Check date filters if using date range

### Slow Performance
- Data is cached for 5 minutes
- Use sidebar "Refresh Data" button
- Reduce date range for faster queries