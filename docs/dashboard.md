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

### Media Coverup Detection

Interactive tool to identify subjects with extreme bias coverage indicating potential media coverups or suppression.

#### Features

- **Filter Panel (Sidebar)**
  - Entity Type: Multi-select (GPE, PERSON, ORG)
  - Direction: Radio (Left, Right, Both)
  - Imbalance Threshold: Slider (0.0 - 1.0, default 0.40)
  - Min Articles: Number input (default 50)
  - Date Range: Select (30d, 90d, 1y, All time)

- **Interactive Visualizations (Plotly)**
  - Sortable data table with click-to-drill-down
  - Bar chart for coverage gaps (clickable for source breakdown)
  - Line chart for temporal trends (hover details, multi-select)
  - Treemap for source concentration (click to drill down)

- **Analysis Types**
  - Extreme Bias Imbalance: Subjects where |R-L| > 0.40
  - Coverage Gaps: Topics with < 100 articles (potential suppression)
  - Source Concentration: Topics with >70% from top 3 sources

- **Refresh Button**
  - Large button with "Last refreshed: X seconds ago" counter
  - Click to re-run analysis with current filters

#### Launch

```bash
make dashboard
# Or: cd dashboard && source .venv/bin/activate && streamlit run app.py
```

Navigate to the "Media Coverup Detection" tab (or see sidebar).

#### CLI Alternative

For command-line analysis:
```bash
python scripts/find_media_coverups.py --output interactive
python scripts/find_media_coverups.py --output csv,json
```

See [docs/media_coverups.md](media_coverups.md) for full documentation.

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