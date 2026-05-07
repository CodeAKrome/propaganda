# Media Coverup Detection

Analyze MongoDB bias data to identify subjects with extreme coverage bias indicating potential media coverups or suppression.

## Quick Start

### Interactive Mode (Recommended)

```bash
python scripts/find_media_coverups.py --output interactive
```

### CLI with CSV/JSON Output

```bash
# Output to CSV and JSON
python scripts/find_media_coverups.py --output csv,json

# Output to console only
python scripts/find_media_coverups.py --output console
```

### Using Makefile

```bash
make analyze-bias-coverage
# Equivalent to: python scripts/find_media_coverups.py --output interactive
```

---

## CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--min-articles` | `-n` | Minimum articles for subject to be analyzed | 50 |
| `--imbalance-threshold` | `-i` | Bias imbalance score above which = extreme (0.0-1.0) | 0.40 |
| `--coverage-gap-threshold` | `-c` | Articles below this = coverage gap | 100 |
| `--entity-type` | `-e` | Entity type: GPE, PERSON, ORG, or ALL | ALL |
| `--output` | `-o` | Output format: console, csv, json, interactive | interactive |
| `--output-csv` | | Path for CSV output | output/media_coverups.csv |
| `--output-json` | | Path for JSON output | output/media_coverups.json |

---

## Analysis Types

### 1. Extreme Bias Imbalance

Identifies subjects where coverage is predominantly from one political direction (>80% Left or Right). This may indicate coordinated narrative pushing or suppression of alternative viewpoints.

**Example Output:**
```
=== EXTREME BIAS IMBALANCE (Potential Coverup) ===
Subject                   | Type   | Count | L    | C    | R    | Degree | Imbalance | Direction
-----------------------------------------------------------------------------------------
Kharkov Region            | GPE    |    85 | 0.00 | 0.15 | 0.84 | 0.72   | 0.84      | RIGHT ⚠️
Donetsk People's Republic | GPE    |   125 | 0.03 | 0.17 | 0.80 | 0.65   | 0.77      | RIGHT ⚠️
Zaporozhye Region         | GPE    |   105 | 0.02 | 0.20 | 0.78 | 0.68   | 0.76      | RIGHT ⚠️
Schweizers                | PERSON |   183 | 0.06 | 0.12 | 0.82 | 0.76   | 0.76      | RIGHT ⚠️
Guan                      | PERSON |   107 | 0.68 | 0.25 | 0.07 | 0.30   | 0.61      | LEFT ⚠️
```

**Interpretation:**
- Imbalance score = |R - L| (0 = perfectly balanced, 1 = 100% one direction)
- Degree = High (H) bias means strong editorializing, not just mild framing
- ⚠️ indicates subjects requiring further investigation

### 2. Coverage Gaps

Identifies topics with minimal or zero coverage. Zero coverage of important topics may indicate intentional suppression.

**Built-in Sensitive Topics Checked:**
| Topic | Keywords | Signal |
|-------|----------|--------|
| Xinjiang/Uyghur | Uighur, Xinjiang, Uyghur | Low coverage |
| Falun Gong | Falun Gong, Shen Yun | Limited coverage |
| Tiananmen Square | Tiananmen, 1989 | Near-zero coverage |
| Nigerian Christians | Nigeria Christian persecution | ZERO COVERAGE |
| North Korea Prison | Gulag, labor camp | ZERO COVERAGE |
| Hong Kong Democracy | Hong Kong protests | Near-zero coverage |

**Example Output:**
```
=== COVERAGE GAPS (Potential Suppression) ===
Topic                          | Articles | Sources | Signal
-----------------------------------------------------------------
Nigerian Christian Persecution |        0 |       0 | 🚨 ZERO COVERAGE
North Korea Prison Camps       |        0 |       0 | 🚨 ZERO COVERAGE
Xinjiang/Uyghur Genocide       |        9 |       3 | ⚠️ LOW COVERAGE
Tiananmen Square               |       52 |       8 | ⚠️ LIMITED
Falun Gong / Shen Yun          |       62 |       8 | ⚠️ LIMITED
```

### 3. Source Concentration

Identifies topics covered by very few sources. High concentration (>70% from top 3 sources) may indicate gatekeeping or limited perspective.

**Example Output:**
```
=== SOURCE CONCENTRATION ===
Topic               | Articles | Top 3 %  | Signal
-------------------------------------------------
Xinjiang            |        9 |    78%  | ⚠️ HIGH CONCENTRATION
Falun Gong          |       62 |    65%  | ⚠️ MODERATE CONCENTRATION
Taiwan             |     1379 |    41%  | ✓ DIVERSE COVERAGE
```

### 4. Temporal Trends

Tracks how coverage of specific topics changes over time. Declining coverage of previously-covered topics may indicate shifting narrative priorities.

**Example Output:**
```
=== TEMPORAL ANALYSIS: Coverage Trends ===
Subject                    | 30-Day Trend | 90-Day Trend
-------------------------------------------------------------
Ukraine War Coverage       | 📈 +15%      | 📈 +42%
Taiwan Strait              | 📉 -8%       | 📈 +5%
Hong Kong Democracy        | 📉 -65%      | 📉 -80%
```

---

## Examples

### Example 1: Find All Extreme Bias Topics (GPE Only)

```bash
python scripts/find_media_coverups.py -e GPE --min-articles 30
```

**Output:**
```
=== EXTREME BIAS IMBALANCE ===
Subject                   | Type | Count | L    | C    | R    | Imbalance | Direction
-----------------------------------------------------------------------------------------
Kharkov Region            | GPE  |    85 | 0.00 | 0.15 | 0.84 | 0.84      | RIGHT ⚠️
Donetsk People's Republic | GPE  |   125 | 0.03 | 0.17 | 0.80 | 0.77      | RIGHT ⚠️
...
```

### Example 2: Output to CSV for Further Analysis

```bash
python scripts/find_media_coverups.py \
  --output csv \
  --output-csv output/bias_analysis.csv \
  --entity-type ALL \
  --min-articles 50 \
  --imbalance-threshold 0.40
```

**CSV Output (output/bias_analysis.csv):**
```csv
Subject,EntityType,ArticleCount,AvgL,AvgC,AvgR,AvgDegree,ImbalanceScore,Direction,CoverageGap,SourceConcentration
Kharkov Region,GPE,85,0.00,0.15,0.84,0.72,0.84,RIGHT,NONE,LOW
Donetsk People's Republic,GPE,125,0.03,0.17,0.80,0.65,0.77,RIGHT,NONE,LOW
...
```

### Example 3: Strict Threshold (Higher Sensitivity)

```bash
python scripts/find_media_coverups.py -i 0.60 --min-articles 100 -e PERSON
```

This finds only VERY extreme bias (imbalance > 0.60) with at least 100 articles for PERSON entities.

### Example 4: Interactive Rich CLI

```bash
python scripts/find_media_coverups.py --output interactive
```

Opens an interactive menu using Rich library:
- Select analysis type (bias imbalance, coverage gaps, temporal, source concentration)
- Filter by entity type and direction
- Drill down into specific subjects
- Adjust thresholds in real-time

---

## Bias Data Structure

The analysis works with MongoDB articles that have the following bias field:

```json
{
  "_id": "...",
  "title": "Article Title",
  "source": "news-source",
  "bias": {
    "dir": {"L": 0.3, "C": 0.5, "R": 0.2},
    "deg": {"L": 0.1, "M": 0.3, "H": 0.6},
    "reason": "Analysis explanation..."
  },
  "ner": {
    "entities": [
      {"label": "GPE", "text": "Ukraine"},
      {"label": "PERSON", "text": "Zelensky"}
    ]
  }
}
```

**Field Meanings:**
- `dir.L/C/R`: Probability of Left/Center/Right political leaning
- `deg.L/M/H`: Degree (intensity) of Low/Medium/High bias
- `ner.entities`: Named entities extracted from article

---

## Integration with Dashboard

Access via Streamlit dashboard at http://localhost:8501

### Dashboard Features

1. **Filter Sidebar**
   - Entity Type: Multi-select (GPE, PERSON, ORG)
   - Direction: Radio (Left, Right, Both)
   - Imbalance Threshold: Slider (0.0 - 1.0, default 0.40)
   - Min Articles: Number input (default 50)
   - Date Range: Select (30d, 90d, 1y, All time)

2. **Interactive Charts**
   - Sortable data table with click-to-drill-down
   - Bar chart for coverage gaps
   - Line chart for temporal trends
   - Treemap for source concentration

3. **Refresh Button**
   - Large button with "Last refreshed: X seconds ago" counter
   - Click to re-run analysis

### Launch Dashboard

```bash
make dashboard
# Or: cd dashboard && source .venv/bin/activate && streamlit run app.py
```

Navigate to the "Media Coverup Detection" tab to view interactive analysis.

---

## Algorithm

### Bias Imbalance Calculation

```
For each subject (entity):
  1. Get all articles mentioning the subject
  2. Calculate average L, C, R bias direction
  3. Calculate average degree (especially H = High)
  4. ImbalanceScore = |AvgR - AvgL|
  5. Severity = ImbalanceScore * Degree

  Flag as EXTREME if:
    - ImbalanceScore > threshold (default 0.40)
    - Article count >= min_articles (default 50)
```

### Coverage Gap Detection

```
For each sensitive topic:
  1. Search articles containing topic keywords
  2. Count articles with bias data
  3. Identify sources covering the topic

  Flag as COVERAGE GAP if:
    - Article count < coverage_gap_threshold (default 100)
    - Or: ZERO articles (complete suppression)
```

---

## Troubleshooting

### No Bias Data Found

**Error:** "No articles with bias data found"

**Solution:** Ensure bias detection has been run:
```bash
make bias
# Or: source db/.venv/bin/activate && cd db && ./runbias.sh
```

### MongoDB Connection Error

**Error:** "Connection refused" or "Authentication failed"

**Solution:** Check MONGO_URI in environment:
```bash
echo $MONGO_URI
# Should be: mongodb://root:example@localhost:27017
```

### Empty Results

**Error:** "No extreme bias subjects found"

**Solution:** Lower the thresholds:
```bash
python scripts/find_media_coverups.py -i 0.30 -n 20
```

---

## Files

| File | Description |
|------|-------------|
| `scripts/find_media_coverups.py` | Main analysis script |
| `output/media_coverups.csv` | Generated CSV output |
| `output/media_coverups.json` | Generated JSON output |
| `docs/media_coverups.md` | This documentation |