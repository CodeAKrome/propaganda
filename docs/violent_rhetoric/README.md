# Violent Rhetoric Analysis by Political Orientation

Comprehensive analysis comparing violent rhetoric prevalence in left vs right news media, using MongoDB article database.

## Quick Start

```bash
cd /Users/kyle/hub/propaganda/docs/violent_rhetoric

# Run analysis
python3 analysis/mongo_analysis.py

# Generate visualizations
python3 visualizations/generate_charts.py
```

## Directory Structure

```
docs/violent_rhetoric/
├── README.md                           # This file
├── analysis/
│   ├── mongo_analysis.py               # MongoDB violent content extraction
│   └── (web research scripts)
├── data/
│   ├── mongo_results.json              # MongoDB query results
│   └── web_search.json                 # Web research data
├── visualizations/
│   ├── coverage.svg                    # Violent % by orientation
│   ├── terms.svg                       # Top violent terms
│   └── comparison.svg                  # Summary table
└── output/
    └── violent_rhetoric.vec            # All violent articles
```

## Key Findings

### Statistics by Political Orientation

| Orientation | Total Articles | Violent Articles | Violent % |
|-------------|---------------|-----------------|-----------|
| LEFT        | 570           | 39              | 6.84%     |
| CENTER      | 67,404        | 15,410          | 22.86%    |
| RIGHT       | 16,349        | 4,551           | 27.84%    |

### Top Violent Terms by Orientation

**LEFT:**
- attack (10), killed (8), torture (5), killing (5), blood (4)

**CENTER:**
- war (4,388), attack (2,873), killed (2,674), killing (1,193), fight (1,082)

**RIGHT:**
- war (1,096), attack (857), killed (675), fight (507), shot (423)

### Top Targets

**LEFT targets:** Chinese (23), Iranian (10), Donald Trump (8), Israeli (6)

**CENTER targets:** Trump (4,207), Donald Trump (3,131), Russian (2,697), Ukrainian (1,438)

**RIGHT targets:** Trump (1,522), Donald Trump (1,006), American (640), British (563)

## Methodology

### Data Sources
- MongoDB: 236,393 total articles (84,323 with bias scores)
- Sources classified as Left/Center/Right based on bias analysis

### Violent Keywords
- Explicit: kill, murder, massacre, shoot, attack, bomb, torture, brutality
- Implicit: enemy, vermin, invasion, existential threat, eradicate

### Source Classification
Based on bias score analysis:
- **Left**: et-chinahuman, voa-intnl, et-la, et-me, voa-africa
- **Center**: ABC, BBC, CNBC, DW, Reuters, most international wires
- **Right**: Breitbart, Fox, GB News, arutz-sheva, NY Post

## Visualizations

Generated following Tufte principles:
- **coverage.svg**: Bar chart of violent % by political orientation
- **terms.svg**: Top violent terms comparison
- **comparison.svg**: Summary statistics table

## Notes

- Left sources have significantly less coverage in dataset (570 vs 16,349 right)
- "War" is the most common term across all orientations
- Center/Right show ~3-4x higher violent content percentage than Left
- Results may be affected by source distribution in dataset