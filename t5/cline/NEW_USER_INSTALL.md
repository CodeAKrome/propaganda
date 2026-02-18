# Bias Detector - Out-of-the-Box Experience

## One Command. Done.

```bash
pip install bias-mcp-server
bias "The president announced new policies today."
```

**Output:**
```
==================================================
BIAS ANALYSIS
==================================================

Direction: Right-leaning (50%)
  Left: 20%  Center: 30%  Right: 50%

Intensity: High (50%)
  Low: 20%  Medium: 30%  High: 50%

Reasoning: The article uses loaded language...
==================================================
```

---

## All Usage Options

### Command Line

```bash
# Analyze text directly
bias "Your text here"

# Analyze a file
bias --file article.txt

# Pipe text
echo "Your text" | bias

# JSON output
bias --json "text"

# Minimal output (just L, C, or R)
bias --quiet "text"

# Show model info
bias --info
```

### Python API

```python
from mcp_bias_server import quick_analyze

# One-line analysis
result = quick_analyze("The president announced new policies.")
print(result['dominant_direction'])  # "Left", "Center", or "Right"
print(result['dominant_degree'])     # "Low", "Medium", or "High"
print(result['direction_percent'])  # {"L": 20.0, "C": 30.0, "R": 50.0}
```

### Analyze a File

```python
from mcp_bias_server import analyze_file

result = analyze_file("article.txt")
print(result['reason'])
```

---

## How It Works

1. **First run:** Model auto-downloads from HuggingFace (~18MB)
2. **Subsequent runs:** Uses cached model (instant startup)
3. **No configuration needed** - just install and use

---

## Requirements

- Python 3.10+
- ~500MB disk space (for model + dependencies)
- Internet connection (first run only)

---

## Advanced Options

### Use Local Model Only

```bash
bias --no-huggingface "text"
```

```python
result = quick_analyze("text", use_huggingface=False)
```

### Set Custom Model Path

```bash
export BIAS_MODEL_PATH=/path/to/your/model
bias "text"
```

---

## Troubleshooting

**"Model not found" error:**
- Ensure internet connection for first run (HuggingFace download)
- Or set `BIAS_MODEL_PATH` to local model directory

**Slow first run:**
- Normal - model downloads from HuggingFace (~18MB)
- Subsequent runs are fast

**Memory error:**
- Close other applications
- Model requires ~2GB RAM
