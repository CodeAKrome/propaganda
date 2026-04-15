# TTS & Video Generation

Text-to-speech and video generation utilities.

## Components

| File | Description |
|------|-------------|
| `mp3/ttskokoro.py` | Kokoro TTS |
| `mp3/text2mp3.py` | Simple text-to-MP3 |
| `mp3/jsonl2mp3.py` | JSONL batch to MP3 |
| `mgm/mgm.py` | Video generation (documented in docs/mgm.md) |

---

## ttskokoro.py

Kokoro TTS — high-quality neural TTS for news narration.

### Usage

```bash
python mp3/ttskokoro.py <input> <output.mp3>
```

### Arguments

| Position | Description | Example |
|----------|-------------|---------|
| `input` | Text file or `-` for stdin | `script.txt` or `-` |
| `output` | Output MP3 file | `output.mp3` |

### Examples

```bash
# From file
python mp3/ttskokoro.py script.txt narration.mp3

# From stdin
echo "Breaking news report" | python mp3/ttskokoro.py - narration.mp3
```

### Features

- **Voice**: `af_heart` (default female voice)
- **Sample Rate**: 24kHz
- **Language**: English (`a` code)
- **Hardware**: Metal MPS / CUDA / CPU

### Text Cleaning

Removes special characters, keeps:
- Alphanumeric
- Punctuation: `. , ! ? ; : ' " ( ) -`
- Whitespace

---

## text2mp3.py

Simplified text-to-MP3 wrapper.

### Usage

```bash
python mp3/text2mp3.py [text] <output.mp3>
```

### Arguments

| Position | Description | Example |
|----------|-------------|---------|
| `text` | Text to speak | "Hello world" |
| `output` | Output MP3 file | `output.mp3` |

### Examples

```bash
python mp3/text2mp3.py "Breaking news" output.mp3
```

---

## jsonl2mp3.py

Batch process JSONL file to MP3 narrations.

### Usage

```bash
python mp3/jsonl2mp3.py <input.jsonl> <output_dir>
```

### Input Format (JSONL)

```json
{"id": "1", "text": "First article text..."}
{"id": "2", "text": "Second article text..."}
```

### Output

Creates `output_dir/{id}.mp3` files.

### Examples

```bash
python mp3/jsonl2mp3.py articles.jsonl output/
```

---

## mkscript.py

Generate narration script from article text.

### Usage

```bash
python mgm/mkscript.py <article.txt> <output.md>
```

### Examples

```bash
python mgm/mkscript.py article.txt script.md
```

### Script Format

```markdown
# Breaking News

[Music: upbeat intro]

## Main Story

President Biden announced new climate legislation today...

## Key Details

- Carbon emissions target: 40% by 2030
- Timeline: 10 years

[Music: fade out]
```

---

## Python API

```python
from kokoro import KPipeline
import torch

# Initialize
pipeline = KPipeline(lang_code='a')

# Generate
text = "Breaking news report"
generator = pipeline(text, voice='af_heart')

# Collect audio chunks
audio_chunks = []
for gs, ps, audio in generator:
    audio_chunks.append(audio)

# Concatenate
full_audio = torch.cat(audio_chunks)

# Save
import soundfile as sf
sf.write("output.wav", full_audio, 24000)
```

---

## Voice Options

Kokoro voices available:

| Voice Code | Description |
|-----------|-------------|
| `af_heart` | Female, warm (default) |
| `af_sarah` | Female, professional |
| `am_michael` | Male, conversational |
| `am_peter` | Male, formal |

---

## Dependencies

```bash
pip install kokoro-mlx soundfile torch
```

## Makefile Targets

```makefile
# Generate TTS
mp3small:
	python mgm/mgm.py $(TITLEFILE) output/$(NAMESPACE).mp4

# Direct TTS
tts:
	python mp3/ttskokoro.py script.txt output.mp3
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (no audio generated) |

## Troubleshooting

### "kokoro module not found"

```bash
pip install kokoro-mlx
```

### "No audio generated"

- Text may be empty or too short
- Check text cleaning didn't remove all content