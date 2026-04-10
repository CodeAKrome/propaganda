# mgm.py

Enhanced News Article to Video Generator — 100% local, free, GPU-accelerated.

## Overview

Creates narrated news video clips with AI-generated backgrounds using Stable Diffusion Turbo and TTS via Kokoro (Metal/CUDA). Supports entity-aware backgrounds, procedural portraits, and professional news overlays.

## Usage

```bash
python mgm/mgm.py <input> <output> [options]
```

## Arguments

### Positional

| Argument | Description | Example |
|----------|-------------|---------|
| `input` | Article text file, `-` for stdin, or `--story` for markdown | `article.txt` |
| `output` | Output .mp4 filename | `news.mp4` |

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--data FILE` | Entities metadata file | `--data entities.txt` |
| `--report FILE` | Report file with relationships | `--report relations.reporter` |
| `--story FILE` | Markdown story file | `--story story.md` |

## Input Formats

### Plain Text

```text
President Biden announced new climate legislation today.
The bill aims to reduce carbon emissions by 40% by 2030.
```

### Markdown Story

```markdown
# Breaking News

President Biden announced new climate legislation...

## Key Points

- Carbon emissions target
- Timeline by 2030
```

### Entities File Format

```
<entities>
PERSON: Joe Biden, Kamala Harris
ORG: White House, Congress
GPE: United States, Washington
EVENT: Climate Act
DATE: 2025
</entities>
```

### Report File Format

```
<report>
<relations>
("Biden","Climate Bill","announced","New climate legislation")
("Congress","Climate Bill","passed","With bipartisan support")
</relations>
</report>
```

## Examples

### Basic Video from Text

```bash
python mgm/mgm.py article.txt output.mp4
```

### With Entity Awareness

```bash
python mgm/mgm.py article.txt output.mp4 --data entities.txt
```

### With Relationship Context

```bash
python mgm/mgm.py article.txt output.mp4 --data entities.txt --report relations.reporter
```

### From Markdown Story

```bash
python mgm/mgm.py "" output.mp4 --story story.md
```

### From Stdin

```bash
cat article.txt | python mgm/mgm.py - output.mp4
```

## Features

### Entity-Aware Backgrounds

- Prioritizes EVENT > PERSON > ORG > GPE > FAC > PRODUCT > NORP > MONEY > CARDINAL > DATE
- Generates multiple backgrounds per segment based on detected entities
- Location-specific prompts (e.g., Middle East conflict gets war zone backgrounds)

### Relationship-Aware

- Uses relationship data to generate contextually relevant backgrounds
- Violence-related → war/casualty backgrounds
- Diplomatic → negotiation/talk backgrounds
- Official actions → government briefing backgrounds

### TTS (Kokoro)

- Voice: `af_heart` (default)
- Sample rate: 24kHz
- GPU acceleration on Metal (Apple Silicon) or CUDA
- Falls back to CPU on failure

### Video Features

- Resolution: 1920x1080 (default)
- FPS: 30
- Hardware encoding: h264_videotoolbox on macOS
- Multiple backgrounds per segment with crossfade
- News banner with scrolling ticker
- Person portrait overlays
- Text overlays with subtitles

## Output

```
=== Enhanced System ===
Platform: Darwin arm64
PyTorch: 2.1.0
Metal: True
Parsing article …
Found 5 segments
🎨 Generating background 1/6: breaking news background...
TTS: President Biden announced new...
=== Compositing final video ===
⚡ Total: 45.2s
✓ Saved: output.mp4
🎨 Enhanced with multiple context-aware backgrounds!
```

## Dependencies

```bash
pip install torch diffusers kokoro-mlx moviepy soundfile pillow
```

Or use the conda environment defined in Makefile.

## Makefile Integration

```makefile
mp3small:
	source $(MP3_ENV)/bin/activate && python mgm/mgm.py $(TITLEFILE) output/$(NAMESPACE).mp4
```

## Hardware Acceleration

- **macOS (Apple Silicon)**: Metal GPU via PyTorch MPS
- **Linux (NVIDIA)**: CUDA via PyTorch
- **Fallback**: CPU (slower)

## Cache

Background images are cached in `/tmp/video_cache/` to avoid regenerating.

## Troubleshooting

### "too many open files"

Fixed — audio clips are now loaded into RAM and closed immediately.

### Metal not available

Will fall back to CPU with warning: `⚠ Using CPU`

### VideoToolbox not available

Falls back to libx264 software encoding.