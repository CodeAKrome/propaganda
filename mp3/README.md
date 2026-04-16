# MP3 - Text-to-Speech Processing

Generate MP3 audio files from news reports using Kokoro TTS.

## Quick Reference

| Task | Script |
|------|--------|
| Batch TTS conversion | `./batch.sh` |
| Generate batch script | `./mkmkbatch.sh` |
| Install TTS dependencies | `./INSTALL_TTS.sh` |
| Single file conversion | `python ttskokoro.py <input.md> <output.mp3>` |

## Shell Scripts

### batch.sh
Converts multiple report files to MP3 in batch.

```bash
./batch.sh

# Converts files from ../db/output/ to mp3/:
# - ai.md        → mp3/ai.mp3
# - cambodia.md  → mp3/cambodia.mp3
# - hk.md        → mp3/hk.mp3
# - jihad.md     → mp3/jihad.mp3
# - maduro.md    → mp3/maduro.mp3
# - mexico.md    → mp3/mexico.mp3
# - thailand.md  → mp3/thailand.mp3
# - venezuela.md → mp3/venezuela.mp3
```

### mkmkbatch.sh
Generates batch.sh from all .md files in ../db/output/.

```bash
./mkmkbatch.sh

# Reads: ls -1 ../db/output/*.md
# Writes: batch.sh (TTS commands for each file)

# Usage:
# 1. Generate reports with db/ scripts
# 2. Run ./mkmkbatch.sh to create batch.sh
# 3. Run ./batch.sh to convert all to MP3
```

### INSTALL_TTS.sh
Installs Coqui TTS system dependencies.

```bash
./INSTALL_TTS.sh

# Runs:
# git clone https://github.com/coqui-ai/TTS/
# cd TTS
# make install
```

## Python Scripts

### ttskokoro.py
Converts text to speech using Kokoro TTS.

```bash
python ttskokoro.py <input.txt> <output.mp3>

Arguments:
  input.txt    Text file to convert (or .md file)
  output.mp3  Output MP3 filename

Options:
  --voice <name>    Voice to use (default: af_sarah)
  --speed <float>  Speech speed (default: 1.0)
  --quality <int>  Quality 1-5 (default: 1)

Example:
python ttskokoro.py ../db/output/ai.md mp3/ai.mp3
python ttskokoro.py --voice am_michael --speed 0.9 story.txt audio.mp3
```

### tts.py
Legacy TTS script.

```bash
python tts.py <input.txt> <output.mp3>
```

## Available Voices

| Voice ID | Description |
|----------|-------------|
| `af_sarah` | American Female - Sarah |
| `af_nicole` | American Female - Nicole |
| `af_sky` | American Female - Sky |
| `am_michael` | American Male - Michael |
| `am_eric` | American Male - Eric |
| `bf_emma` | British Female - Emma |
| `bf_lily` | British Female - Lily |
| `bm_george` | British Male - George |
| `bm_lewis` | British Male - Lewis |

## Configuration

### Environment Variables
```bash
# Optional
KOKORO_VOICE=af_sarah
KOKORO_SPEED=1.0
```

### Output Directory
MP3 files are saved to `mp3/` directory.

## Requirements

```bash
# Python dependencies
pip install kokoro>=0.9
pip install scipy
pip install soundfile

# System dependencies (Linux)
apt-get install libsndfile1 ffmpeg
```

## Workflow

```bash
# 1. Generate reports
cd ../db
./runmkvecbatch.sh
./runentitybatch.sh

# 2. Create batch script
cd ../mp3
./mkmkbatch.sh

# 3. Convert to MP3
./batch.sh

# 4. Listen to results
ls mp3/*.mp3
```

## Troubleshooting

### "No module named kokoro"
```bash
pip install kokoro
```

### Audio sounds robotic
- Try different voice
- Adjust speed: `--speed 0.9`
- Use higher quality: `--quality 3`

### File not found
- Ensure source .md files exist in ../db/output/
- Check relative paths are correct

## See Also

- Main README: [../README.md](../README.md)
- MGM Video Generation: [../mgm/README.md](../mgm/README.md)
- TTS Documentation: [../docs/tts.md](../docs/tts.md)