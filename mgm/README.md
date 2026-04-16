# MGM - Video Generation

Generate videos from news reports using SD Turbo for images and Kokoro TTS for audio.

## Quick Reference

| Task | Command |
|------|---------|
| Generate video | `./mkgaza.sh` |
| Run MGM directly | `python enhanced_mgm.py --story <file> --data <vec> --report <cypher> <md> <output>` |

## Shell Scripts

### mkgaza.sh
Generates video from Gaza report data.

```bash
./mkgaza.sh

# Requires pre-existing files in ../db/output_presvo/:
# - gaza.md      (story text)
# - gaza.vec     (vector data)
# - gaza.cypher  (relationships)
# Output: gaza.mp4
```

**Usage:**
```bash
./enhanced_mgm.py --story <story_file> --data <vec_file> --report <cypher_file> <md_file> <output_name>.mp4

# Full example:
./enhanced_mgm.py \
  --story ../db/output/gaza.md \
  --data ../db/output/gaza.vec \
  --report ../db/output/gaza.cypher \
  ../db/output/gaza.md \
  gaza.mp4
```

## Python Scripts

### enhanced_mgm.py
Main video generation script.

```bash
python enhanced_mgm.py [options] <story.md> <output.mp4>

Options:
  --story <file>     Story/script text file
  --data <file>     Vector data file  
  --report <file>   Cypher relationships file
  --voice <name>    Voice name (default: af_sarah)
  --quality <n>     Quality 1-5 (default: 1)
  --fps <n>         Frames per second (default: 24)

Example:
python enhanced_mgm.py --story story.txt --voice af_sarah report.mp4
```

### mgm.py
Original MGM script (legacy).

```bash
python mgm.py <article.txt> <output.mp4>
```

## Configuration

### Voices
Available Kokoro TTS voices:
- `af_sarah` - Sarah (American Female)
- `af_nicole` - Nicole
- `am_michael` - Michael (American Male)
- `bf_emma` - Emma (British Female)
- `bm_george` - George (British Male)

### Environment Variables
```bash
# Required for image generation
HF_TOKEN=your_huggingface_token

# Optional
MGM_QUALITY=1
MGM_VOICE=af_sarah
```

## Requirements

```bash
# Python dependencies
pip install torch torchvision
pip install diffusers
pip install kokoro>=0.9
pip install scipy

# System dependencies
# - ffmpeg (for video encoding)
# - libsndfile1 (for audio)
```

## Output

- `<output>.mp4` - Final video with images and TTS
- `<output>_images/` - Generated images (temp)
- `<output>.wav` - Audio track (temp)

## Troubleshooting

### "No CUDA available"
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- MGM requires CUDA for reasonable performance

### Audio quality issues
- Try different voice: `--voice af_nicole`
- Adjust audio settings in script

### Image generation slow
- Reduce quality: `--quality 1`
- Use smaller diffusion model
- Check GPU memory: `nvidia-smi`

## See Also

- Main README: [../README.md](../README.md)
- Documentation: [../docs/mgm.md](../docs/mgm.md)
- TTS Documentation: [../docs/tts.md](../docs/tts.md)