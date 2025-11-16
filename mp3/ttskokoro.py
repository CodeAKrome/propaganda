#!/usr/bin/env python
import sys
import re
from kokoro import KPipeline
import soundfile as sf
import torch

def main():
    if len(sys.argv) != 3:
        print("Usage: ./ttskokoro.py <input_file|-> <output.mp3>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read text from file or stdin
    if input_file == '-':
        text = sys.stdin.read()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Remove all characters except alphanumeric, punctuation, and whitespace
    text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)
    
    # Initialize pipeline and generate audio
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(text, voice='af_heart')
    
    # Collect all audio chunks
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        print(f"Processing chunk {i}: gs={gs}, ps={ps}", file=sys.stderr)
        audio_chunks.append(audio)
    
    # Concatenate and save
    if audio_chunks:
        full_audio = torch.cat(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        sf.write(output_file, full_audio, 24000)
        print(f"Audio saved to {output_file}", file=sys.stderr)
    else:
        print("No audio generated", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()