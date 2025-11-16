#!/usr/bin/env python

import sys
import re
import torch
from TTS.api import TTS

modl = "tts_models/en/jenny/jenny"
# Get device
device = "cuda" if torch.cuda.is_available() else "mps"
tts = TTS(modl).to(device)


def read_and_process_file(file_path):
    """Read file, remove newlines, and replace multiple whitespaces with single space."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            
        # Remove all newlines
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Keep only alphanumerics, punctuation, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)


def generate_speech(text, output_file):
    """Convert text to speech and save as MP3."""
    try:
        tts.tts_to_file(text=text, file_path=output_file)
        print(f"Successfully generated audio: {output_file}")
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python text2mp3.py <input_file> <output_mp3>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read and process the input file
    text = read_and_process_file(input_file)
    
    if not text:
        print("Error: Input file is empty or contains only whitespace")
        sys.exit(1)
    
    # Generate speech
    generate_speech(text, output_file)


if __name__ == "__main__":
    main()
