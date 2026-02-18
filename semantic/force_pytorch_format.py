#!/usr/bin/env python3
"""
Force download of PyTorch format (not safetensors)
"""
import os
import shutil
from pathlib import Path

# Delete safetensors file to force PyTorch format
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--zilliz--semantic-highlight-bilingual-v1"))

if model_dirs:
    model_dir = model_dirs[0]
    snapshots = model_dir / "snapshots"
    
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            safetensors_file = snapshot / "model.safetensors"
            if safetensors_file.exists():
                print(f"Removing corrupted safetensors file: {safetensors_file}")
                safetensors_file.unlink()
                print("✓ Removed")

print("\nNow loading model (will download pytorch_model.bin)...")

from transformers import AutoModel
import warnings
warnings.filterwarnings('ignore')

model = AutoModel.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True,
    use_safetensors=False,
)

print("✓ Model loaded successfully!")

# Test it
question = "What are the symptoms of dehydration?"
context = """
Dehydration occurs when your body loses more fluid than you take in.
Common signs include feeling thirsty and having a dry mouth.
The human body is composed of about 60% water.
Dark yellow urine and infrequent urination are warning signs.
Water is essential for many bodily functions.
Dizziness, fatigue, and headaches can indicate severe dehydration.
Drinking 8 glasses of water daily is often recommended.
"""

print("\nProcessing...")
result = model.process(
    question=question,
    context=context,
    threshold=0.5,
    return_sentence_metrics=True,
)

highlighted = result["highlighted_sentences"]
print(f"\n✓ Highlighted {len(highlighted)} sentences:")
for i, sent in enumerate(highlighted, 1):
    print(f"  {i}. {sent}")

if "sentence_probabilities" in result:
    probs = result["sentence_probabilities"]
    print(f"\n✓ Sentence probabilities:")
    for i, prob in enumerate(probs, 1):
        print(f"  Sentence {i}: {prob:.4f}")
