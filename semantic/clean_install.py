#!/usr/bin/env python3
"""
Completely clear cache and re-download with safetensors support
"""
import shutil
from pathlib import Path

# Remove entire model cache
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--zilliz--semantic-highlight-bilingual-v1"))

if model_dirs:
    for model_dir in model_dirs:
        print(f"Removing: {model_dir}")
        shutil.rmtree(model_dir)
        print("✓ Removed")

# Also clear transformers_modules cache
modules_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "zilliz"
if modules_dir.exists():
    print(f"Removing: {modules_dir}")
    shutil.rmtree(modules_dir)
    print("✓ Removed")

print("\n" + "="*60)
print("Installing safetensors...")
print("="*60)
import subprocess
subprocess.run(["pip", "install", "safetensors"], check=True)

print("\n" + "="*60)
print("Downloading model (this will take a few minutes)...")
print("="*60)

from transformers import AutoModel
import warnings
warnings.filterwarnings('ignore')

model = AutoModel.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True,
)

print("\n✓ Model loaded successfully!")

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

print("\n" + "="*60)
print("✓ Setup complete! You can now use orig.py")
print("="*60)
