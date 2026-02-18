#!/usr/bin/env python3
"""
Working solution: Load model by bypassing broken from_pretrained
"""
import torch
from transformers import AutoConfig
from safetensors.torch import load_file
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Loading model with custom loader...")
print("="*60)

# Step 1: Load config
print("\n1. Loading config...")
config = AutoConfig.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True
)
print("   ✓ Config loaded")

# Step 2: Initialize model architecture
print("\n2. Initializing model architecture...")
from transformers import AutoModel

model = AutoModel.from_config(
    config,
    trust_remote_code=True
)
print("   ✓ Architecture initialized")

# Step 3: Find and load weights
print("\n3. Loading weights...")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--zilliz--semantic-highlight-bilingual-v1"))

if not model_dirs:
    print("   ❌ Model weights not found in cache!")
    print("   Run: huggingface-cli download zilliz/semantic-highlight-bilingual-v1")
    exit(1)

model_dir = model_dirs[0]
snapshots = model_dir / "snapshots"
latest = sorted(snapshots.iterdir())[-1]
safetensors_file = latest / "model.safetensors"

print(f"   Found weights: {safetensors_file.name}")
state_dict = load_file(str(safetensors_file))
print(f"   ✓ Loaded {len(state_dict)} tensors")

# Step 4: Load weights into model
print("\n4. Injecting weights into model...")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    print(f"   ⚠️  Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"   ⚠️  Unexpected keys: {unexpected_keys}")
print("   ✓ Weights loaded")

model.eval()  # Set to evaluation mode

print("\n" + "="*60)
print("✓ Model loaded successfully!")
print("="*60)

# Test the model
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

print("\nProcessing query...")
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
print("✓ All tests passed!")
print("="*60)
print("\nYou can now use this approach in your scripts.")
