#!/usr/bin/env python3

"""
Module for diagnosse_model.py.
"""
"""
Diagnose the model download issue
"""
import os
from pathlib import Path
import safetensors

# Find the model cache
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--zilliz--semantic-highlight-bilingual-v1"))

if not model_dirs:
    print("❌ Model not found in cache!")
    exit(1)

model_dir = model_dirs[0]
print(f"✓ Found model at: {model_dir}")

# Find the snapshots directory
snapshots = model_dir / "snapshots"
if not snapshots.exists():
    print("❌ No snapshots directory!")
    exit(1)

# Get the latest snapshot
snapshot_dirs = list(snapshots.iterdir())
if not snapshot_dirs:
    print("❌ No snapshots found!")
    exit(1)

latest = sorted(snapshot_dirs)[-1]
print(f"✓ Latest snapshot: {latest.name}")

# Check for model files
print("\n📁 Files in snapshot:")
for file in sorted(latest.iterdir()):
    size = file.stat().st_size / (1024**3)  # GB
    print(f"  {file.name:<40} {size:>8.2f} GB")

# Try to load the safetensors file
safetensors_file = latest / "model.safetensors"
if safetensors_file.exists():
    print(f"\n🔍 Inspecting model.safetensors...")
    try:
        from safetensors import safe_open
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            print(f"  ✓ Metadata: {metadata}")
            keys = list(f.keys())
            print(f"  ✓ Number of tensors: {len(keys)}")
            print(f"  ✓ First few keys: {keys[:5]}")
    except Exception as e:
        print(f"  ❌ Error reading safetensors: {e}")
        print("\n💡 Solution: Delete the file and re-download")
        print(f"  rm -f '{safetensors_file}'")
else:
    print("\n❌ model.safetensors not found!")

# Check for pytorch_model.bin
pytorch_file = latest / "pytorch_model.bin"
if pytorch_file.exists():
    print(f"\n✓ pytorch_model.bin exists ({pytorch_file.stat().st_size / (1024**3):.2f} GB)")
else:
    print(f"\n❌ pytorch_model.bin not found")
