#!/usr/bin/env python3
"""
Clear HuggingFace cache and re-download the model
"""
import os
import shutil
from pathlib import Path

# Find HuggingFace cache
cache_dir = Path.home() / ".cache" / "huggingface"
model_name = "zilliz--semantic-highlight-bilingual-v1"

print("Searching for cached model files...")
print(f"Cache directory: {cache_dir}")

# Look for the model in different cache locations
locations_to_check = [
    cache_dir / "hub" / f"models--{model_name}",
    cache_dir / "modules" / "transformers_modules" / "zilliz" / "semantic-highlight-bilingual-v1",
]

removed_any = False
for location in locations_to_check:
    if location.exists():
        print(f"\n✓ Found: {location}")
        print(f"  Removing...")
        shutil.rmtree(location)
        print(f"  ✓ Removed")
        removed_any = True
    else:
        print(f"\n✗ Not found: {location}")

if removed_any:
    print("\n" + "="*60)
    print("Cache cleared! Now run:")
    print("  python orig.py")
    print("="*60)
else:
    print("\n" + "="*60)
    print("No cached files found. The model will download fresh.")
    print("="*60)
