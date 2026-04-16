# Semantic - Semantic Search & Embeddings

Semantic search functionality using transformers and embedding models.

## Quick Reference

| Task | Script |
|------|--------|
| Fix dependencies | `./lastfix.sh` |
| Colab setup | `./exact_colab_setup.sh` |
| Run semantic search | `python orig_colab_style.py` |

## Shell Scripts

### lastfix.sh
Fixes dependency issues for semantic search models.

```bash
./lastfix.sh

# Steps:
# 1. Reinstall transformers, tokenizers, torch, nltk
# 2. Clear HuggingFace cache for semantic model
# 3. Run orig_colab_style.py

# Issues addressed:
# - Version conflicts between transformers/tokenizers
# - Corrupted HuggingFace cache
# - Missing dependencies
```

### exact_colab_setup.sh
Sets up environment exactly as Colab would configure it.

```bash
./exact_colab_setup.sh

# Configures Python environment for:
# - Transformer models
# - Semantic highlighting
# - Bilingual embeddings
```

### complete_fix.sh
Comprehensive fix for all semantic search issues.

```bash
./complete_fix.sh

# Runs full diagnostic and repair
# Checks:
# - Package versions
# - Model downloads
# - GPU availability
# - Cache integrity
```

## Python Scripts

### orig_colab_style.py
Main semantic search implementation.

```bash
python orig_colab_style.py [options]

Options:
  --model <name>    Model to use (default: zilliz/semantic-highlight-bilingual-v1)
  --device <dev>    Device: cpu, cuda, mps (default: cuda if available)
  --batch-size <n> Batch size (default: 32)

Example:
python orig_colab_style.py
python orig_colab_style.py --device mps --batch-size 16
```

## Models

### semantic-highlight-bilingual-v1
Zilliz semantic highlighting model.

```python
# Usage
from transformers import AutoModel, AutoTokenizer

model_name = "zilliz/semantic-highlight-bilingual-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### bge-large-en-v1.5
BAAI BGE large embedding model.

```bash
# Used for:
# - Article vectorization
# - Semantic similarity
# - Hybrid search ranking
```

## Configuration

### Environment Variables
```bash
# HuggingFace
export HF_TOKEN="your_token"

# Device
export DEVICE=cuda  # or cpu, mps
```

### Python Dependencies
```bash
pip install transformers>=4.30
pip install torch
pip install tokenizers>=0.13
pip install nltk
pip install numpy
```

## Troubleshooting

### Import errors
```bash
# Reinstall dependencies
pip install transformers --force-reinstall
pip install tokenizers --force-reinstall
```

### Model not found
```bash
# Clear cache
rm -rf ~/.cache/huggingface/hub/

# Login to HuggingFace
huggingface-cli login
```

### GPU out of memory
```bash
# Use smaller batch
python orig_colab_style.py --batch-size 8

# Or use CPU
python orig_colab_style.py --device cpu
```

## See Also

- Main README: [../README.md](../README.md)
- Vector Search: [../docs/mongo2chroma.md](../docs/mongo2chroma.md)
- Hybrid Search: [../docs/hybrid.md](../docs/hybrid.md)