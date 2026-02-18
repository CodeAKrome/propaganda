#!/bin/bash

echo "================================================================"
echo "Exact Colab Setup for semantic-highlight-bilingual-v1"
echo "================================================================"
echo ""

# Uninstall everything to start fresh
pip uninstall transformers torch tokenizers safetensors -y

# Install latest versions (like Colab)
echo "Installing transformers and torch (latest versions)..."
pip install transformers torch

# Install NLTK for sentence tokenization
echo "Installing nltk..."
pip install nltk

echo ""
echo "================================================================"
echo "✓ Packages installed (latest stable)"
echo "================================================================"
echo ""

# Clear cache to force fresh download
echo "Clearing model cache..."
rm -rf ~/.cache/huggingface/hub/models--zilliz--semantic-highlight-bilingual-v1
rm -rf ~/.cache/huggingface/modules/transformers_modules/zilliz

echo ""
echo "Downloading NLTK punkt tokenizer..."
python -c "import nltk; nltk.download('punkt')"

echo ""
echo "================================================================"
echo "✓ Setup complete!"
echo "================================================================"
echo ""
echo "Now run:"
echo "  python orig.py"
