#!/bin/bash

echo "================================================================"
echo "Complete Fix for zilliz/semantic-highlight-bilingual-v1"
echo "================================================================"
echo ""
echo "Installing compatible package versions..."
echo ""

# Install exact compatible versions
pip install transformers==4.30.2 --force-reinstall
pip install tokenizers==0.13.3 --force-reinstall
pip install safetensors==0.3.1 --force-reinstall

echo ""
echo "================================================================"
echo "✓ Packages installed:"
echo "  - transformers==4.30.2"
echo "  - tokenizers==0.13.3"
echo "  - safetensors==0.3.1"
echo "================================================================"
echo ""
echo "Now run:"
echo "  python orig.py"
