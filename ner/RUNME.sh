#!/bin/bash
# NER processing script - called from Makefile
# Usage: RUNME.sh <date>

DATE=${1:-$(date +%Y-%m-%d)}

# Activate flair conda env and run NER
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate flair

cd /Users/kyle/hub/propaganda/ner

# Run NER processor that queries MongoDB for articles since $DATE
# and calls the NER service on port 8100
python -c "
import sys
sys.path.insert(0, '.')
from processor import process_date
process_date('$DATE')
"