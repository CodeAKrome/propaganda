#!/usr/bin/env zsh
#
# test_memory.sh - Test memory usage for runreport with N articles
# Uses Python to track memory during execution

cd /Users/kyle/hub/propaganda/db

NUM=${1:-1}
OUTPUT_FILE="../memory_test.csv"

echo "Testing with $NUM article(s)..."

# Get first N rows from hybrid_batch.tsv (skip header)
if [[ $NUM -eq 1 ]]; then
    rows=$(tail -n +2 hybrid_batch.tsv | head -1)
elif [[ $NUM -eq 2 ]]; then
    rows=$(tail -n +2 hybrid_batch.tsv | head -2)
fi

# Create temp TSV with header
head -1 hybrid_batch.tsv > /tmp/test_batch.tsv
echo "$rows" >> /tmp/test_batch.tsv

# Run with Python memory tracking
python3 << 'EOF'
import subprocess
import psutil
import os
import time
import sys

# Start the process
proc = subprocess.Popen(
    ["python", "runreport.py", "/tmp/test_batch.tsv", "--output", "output"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

max_mem = 0
pid = proc.pid

try:
    p = psutil.Process(pid)
    while proc.poll() is None:
        try:
            mem = p.memory_info().rss / 1024 / 1024  # MB
            max_mem = max(max_mem, mem)
        except:
            pass
        time.sleep(0.5)
    
    # Get final output
    output, _ = proc.communicate()
    print(output[-2000:] if len(output) > 2000 else output)
    
    print(f"\n=== Peak Memory: {max_mem:.1f} MB ===", file=sys.stderr)
    print(f"ARTICLES:{os.environ.get('NUM_ARTICLES', '1')}", file=sys.stderr)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)

EOF

NUM_ARTICLES=$NUM python3 << 'EOF'
import subprocess
import psutil
import os
import time
import sys

NUM = int(os.environ.get('NUM_ARTICLES', '1'))

proc = subprocess.Popen(
    ["python", "runreport.py", "/tmp/test_batch.tsv", "--output", "output"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

max_mem = 0
pid = proc.pid

try:
    p = psutil.Process(pid)
    while proc.poll() is None:
        try:
            mem = p.memory_info().rss / 1024 / 1024  # MB
            max_mem = max(max_mem, mem)
        except:
            pass
        time.sleep(0.5)
    
    output, _ = proc.communicate()
    print(output[-2000:] if len(output) > 2000 else output)
    
    print(f"\n=== Peak Memory: {max_mem:.1f} MB ({NUM} articles) ===", file=sys.stderr)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)

rm -f /tmp/test_batch.tsv