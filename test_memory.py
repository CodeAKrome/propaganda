#!/usr/bin/env python3
"""
test_memory.py - Test concurrent article processing

Usage: python test_memory.py [num_concurrent]
"""

import subprocess
import psutil
import os
import sys
import time
import threading

NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUTPUT_FILE = os.path.dirname(os.path.abspath(__file__)) + "/memory_test.csv"
db_dir = os.path.dirname(os.path.abspath(__file__)) + "/db"


def get_system_memory():
    """Get used system memory"""
    return psutil.virtual_memory().used / 1024 / 1024 / 1024  # GB


# Get available vec files
with open(db_dir + "/hybrid_batch.tsv") as f:
    header = f.readline()
    data_lines = f.readlines()

available = []
for line in data_lines:
    label = line.split("\t")[0]
    vec_path = db_dir + "/output/" + label + ".vec"
    if os.path.exists(vec_path):
        available.append((label, line))

selected = available[:NUM]
print(f"Testing {NUM} concurrent articles: {[s[0] for s in selected]}")

# Get baseline
time.sleep(1)
baseline_mem = get_system_memory()
print(f"Baseline system memory: {baseline_mem:.2f} GB")


# Create individual TSVs and run concurrently
def run_article(label, line):
    tsv = f"/tmp/test_{label}.tsv"
    with open(tsv, "w") as f:
        f.write(header)
        f.write(line)

    proc = subprocess.Popen(
        ["python", db_dir + "/runreport.py", tsv, "--output", db_dir + "/output"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Monitor peak memory for this process
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, p.memory_info().rss / 1024 / 1024)
            except:
                pass
            time.sleep(0.2)
    except:
        pass

    output, _ = proc.communicate()
    return output, peak


# Run concurrently
results = []
threads = []
peak_system = baseline_mem


def monitor_system():
    global peak_system
    while any(t.is_alive() for t in threads):
        current = get_system_memory()
        peak_system = max(peak_system, current)
        time.sleep(0.3)


monitor_thread = threading.Thread(target=monitor_system)
monitor_thread.start()

# Start all at once
for label, line in selected:
    t = threading.Thread(
        target=lambda l=label, ln=line: results.append(run_article(l, ln))
    )
    t.start()
    threads.append(t)

# Wait for all
for t in threads:
    t.join()

monitor_thread.join()

print(f"Peak system memory: {peak_system:.2f} GB")
delta = peak_system - baseline_mem
print(f"Memory delta: {delta:.2f} GB")

# Print outputs
for r in results:
    print(r[0][-500:] if len(r[0]) > 500 else r[0])

print(f"\n=== Results: {delta:.2f} GB for {NUM} concurrent ===")

with open(OUTPUT_FILE, "a") as f:
    f.write(f"{NUM},{delta:.2f}\n")
