#!/usr/bin/env zsh
#
# profile_testrun.sh - Time each target in the testrun pipeline
# Run with timeout support and incremental output
#
# Usage: ./profile_testrun.sh [timeout_seconds_per_target]

OUTPUT_FILE="/Users/kyle/hub/propaganda/timing_results.csv"
MAKEFILE_PATH="/Users/kyle/hub/propaganda/Makefile"
TIMEOUT=${1:-1800}  # Default 30 min per target

# Targets in testrun order (from Makefile line 20)
TARGETS=(
    "timestamp"
    "load"
    "ner"
    "t5bias"
    "vector"
    "entity"
    "cleantext"
    "cleanoutput"
    "runhybrid"
    "runreport"
    "cyphertograph"
    "cleanmp3"
    "mp3small"
    "dbscan"
    "vecdbscan"
    "mddbscan"
    "fini"
)

echo "target,real,user,sys" > "$OUTPUT_FILE"
echo "Profiling testrun targets (timeout: ${TIMEOUT}s per target)..."
echo "============================================"

cd /Users/kyle/hub/propaganda

for target in "${TARGETS[@]}"; do
    echo ""
    echo ">>> Running: $target"
    
    # Run with timeout, capture timing
    output=$(gtimeout "$TIMEOUT" /usr/bin/time -p make -f "$MAKEFILE_PATH" "$target" 2>&1)
    exit_code=$?
    
    # Extract timing from stderr
    real_time=$(echo "$output" | grep "^real" | awk '{print $2}')
    user_time=$(echo "$output" | grep "^user" | awk '{print $2}')
    sys_time=$(echo "$output" | grep "^sys" | awk '{print $2}')
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ $target completed in ${real_time}s"
        echo "$target,$real_time,$user_time,$sys_time" >> "$OUTPUT_FILE"
    elif [[ $exit_code -eq 124 ]]; then
        echo "⚠ $target TIMED OUT after ${TIMEOUT}s"
        echo "$target,TIMEOUT,TIMEOUT,TIMEOUT" >> "$OUTPUT_FILE"
    else
        echo "✗ $target FAILED (exit $exit_code)"
        echo "$target,FAILED,$user_time,$sys_time" >> "$OUTPUT_FILE"
    fi
    
    # Show current results
    echo ""
    echo "--- Current Results ---"
    cat "$OUTPUT_FILE"
    echo "----------------------"
done

echo ""
echo "============================================"
echo "FINAL RESULTS:"
echo ""
column -t -s',' "$OUTPUT_FILE"