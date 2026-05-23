#!/usr/bin/env zsh
#
# profile_single.sh - Time a single make target
# Usage: ./profile_single.sh <target> [timeout_seconds]

OUTPUT_FILE="/Users/kyle/hub/propaganda/timing_results.csv"
MAKEFILE_PATH="/Users/kyle/hub/propaganda/Makefile"
TARGET=$1
TIMEOUT=${2:-600}

if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <target> [timeout_seconds]"
    echo "Example: $0 load 300"
    exit 1
fi

cd /Users/kyle/hub/propaganda

echo ">>> Running: $TARGET (timeout: ${TIMEOUT}s)"

output=$(gtimeout "$TIMEOUT" /usr/bin/time -p make -f "$MAKEFILE_PATH" "$TARGET" 2>&1)
exit_code=$?

real_time=$(echo "$output" | grep "^real" | awk '{print $2}')
user_time=$(echo "$output" | grep "^user" | awk '{print $2}')
sys_time=$(echo "$output" | grep "^sys" | awk '{print $2}')

if [[ $exit_code -eq 0 ]]; then
    echo "✓ $TARGET completed in ${real_time}s"
    echo "$TARGET,$real_time,$user_time,$sys_time" >> "$OUTPUT_FILE"
elif [[ $exit_code -eq 124 ]]; then
    echo "⚠ $TARGET TIMED OUT after ${TIMEOUT}s"
    echo "$TARGET,TIMEOUT,TIMEOUT,TIMEOUT" >> "$OUTPUT_FILE"
else
    echo "✗ $TARGET FAILED (exit $exit_code)"
    echo "$TARGET,FAILED,$user_time,$sys_time" >> "$OUTPUT_FILE"
fi

echo ""
echo "--- Current Results ---"
cat "$OUTPUT_FILE"