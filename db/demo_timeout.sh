#!/bin/bash

# Demo script to show timeout functionality

echo "=== LLM Timeout Feature Demo ==="
echo ""

# Create a test prompt
echo "Creating test prompt..."
echo "Write a comprehensive analysis of global economic trends over the past century." > /tmp/test_prompt.txt

echo "Running LLM with 2-second timeout (should timeout)..."
echo ""

# Run with very short timeout to demonstrate timeout behavior
python db/mlxllm.py /tmp/test_prompt.txt --time_limit 2 --tokens 1000

echo ""
echo "Demo completed. The timeout feature will:"
echo "1. Terminate LLM generation after specified time limit"
echo "2. Return graceful error message with [TIMEOUT_ERROR] prefix"
echo "3. Provide detailed reporting to stderr"
echo "4. Include model info, prompt length, and other metadata"
