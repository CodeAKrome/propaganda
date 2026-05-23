#!/usr/bin/env zsh
#
# run_bias.sh - CLI wrapper for T5 bias detection
#
# Usage:
#   ./run_bias.sh --text "Your article text"
#   ./run_bias.sh --input-file article.txt
#   echo "text" | ./run_bias.sh
#   ./run_bias.sh --lora ./my-adapter --model t5-large --text "text"
#
# Options:
#   -t, --text TEXT       Text to classify
#   -i, --input-file FILE Input file with text
#   -l, --lora PATH       LoRA adapter path (default: ./bias-detector-output)
#   -m, --model MODEL    Base model (default: t5-large)
#   -d, --device DEV     Device: mps, cuda, cpu, auto (default: auto)
#   -j, --json           Output compact JSON
#   -h, --help           Show help
#
# Environment:
#   BIAS_LORA_PATH       LoRA adapter path
#   BIAS_MODEL           Base model name
#   T5_DEVICE            Force device

# Default values
LORA_PATH="${BIAS_LORA_PATH:-./bias-detector-output}"
MODEL_NAME="${BIAS_MODEL:-t5-large}"
DEVICE="auto"

# Parse options
zparseopts -D -E -K \
    t:=text_arg -text:=text_arg \
    i:=input_arg -input-file:=input_arg \
    l:=lora_arg -lora:=lora_arg \
    m:=model_arg -model:=model_arg \
    d:=device_arg -device:=device_arg \
    j=json_flag -json=json_flag \
    h=help_flag -help=help_flag

if (( $+help_flag )); then
    print "T5 Bias Detection CLI"
    print ""
    print "Usage: run_bias.sh [options]"
    print ""
    print "Options:"
    print "  -t, --text TEXT       Text to classify"
    print "  -i, --input-file FILE Input file with text"
    print "  -l, --lora PATH       LoRA adapter path (default: ./bias-detector-output)"
    print "  -m, --model MODEL    Base model (default: t5-large)"
    print "  -d, --device DEV     Device: mps, cuda, cpu, auto (default: auto)"
    print "  -j, --json           Output compact JSON"
    print "  -h, --help           Show this help"
    print ""
    print "Examples:"
    print "  ./run_bias.sh --text \"Your article text\""
    print "  ./run_bias.sh --input-file article.txt"
    print "  echo \"text\" | ./run_bias.sh"
    print "  ./run_bias.sh --lora ./my-adapter --text \"text\""
    print ""
    print "Environment:"
    print "  BIAS_LORA_PATH       LoRA adapter path"
    print "  BIAS_MODEL           Base model name"
    print "  T5_DEVICE            Force device"
    exit 0
fi

# Extract values from parsed options
TEXT=""
INPUT_FILE=""
JSON_OUTPUT=""

# Get positional argument (text from stdin if no flag)
if (( $+text_arg )); then
    TEXT="${text_arg[2]}"
elif (( $+input_arg )); then
    INPUT_FILE="${input_arg[2]}"
fi

if (( $+lora_arg )); then
    LORA_PATH="${lora_arg[2]}"
fi

if (( $+model_arg )); then
    MODEL_NAME="${model_arg[2]}"
fi

if (( $+device_arg )); then
    DEVICE="${device_arg[2]}"
fi

if (( $+json_flag )); then
    JSON_OUTPUT="--json"
fi

# Determine text input
if [[ -z "$TEXT" && -z "$INPUT_FILE" ]]; then
    # Check if stdin has content
    if ! tty -s < /dev/stdin 2>/dev/null; then
        TEXT=$(cat -)
    fi
fi

# Build command
CMD=(python3 "${0:h}/bias_cli.py")

if [[ -n "$TEXT" ]]; then
    CMD+=(--text "$TEXT")
elif [[ -n "$INPUT_FILE" ]]; then
    CMD+=(--input-file "$INPUT_FILE")
fi

CMD+=(--lora "$LORA_PATH")
CMD+=(--model "$MODEL_NAME")

if [[ "$DEVICE" != "auto" ]]; then
    CMD+=(--device "$DEVICE")
fi

if [[ -n "$JSON_OUTPUT" ]]; then
    CMD+=("$JSON_OUTPUT")
fi

# Run command
"${CMD[@]}"