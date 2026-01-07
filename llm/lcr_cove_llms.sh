xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat $1 $2 | ollama run --verbose --hidethinking {} 2>&1' < llms.txt
