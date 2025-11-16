xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/left.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt
