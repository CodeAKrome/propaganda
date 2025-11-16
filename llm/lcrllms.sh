xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt0.txt rightstrong.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt
