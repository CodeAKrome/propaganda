xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/lcr_inst.txt prompt/right.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt
