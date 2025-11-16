xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/lcr_inst.txt prompt/center.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt
