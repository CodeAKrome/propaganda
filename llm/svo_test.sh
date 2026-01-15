xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/svo_test.xml | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/svo_test.txt
