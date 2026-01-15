xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/svo_test.xml | ollama run --verbose --hidethinking {} 2>&1' < llms_big.txt | tee out/svo_bigtest.txt
