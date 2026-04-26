cat llms.txt | parallel 'cat prompt_svo_gaza.txt | ollama run --hidethinking {} > out/{= s/:/-/g =}.txt'
