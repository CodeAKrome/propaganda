cat llms.txt | parallel 'cat prompt_svo_gaza.txt | ollama run {} > out/{= s/:/-/g =}.txt'
