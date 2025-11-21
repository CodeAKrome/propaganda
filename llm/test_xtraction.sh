cat llms.txt | parallel -j1 'cat prompt/polxtract.txt prompt/israel.vec | ollama run {} > out/{= s/:/-/g =}.txt'
