#./mongo2chroma.py query "$2" > tmp_vec.txt
#./mongo2chroma.py query "$2" --start-date -7 > tmp_vec.txt
./mongo2chroma.py query "$2" --start-date -7 > output/${1}.vec

echo "You are a seasoned veteran reporter named Max Blowhard. You are an expert in geopolitical matters. Format your answer as if you are broadcasting a news report on radio without any music or sound effects. Do not show thinking. Just show your answer. Do not use a table. Do not refer to mongo db ids. Only pay attention to relevant facts. If there are no relevent articles, simply say 'No relevant articles found.'\n $2 \nUse the following data to answer:\n" > tmp_q.txt
#echo "\nkimi-k2:1t-cloud\n===\n" > output/$1
#echo "\nmodels/gemini-2.5-flash\n===\n" >> output/$1

# ----------

#cat tmp_q.txt tmp_vec.txt | ollama run --verbose  > output/$1
#cat tmp_q.txt output/${1}.vec | ollama run --verbose  > output/${1}.md

# reporter is llama3.3:70b
cat tmp_q.txt output/${1}.vec | ollama run --verbose reporterqwen314b:latest > output/${1}.md
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose qwen3:14b > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose granite4:small-h > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose glm4:9b > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama3.1:70b > output/$1

# -- gemini --

#cat tmp_q.txt output/${1}.vec | ./gemini.py > output/${1}.md
#cat tmp_q.txt tmp_vec.txt | ./gemini.py models/gemini-2.5-flash > output/$1
#cat tmp_q.txt tmp_vec.txt | ./gemini.py > output/$1

# --- mlx ---

#cat tmp_q.txt tmp_vec.txt | ./mlxllm.py - > output/$1
#cat tmp_q.txt tmp_vec.txt | ./mlxllm.py - mlx-community/gemma-3-27b-it-bf16 > output/$1

# == cloud ==

# NO
#cat tmp_q.txt output/${1}.vec | ollama run --verbose gpt-oss:120b-cloud > output/${1}.md
#cat tmp_q.txt output/${1}.vec | ollama run --verbose qwen3-coder:480b-cloud > output/${1}.md
# YES
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose kimi-k2:1t-cloud > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose deepseek-v3.1:671b-cloud > output/$1

# -- slow --

#cat tmp_q.txt tmp_vec.txt | ollama run --verbose --hidethinking deepseek-r1:70b > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose granite4:tiny-h > output/$1

# ----------

#echo "\nmagistral:24b\n===\n" >> output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b >> output/$1
#echo "\nllama4:scout\n===\n" >> output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama4:scout >> output/$1
