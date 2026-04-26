./mongo2chroma.py query "$1" > tmp_vec.txt
echo "Answer using markdown. Do not use a table. $1 use the following data to answer:\n" > tmp_q.txt
echo "\nkimi-k2:1t-cloud\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --hidethinking --verbose kimi-k2:1t-cloud
echo "\nmodels/gemini-2.5-flash\n===\n"
cat tmp_q.txt tmp_vec.txt | ./gemini.py models/gemini-2.5-flash
echo "\nmistral-small3.2:latest\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --hidethinking --verbose mistral-small3.2:latest
echo "\nmagistral:24b\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --hidethinking --verbose magistral:24b
echo "\nllama4:scout\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --hidethinking --verbose llama4:scout
