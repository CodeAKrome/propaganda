./mongo2chroma.py query "$1" > tmp_vec.txt
echo "Answer using markdown. $1 use the following data to answer:\n" > tmp_q.txt
echo "\n===\ngemini\n===\n"
cat tmp_q.txt tmp_vec.txt | ./gemini.py
echo "\n===\nmistral-small3.2:latest\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose mistral-small3.2:latest
echo "\n===\nmagistral:24b\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b
echo "\n===\nllama4:scout\n===\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama4:scout
