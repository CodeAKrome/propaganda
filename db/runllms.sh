cat tmp_q.txt tmp_vec.txt
echo "\n\n=====\n\n"
date
echo "\ngemini-2.5-pro\n"
cat tmp_q.txt tmp_vec.txt | ./gemtest.py
date
echo "\n--\nllama3.2:3b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama3.2:3b
echo "\n--\nqwen3:14b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose qwen3:14b
echo "\n--\ngemma3:27b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose gemma3:27b
echo "\n--\ngpt-oss:20b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose gpt-oss:20b
echo "\n--\nllama3.3:70b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama3.3:70b
echo "\n--\ncogito:70b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose cogito:70b
echo "\n--\ngranite3.3:8b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose granite3.3:8b
echo "\n--\nmistral-small3.2:latest\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose mistral-small3.2:latest
echo "\n--\nllama3.1:70b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama3.1:70b
echo "\n--\nqwen3:8b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose qwen3:8b
echo "\n--\nmagistral:24b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b
echo "\n--\nllama4:scout\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama4:scout
echo "\n--\ndeepseek-r1:70b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose deepseek-r1:70b
echo "\n--\nqwen3:32b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose qwen3:32b
echo "\n--\nllama3.1:8b\n--\n"
cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama3.1:8b
