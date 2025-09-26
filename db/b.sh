./mongo2chroma.py query "$1" > tmp_vec.txt
echo "Answer using markdown. $1 use the following data to answer:\n" > tmp_q.txt
cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b
