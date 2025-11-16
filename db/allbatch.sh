./mongo2chroma.py query "$2" > tmp_vec.txt
echo "Answer using Markdown. Do not use a table. $2 Use the following data to answer:\n" > tmp_q.txt
#echo "\nkimi-k2:1t-cloud\n===\n" > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose kimi-k2:1t-cloud >> output/$1
#echo "\nmodels/gemini-2.5-flash\n===\n" >> output/$1

#cat tmp_q.txt tmp_vec.txt | ./gemini.py models/gemini-2.5-flash > output/$1
cat tmp_q.txt tmp_vec.txt | ./gemini.py > output/$1

#echo "\nmagistral:24b\n===\n" >> output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose magistral:24b >> output/$1
#echo "\nllama4:scout\n===\n" >> output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose llama4:scout >> output/$1
