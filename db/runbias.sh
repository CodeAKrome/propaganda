cat output/*.ids | grep -v '^\#' > output/ids.txt
./geminize.py --dst bias --queryfile ../llm/prompt/claudeopus_CoVe.txt --idsource output/ids.txt --mlx --model mlx-community/gpt-oss-20b-MXFP4-Q8
# ./geminize.py --dst bias --queryfile ../llm/prompt/lcr_inst.txt --idsource output/ids.txt --mlx --model mlx-community/Llama-3.3-70B-Instruct-8bit
#./geminize.py --dst bias --queryfile ../llm/prompt/lcr_inst.txt --idsource output/ids.txt --ollama --model llama3.3:70b
#./geminize.py --dst bias --queryfile ../llm/prompt/lcr_inst.txt --start-date -2 --ollama --model llama3.3:70b
