startdate="-7"
vec="output/${1}.vec"
news="output/${1}.md"
cypherfile="output/${1}.cypher"
barenews="output/${1}"
cypher_prompt="cypher_prompt.txt"
cypher_output="cypher_output.txt"
ner_key="ner_key.txt"
titlefile="output/titles.txt"

# export titles
./mongo2chroma.py title --start-date "$startdate" > "$titlefile"

# -- Do VECTOR search
./mongo2chroma.py query "$2" --start-date "$startdate" | ./clean.pl > "$vec"
#./mongo2chroma.py query "$2" | ./clean.pl > "$vec"


count=$(fgrep 'ID:' "$vec" | wc -l | tr -d '[:space:]')

# check for 0 size.
if [[ $count -eq 0 ]]; then
    printf "zilch\t%s\n" "$2" 
    exit 0
fi

footer=$'\nUse the following data to answer:\n'

printf "%s\t%s\n" "$count" "$2"


read -r -d '' reporter <<'EOF'
You are an expert political analyst and news reporter called Lotta Talker.
The attached file contains the text of news articles.
Summarize the articles in an insightful fashion paying attention to detail.
Describe all the major themes.
If something is irrelevant, ignore it.
If you don't find anything relevant, just say 'Nothing relevant found.'
Describe all the major themes.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only.
EOF

read -r -d '' reporterent <<'EOF'
You are an expert political analyst and news reporter called Lotta Talker.
The attached file contains the text of news articles.
Summarize the articles in an insightful fashion paying attention to detail.
Describe all the major themes.
If something is irrelevant, ignore it.
If you don't find anything relevant, just say 'Nothing relevant found.'

Use the <entities> section which contains a comma delimited lists of ontonotes named entities from NER.
Use the cypher section which shows relationships.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only.
EOF

 
# -----------------------------
# 3. Check input
# -----------------------------
if [[ -z "$2" ]]; then
  echo "Usage: $0 <arg1> <text_to_include>"
  exit 1
fi

# -- cypher


# --

#echo "$2$footer" > tmp_q.txt
#echo "$reporter $2$footer" > tmp_reporter.txt

# -- multi models
reporter() {
    local model="$1"
    echo "$model"
    ( cat "$cypher_prompt" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$cypherfile"
    cypher=$(<"$cypherfile")
    ner=$(<"$ner_key")
    echo "$reporterent $cypher $2 $ner $footer" > tmp_reporter.txt
    ( cat tmp_reporter.txt "$vec" ) | ollama run --verbose --hidethinking "$model" > "$barenews.$model"
}

# -- entities cypher
reporterent() {
    local model="$1"
    ( cat "$cypher_prompt" "$vec" ) | ollama run --verbose --hidethinking "$model" | egrep '\-\[' > "$cypherfile"
    cypher=$(<"$cypherfile")
    ner=$(<"$ner_key")
    echo "$reporterent $cypher $2 $ner $footer" > tmp_reporter.txt
    ( cat tmp_reporter.txt "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}

# -- gemini entities cypher
reportergem() {
    ( cat "$cypher_prompt" "$vec" ) | ./gemini.py models/gemini-2.5-flash > "$cypherfile"
    cypher=$(<"$cypherfile")
    ner=$(<"$ner_key")
    echo "$reporterent $cypher $2 $ner $footer" > tmp_reporter.txt
    ( cat tmp_reporter.txt "$vec" ) | ./gemini.py models/gemini-2.5-flash > "$news"
}

# -- MLX entities cypher
reportermlx() {
    ( cat "$cypher_prompt" "$vec" ) | ./gemini.py models/gemini-2.5-flash > "$cypherfile"
    cypher=$(<"$cypherfile")
    ner=$(<"$ner_key")
    echo "$reporterent $cypher $2 $ner $footer" > tmp_reporter.txt
    ( cat tmp_reporter.txt "$vec" ) | ././mlxllm.py - mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-8bit > "$news"
}

# -- no entities
reporternoent() {
    local model="$1"
    echo "$reporterent $2$footer" > tmp_reporter.txt
    ( cat tmp_reporter.txt "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}


#reportergem
#reportermlx

#cat tmp_q.txt tmp_vec.txt | ./mlxllm.py - mlx-community/gemma-3-27b-it-bf16 > output/$1
#mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-8bit


# -- gemini --

#cat tmp_reporter.txt output/${1}.vec | ./gemini.py models/gemini-2.5-flash > output/${1}.md
#cat tmp_reporter.txt output/${1}.vec | ./gemini.py > output/${1}.md

# shat bed!
#reporter reporterllama4scout:latest

# NO!
#reporter reportergranite4tinyh:latest
#reporter reporterllama318b:latest

# BAD! shows thinking
#reporter reporterqwen332b:latest

# ok = no Lotta
#reporter reportergptoss20b:latest
#reporter reporterqwen314b:latest
#reporter reportergranite4smallh:latest

# Lotta

# -- smol
# doesn't know jew are christians
#reporter reporterqwen38b:latest

# nice 1m
#reporterent reporterqwen25coderlatest:latest

# too short for 1m
#reporter reporterglm49b:latest

# 2m way too slow, short too
#reporter reportergranite338b:latest

# 2.5m repeating BAD!
#reporter reporterllama323b:latest

# -- under 16
# 15G 4m

# WORKS new
#reporterent reportermistralsmall32latest:latest

# 3m
# WORKS new
#reporter reportermagistral24b:latest

# -- 17
# ok
#reporter reportergemma327b:latest

# -- Just Lotta
# doesn't know jews christian 10m

# WORKS new
reporterent reporterdeepseekr170b:latest
# very slow 10m
#reporter reporterllama3170b:latest
#reporter reportercogito70b:latest
#reporter reporterllama3370b:latest



# --- mlx ---

#cat tmp_q.txt tmp_vec.txt | ./mlxllm.py - > output/$1
#cat tmp_q.txt tmp_vec.txt | ./mlxllm.py - mlx-community/gemma-3-27b-it-bf16 > output/$1
#mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-8bit

# == cloud ==

# NO
#cat tmp_q.txt output/${1}.vec | ollama run --verbose gpt-oss:120b-cloud > output/${1}.md
#cat tmp_q.txt output/${1}.vec | ollama run --verbose qwen3-coder:480b-cloud > output/${1}.md
# YES
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose kimi-k2:1t-cloud > output/$1
#cat tmp_q.txt tmp_vec.txt | ollama run --verbose deepseek-v3.1:671b-cloud > output/$1
