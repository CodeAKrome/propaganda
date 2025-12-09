#!/bin/zsh
startdate="$1"
filename="$2"
entity="$3"
query="$4"
fulltext=${5:-}

# -----------------------------
# 3. Check input
# -----------------------------
if [[ -z "$query" ]]; then
  echo "Usage: $0 <date offset> <filebase> <entities> <query>"
  exit 1
fi

vec="output/${filename}.vec"
news="output/${filename}.md"
cypherfile="output/${filename}.cypher"
reporterfile="output/${filename}.reporter"
barenews="output/${filename}"
cypher_prompt="cypher_prompt.txt"
ner_key="ner_key.txt"
titlefile="output/titles.tsv"
topn=40
svo_prompt="prompt/kgsvo.txt"

count=$(fgrep 'ID:' "$vec" | wc -l | tr -d '[:space:]')

printf "$filename\t$startdate\t$count\n"

# check for 0 size.
if [[ $count -eq 0 ]]; then
    exit 0
fi

footer=$'\nUse the following data to answer:\n'

read -r -d '' reporter <<'EOF'
You are an expert political analyst and news reporter called Lotta Talker.
The attached file contains the text of news articles.
Summarize the articles in an insightful fashion paying attention to detail.
Describe all the major themes.
If something is irrelevant, ignore it.
If you don't find anything relevant, just say 'Nothing relevant found.'
Describe all the major themes.

Relationships are shown in the <relations> section in (subject,object,verb,explanation) format.

Bias Analysis:
Each article, has a bias analysis in JSON format with the following structure:

a. DIRECTION - The political leaning:
- L = Left (liberal/progressive bias)
- C = Center (balanced/neutral)
- R = Right (conservative bias)

b. DEGREE - The intensity of bias:
- L = Low (minimal bias, mostly factual)
- M = Medium (noticeable bias in framing or emphasis)
- H = High (strong bias, significant editorializing)

3. REASON - A brief explanation (2-4 sentences) justifying your direction and degree ratings based on specific evidence from the article.

-Example-
{"dir": {"L": 0.1, "C": 0.4, "R": 0.5}, "deg": {"L": 0.1, "M": 0.2, "H": 0.7}, "reason": "The article uses loaded language like 'radical agenda' and 'government overreach' while exclusively quoting conservative sources. It omits counterarguments and frames policy proposals in exclusively negative terms."}

Analize the bias of the articles and summarize the bias findings in a concise paragraph at the end of your output.
Do not menntion the bias numbers directly, just summarize the bias findings in a concise paragraph.
Do not reference mongodb id article numbers.
Use the bias data to determine the overall bias of the articles and give that as a conclusion.
Be specific when mentioning which sources are biased and how.

When reporting, speak in a professional newscaster tone like Walter Kronkite.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only.

EOF
 

# -- Ollama
reporter() {
    local model="$1"
    printf "Using model: $model\n"
    ( cat "$svo_prompt" "$vec" ) | ollama run --verbose --hidethinking "$model" | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter $cypher $query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}

# -- Gemini
reportergem() {
    ( cat "$svo_prompt" "$vec" ) | ./gemini.py models/gemini-2.5-flash | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ./gemini.py models/gemini-2.5-flash > "$news"
}

# -- MLX
reportermlx() {
    local model="$1"
    printf "Using model: $model\n"
    ( cat "$svo_prompt" "$vec" ) | ./mlxllm.py - --model "$model" | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ./mlxllm.py - --model "$model" > "$news"
}

#reportergem
#reporter reporterllama3370b:latest
reportermlx mlx-community/Llama-3.3-70B-Instruct-8bit

#reportermlx Wwayu/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-mlx-6Bit
# good, 38m long
#reporter reportergptoss120b
# good, little thin < 1m. no cypher
#reporter reporterllama323b:latest
# junk
#reporter reporterllama318b:latest
# no lotta, sux
#reporter reporterdeepseekcoderv216b:latest
# good, but 11m
#reporter reporterqwen314b:latest
# good 5m
#reporter reporterglm49b:latest
# naw, As a professional newscaster, I would report:
#reporter reportergranite338b:latest
# good, slow 30m
#reporter reporterllama3170b:latest