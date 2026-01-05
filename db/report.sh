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
Be specific and list sources when mentioning which sources are biased and how.

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
    local model="$1"
    ( cat "$svo_prompt" "$vec" ) | ./gemini.py models/gemini-2.5-flash | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ./gemini.py $1 > "$news"
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

# -- MLX Ollama Use 2 different models
reportermlxollama() {
    local model="$1"
    local ollama_model="$2"
    printf "MLX model: $model\nOllama model: $ollama_model\n"
    ( cat "$svo_prompt" "$vec" ) | ./mlxllm.py - --model "$model" | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}


# cypher() {
#     local src="$1"
#     local model="$2"
#     # local src="mlx"
#     # local model="mlx-community/Llama-3.3-70B-Instruct-8bit"

#     printf "Cypher source: $src model: $model\n"
#     if [[ "$src" == "ollama" ]]; then
#         ( cat "$svo_prompt" "$vec" ) | ollama run --verbose --hidethinking "$model" | sort | uniq > "$cypherfile"
#     elif [[ "$src" == "gemini" ]]; then
#         ( cat "$svo_prompt" "$vec" ) | ./gemini.py "$model" | sort | uniq > "$cypherfile"
#     elif [[ "$src" == "mlx" ]]; then
#         ( cat "$svo_prompt" "$vec" ) | ./mlxllm.py - --model "$model" | sort | uniq > "$cypherfile"
#     else
#         echo "Unknown source: $src"
#     fi
# }

# report() {
#     local src="$1"
#     local model="$2"
#     # local src="gemini"
#     # local model="models/gemini-3-flash-preview"

#     printf "Report source: $src model: $model\n"
#     cypher=$(<"$cypherfile")
#     echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
#     if [[ "$src" == "ollama" ]]; then
#         ( cat "$reporterfile" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
#     elif [[ "$src" == "gemini" ]]; then
#         ( cat "$reporterfile" "$vec" ) | ./gemini.py "$model" > "$news"
#     elif [[ "$src" == "mlx" ]]; then
#         ( cat "$reporterfile" "$vec" ) | ./mlxllm.py - --model "$model" > "$news"
#     else
#         echo "Unknown source: $src"
#     fi
# }




# -----------------------------
# Cypher generation with failover
# -----------------------------
cypher() {
    local pairs=("$@")
    local src model cmd_output exit_code

    for (( i=1; i<=${#pairs[@]}; i+=2 )); do
        src="${pairs[i]}"
        model="${pairs[i+1]}"

        printf "Trying cypher with %s: %s\n" "$src" "$model"

        if [[ "$src" == "ollama" ]]; then
            cmd_output=$( (cat "$svo_prompt" "$vec") | time ollama run --hidethinking "$model" 2>/dev/null | sort | uniq )
            exit_code=$?
        elif [[ "$src" == "gemini" ]]; then
            cmd_output=$( (cat "$svo_prompt" "$vec") | time ./gemini.py "$model" 2>/dev/null | sort | uniq )
            exit_code=$?
        elif [[ "$src" == "mlx" ]]; then
            cmd_output=$( (cat "$svo_prompt" "$vec") | time ./mlxllm.py - --model "$model" 2>/dev/null | sort | uniq )
            exit_code=$?
        else
            echo "Unknown cypher source: $src" >&2
            continue
        fi

        if [[ $exit_code -eq 0 && -n "$cmd_output" ]]; then
            echo "$cmd_output" > "$cypherfile"
            printf "Cypher succeeded with %s: %s\n" "$src" "$model"
            return 0
        else
            printf "Cypher failed with %s: %s (exit_code: %d)\n" "$src" "$model" "$exit_code" >&2
        fi
    done

    echo "All cypher attempts failed." >&2
    > "$cypherfile"  # empty file on total failure
    return 1
}

# -----------------------------
# Report generation with failover
# -----------------------------

report() {
    local pairs=("$@")
    local src model exit_code

    # Ensure cypherfile exists (even if empty)
    [[ -f "$cypherfile" ]] || > "$cypherfile"
    cypher_content=$(<"$cypherfile")

    echo "$reporter\n<relations>\n $cypher_content \n</relations>\n$query $footer" > "$reporterfile"

    for (( i=1; i<=${#pairs[@]}; i+=2 )); do
        src="${pairs[i]}"
        model="${pairs[i+1]}"

        printf "Trying report with %s: %s\n" "$src" "$model" >&2

        if [[ "$src" == "ollama" ]]; then
            ( cat "$reporterfile" "$vec" ) | time ollama run --hidethinking "$model" > "$news" 2>/dev/null
            exit_code=$?
        elif [[ "$src" == "gemini" ]]; then
            ( cat "$reporterfile" "$vec" ) | time ./gemini.py "$model" > "$news" 2>/dev/null
            exit_code=$?
        elif [[ "$src" == "mlx" ]]; then
            ( cat "$reporterfile" "$vec" ) | time ./mlxllm.py - --model "$model" > "$news" 2>/dev/null
            exit_code=$?
        else
            echo "Unknown report source: $src" >&2
            continue
        fi

        if [[ $exit_code -eq 0 && -s "$news" ]]; then
            printf "Report succeeded with %s: %s\n" "$src" "$model" >&2
            return 0
        else
            printf "Report failed with %s: %s (exit_code: %d, output empty: %s)\n" \
                   "$src" "$model" "$exit_code" "$( [[ ! -s "$news" ]] && echo yes || echo no )" >&2
            > "$news"  # clear partial/failed output
        fi
    done

    echo "All report attempts failed." >&2
    echo "Nothing relevant found or generation failed." > "$news"
    return 1
}



# -----------------------------
# Cypher generation with failover
# -----------------------------

# Preferred order: try high-quality MLX first for cypher, then Gemini as fallback
# cypher \
#     "mlx"   "mlx-community/Llama-3.3-70B-Instruct-8bit" \
#     "gemini" "models/gemini-2.5-flash" \
#     "ollama" "llama3.1:70b"

# "mlx-community/MiniMax-M2.1-3bit"

cypher \
    "mlx"   "mlx-community/MiniMax-M2.1-3bit"

# cypher \
#     "mlx"   "mlx-community/Llama-3.3-70B-Instruct-8bit" 

# For reporting, prefer fast Gemini, fall back to others if needed
report \
    "gemini" "models/gemini-3-flash-preview" \
    "gemini" "models/gemini-2.5-flash" \
    "mlx"   "mlx-community/MiniMax-M2.1-3bit"

# report \
#     "gemini" "models/gemini-3-flash-preview" \
#     "gemini" "models/gemini-2.5-flash" \
#     "mlx"    "mlx-community/Llama-3.3-70B-Instruct-8bit"


# cypher "mlx" "mlx-community/Llama-3.3-70B-Instruct-8bit"
# report "gemini" "models/gemini-2.5-flash"
# report "gemini" "models/gemini-3-flash-preview"
# cypher
# report


#reportergem models/gemini-2.5-flash 
#reporter reporterllama3370b:latest
#reporter gemini-3-flash-preview:cloud
#reportermlx mlx-community/Llama-3.3-70B-Instruct-8bit
#reportermlxollama mlx-community/Llama-3.3-70B-Instruct-8bit gemini-3-flash-preview:cloud

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