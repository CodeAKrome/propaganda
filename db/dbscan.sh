workdir=$1
filename=$2

vec="${workdir}/${filename}.vec"
cypherfile="${workdir}/${filename}.cypher"
reporterfile="${workdir}/${filename}.reporter"
news="${workdir}/${filename}.md"


svo_prompt="prompt/kgsvo.txt"
lotta_prompt="prompt/LottaTalker.md"
query="The following are different sources covering the same news story. Summarize all the main points and arguments both for and against."
footer=$'\nUse the following data to answer:\n'


# -- Ollama
reportage() {
    local model="$1"
    printf "Using model: $model\n"
    ( cat "$svo_prompt" "$vec" ) | ollama run --verbose --hidethinking "$model" | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    lotta=$(<"$lotta_prompt")
    echo "$lotta $cypher $query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}

reportage gpt-oss:120b
