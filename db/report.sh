startdate="$1"
filename="$2"
entity="$3"
query="$4"
fulltext=${5:-}

vec="output/${filename}.vec"
news="output/${filename}.md"
cypherfile="output/${filename}.cypher"
reporterfile="output/${filename}.reporter"
barenews="output/${filename}"
cypher_prompt="cypher_prompt.txt"
ner_key="ner_key.txt"
titlefile="output/titles.tsv"
topn=20

printf "\n------------- $startdate days ----------------\n"

# -- Hybrid search
if [[ -z "$fulltext" ]]; then
    printf "VECTOR %s Entity %s only search\n" "$filename" "$entity"
    ./hybrid.py "$query" --orentity "$entity" --start-date "$startdate" --showentity -n "$topn" > "$vec"
else
    if [[ $fulltext == +* ]]; then
        filter="--filter"
        text="${fulltext#+}"   # remove leading '+'
    else
        filter=""
        text="$fulltext"
    fi

    if [[ -z "$filter" ]]; then
        printf "UNION %s Entity %s Full text %s search %s\t%s\t%s\n" "$filename" "$entity" "$text"
        ./hybrid.py "$query" --orentity "$entity" --start-date "$startdate" --showentity -n "$topn" --fulltext "$text" > "$vec"
    else
        printf "PREFILTER %s Entity %s Full text %s search %s\t%s\t%s\n" "$filename" "$entity" "$text"
        ./hybrid.py "$query" --orentity "$entity" --start-date "$startdate" --showentity -n "$topn" --fulltext "$text" "$filter" > "$vec"
    fi

fi

count=$(fgrep 'ID:' "$vec" | wc -l | tr -d '[:space:]')

# check for 0 size.
if [[ $count -eq 0 ]]; then
    printf "zilch\t%s\t%s\n" "$filename" "$entity"
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

Relationships are shown in the <relations> section in subject -> verb -> object format.

Respond as if you are a TV reporter on camera explaining to your audience.
Use a professional newscaster tone like Walter Kronkite.
Only reply with what the reporter says, not any stage direction like musical intros or camera direction.
Do not use markup. Do not make tables. Reply with plain text only.

EOF
 
# -----------------------------
# 3. Check input
# -----------------------------
if [[ -z "$query" ]]; then
  echo "Usage: $0 <filebase> <entities> <query>"
  exit 1
fi

# -- SVO cypher no ent prompt
reportersvo() {
    local model="$1"
    ( cat prompt_svo.txt "$vec" ) | ollama run --verbose --hidethinking "$model" | egrep '\->' | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter $cypher $query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ollama run --verbose --hidethinking "$model" > "$news"
}

# -- SVO cypher no ent prompt Gemini
reportergemsvo() {
    ( cat prompt_svo.txt "$vec" ) | ./gemini.py models/gemini-2.5-flash | egrep '\->' | sort | uniq > "$cypherfile"
    cypher=$(<"$cypherfile")
    echo "$reporter\n<relations>\n $cypher \n</relations>\n$query $footer" > "$reporterfile"
    ( cat "$reporterfile" "$vec" ) | ./gemini.py models/gemini-2.5-flash > "$news"
}

reportergemsvo

