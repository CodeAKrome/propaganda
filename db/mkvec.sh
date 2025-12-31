startdate="$1"
filename="$2"
entity="$3"
query="$4"
fulltext=${5:-}

timestampfile="timestamp.txt"
if [[ -e "$timestampfile" ]]; then
    startdate=$(<"$timestampfile")
fi

# -----------------------------
# 3. Check input
# -----------------------------
if [[ -z "$query" ]]; then
  echo "Usage: $0 <date offset> <filebase> <entities> <query> <+fulltext> means prefilter"
  exit 1
fi

vec="output/${filename}.vec"
topn=40
idfile="output/${filename}.ids"

printf "\n------------- $startdate days ----------------\n"

common_args=(
    "$query"
    --bm25
    --orentity "$entity"
    --start-date "$startdate"
    --showentity
    -n "$topn"
    --ids "$idfile"
)

# -- Hybrid search
if [[ -z "$fulltext" ]]; then
    printf "VECTOR %s Entity %s only search\n" "$filename" "$entity"
    ./hybrid.py "${common_args[@]}" > "$vec"
else
    text="${fulltext#+}" # remove leading '+' if present
    filter_arg=$([[ $fulltext == +* ]] && echo "--filter" || echo "")

    mode=$([[ -z "$filter_arg" ]] && echo "UNION" || echo "PREFILTER")
    printf "%s %s Entity %s Full text %s search\n" "$mode" "$filename" "$entity" "$text"
    
    ./hybrid.py "${common_args[@]}" --fulltext "$text" $filter_arg > "$vec"
fi
