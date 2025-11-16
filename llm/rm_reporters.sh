ollama list | awk '/reporter/ {print $1}' | xargs -n1 ollama rm
