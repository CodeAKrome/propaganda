#!/bin/bash
mkdir -p logs

MAX_JOBS=8
JOBS_RUNNING=0

while IFS= read -r cmd; do
  [[ -z "$cmd" ]] || [[ "$cmd" =~ ^# ]] && continue  # Skip empty/comment lines
  
  base=$(echo "$cmd" | awk '{print $2}')
  log="logs/${base}_$(date +%Y%m%d_%H%M%S).log"
  
  echo "[$(date)] Starting: $cmd" | tee "$log"
  eval "$cmd" > >(tee -a "$log") 2>&1 &
  
  ((JOBS_RUNNING++))
  
  if ((JOBS_RUNNING >= MAX_JOBS)); then
    wait -n
    ((JOBS_RUNNING--))
  fi
  
done < runentitybatch.sh

wait
echo "All tasks completed!"
