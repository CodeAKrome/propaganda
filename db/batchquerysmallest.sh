#!/bin/zsh
startdate="-1"
#queries=${#${(f)$(<$0)}}
queries=3
echo "Total queries: $queries"
$1 "$startdate" skorea "South Korea" "Summarize news about South Korea. Analize actions and motivations."
echo "$LINENO / $queries"
$1 "$startdate" nkorea "North Korea" "Summarize news about North Korea. Analize actions and motivations."
echo "$LINENO / $queries"
$1 "$startdate" gaza "Gaza" "Summarize events in Gaza. Analize actions and motivations."
