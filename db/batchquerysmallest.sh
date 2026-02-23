#!/bin/zsh
startdate=$2
queries=3
current=0

echo "Total queries: $queries"

((current++))
echo "$current / $queries"
# echo "$current / $queries"
# ((current++))
# echo "$current / $queries"
# ((current++))
# echo "$current / $queries"
#$1 "$startdate" venezuela "Venezuela" "Summarize US involvement in Venezuela. Analize actions and motivations."
#$1 "$startdate" maduro "Maduro" "Summarize legal actions involving Maduro. Analize actions and motivations." "+legal,court,trial,arrest,indictment,charges,prosecution"
$1 "$startdate" ai "Artificial Intelligence,AI" "Summarize current events involving Artificial Intelligence. Analize actions, economic impacts and motivations." "+Artificial Intelligence,AI"
$1 "$startdate" mexico "Mexico,Puerto Vallarta,Jalisco" "Summarize current events in Mexico and Puerto Vallarta" "+Mexico"
