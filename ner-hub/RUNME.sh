#!/usr/bin/env zsh

go build

# Default value for start-date
START_DATE="-3"

# If a command-line argument is provided, override the default
if [[ -n "$1" ]]; then
  START_DATE="$1"
fi

# Run the command
./multiflair --start-date "$START_DATE" endpoints.tsv
