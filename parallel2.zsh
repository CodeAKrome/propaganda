#!/usr/bin/env zsh

# Run 2 commands in parallel, block until both are done and print return codes.

# Read two commands (one per line) from stdin
read -r cmd1
read -r cmd2

# Run both in background
eval "$cmd1" & p1=$!
eval "$cmd2" & p2=$!

# Wait for both to finish
wait $p1
st1=$?
wait $p2
st2=$?

printf 'cmd1 exit: %d\n' "$st1"
printf 'cmd2 exit: %d\n' "$st2"
