cat llms.txt| perl -e 'while (<>) {chomp; print "echo \"\\n--\\n$_\\n--\\n\"\n";print "cat testprompt.txt | ollama run --hidethinking --verbose $_\n";}' > runllms.sh
