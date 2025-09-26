cat llms.txt| perl -e 'while (<>) {chomp; print "echo \"\\n--\\n$_\\n--\\n\"\n";print "cat tmp_q.txt tmp_vec.txt | ollama run --verbose $_\n";}' > runllms.sh
