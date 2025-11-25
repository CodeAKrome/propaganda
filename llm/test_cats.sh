head -n 300 titles_ids.txt > tmp_cats.txt
cat prompt/sort_titles.txt tmp_cats.txt | ollama run $1 | tee tmp
