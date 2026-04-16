rhtest() {
    cat test/rh.txt | ollama run $1 --verbose | tee test/${1}.txt
}
