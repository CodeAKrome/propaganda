rhtest() {
    cat test/rh.txt | ollama run --hidethinking $1 --verbose | tee test/${1}.txt
}
lhtest() {
    cat test/lh.txt | ollama run --hidethinking $1 --verbose | tee test/${1}.txt
}
cltest() {
    cat test/cl.txt | ollama run --hidethinking $1 --verbose | tee test/${1}.txt
}
