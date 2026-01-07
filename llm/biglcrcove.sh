#./lcr_cove_llms.sh prompt/lcr_CoVe.txt prompt/left.txt | tee out/lcr_cove_l.txt
#./lcr_cove_llms.sh prompt/lcr_CoVe.txt prompt/center.txt | tee out/lcr_cove_c.txt
#./lcr_cove_llms.sh prompt/lcr_CoVe.txt prompt/right.txt | tee out/lcr_cove_r.txt
#./lcr_cove_llms.sh prompt/claudeopus_CoVe.txt prompt/left.txt | tee out/claudeopus_CoVe_l.txt
#./lcr_cove_llms.sh prompt/claudeopus_CoVe.txt prompt/center.txt | tee out/claudeopus_CoVe_c.txt
#./lcr_cove_llms.sh prompt/claudeopus_CoVe.txt prompt/right.txt | tee out/claudeopus_CoVe_r.txt
#./lcr_cove_llms.sh prompt/gpt51highngrok41think.txt prompt/left.txt | tee out/gpt51highngrok41think_l.txt
#./lcr_cove_llms.sh prompt/gpt51highngrok41think.txt prompt/center.txt | tee out/gpt51highngrok41think_c.txt
#./lcr_cove_llms.sh prompt/gpt51highngrok41think.txt prompt/right.txt | tee out/gpt51highngrok41think_r.txt
#./lcr_cove_llms.sh prompt/claudesonnet4520250929.txt prompt/left.txt | tee out/claudesonnet4520250929_l.txt
#./lcr_cove_llms.sh prompt/claudesonnet4520250929.txt prompt/center.txt | tee out/claudesonnet4520250929_c.txt
#./lcr_cove_llms.sh prompt/claudesonnet4520250929.txt prompt/right.txt | tee out/claudesonnet4520250929_r.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudeopus_CoVe.txt prompt/left.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/opus_l.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudeopus_CoVe.txt prompt/center.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/opus_c.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudeopus_CoVe.txt prompt/right.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/opus_r.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/gpt51highngrok41think.txt prompt/left.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/gpt51_l.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/gpt51highngrok41think.txt prompt/center.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/gpt51_c.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/gpt51highngrok41think.txt prompt/right.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/gpt51_r.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudesonnet4520250929.txt prompt/left.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/son_l.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudesonnet4520250929.txt prompt/center.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/son_c.txt
xargs -I {} sh -c 'echo "=== Running model: {} ==="; cat prompt/claudesonnet4520250929.txt prompt/right.txt | ollama run --verbose --hidethinking {} 2>&1' < llms.txt | tee out/son_r.txt
