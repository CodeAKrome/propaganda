pip install transformers==5.0.0 tokenizers==0.22.2 torch nltk --force-reinstall
rm -rf ~/.cache/huggingface/hub/models--zilliz--semantic-highlight-bilingual-v1
rm -rf ~/.cache/huggingface/modules/transformers_modules/zilliz
python orig_colab_style.py
