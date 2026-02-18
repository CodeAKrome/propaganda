#!/usr/bin/env python
"""
Output only semantically relevant text - FIXED with class-level monkey patch
"""

import os
import shutil
import nltk
import zipfile

# ============================================
# NLTK SETUP WORKAROUNDS
# ============================================

default_nltk_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.environ['NLTK_DATA'] = default_nltk_path
if default_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, default_nltk_path)

if os.path.exists(default_nltk_path):
    try:
        shutil.rmtree(default_nltk_path)
    except OSError:
        pass

os.makedirs(default_nltk_path, exist_ok=True)
nltk.download('punkt', download_dir=default_nltk_path, quiet=True)

punkt_tab_english_dir = os.path.join(default_nltk_path, 'tokenizers', 'punkt_tab', 'english')
if not os.path.exists(punkt_tab_english_dir):
    os.makedirs(punkt_tab_english_dir, exist_ok=True)
    for fname in ['collocations.tab', 'sent_starters.txt', 'abbrev_types.txt', 'ortho_context.tab']:
        with open(os.path.join(punkt_tab_english_dir, fname), 'w') as f:
            pass

# ============================================
# CRITICAL FIX: Patch TokenizersBackend class BEFORE loading model
# ============================================

from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Store original __getattr__ to chain to it
_original_getattr = PreTrainedTokenizerBase.__getattr__

def patched_getattr(self, key):
    # If asking for build_inputs_with_special_tokens, return our implementation
    if key == 'build_inputs_with_special_tokens':
        def build_inputs_with_special_tokens(token_ids_0, token_ids_1=None):
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + token_ids_1 + sep
        return build_inputs_with_special_tokens
    # Otherwise, use original __getattr__
    return _original_getattr(self, key)

# Apply the monkey patch to the base class
PreTrainedTokenizerBase.__getattr__ = patched_getattr

# Also patch specific tokenizer class if it exists
try:
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    # Try to patch TokenizersBackend specifically if accessible
    import transformers
    if hasattr(transformers, 'TokenizersBackend'):
        transformers.TokenizersBackend.build_inputs_with_special_tokens = lambda self, token_ids_0, token_ids_1=None: (
            [self.cls_token_id] + token_ids_0 + [self.sep_token_id] if token_ids_1 is None 
            else [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
        )
except:
    pass

# ============================================
# LOAD MODEL
# ============================================

model = AutoModel.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True
)

question = "What are the symptoms of dehydration?"
context = """
Dehydration occurs when your body loses more fluid than you take in.
Common signs include feeling thirsty and having a dry mouth.
The human body is composed of about 60% water.
Dark yellow urine and infrequent urination are warning signs.
Water is essential for many bodily functions.
Dizziness, fatigue, and headaches can indicate severe dehydration.
Drinking 8 glasses of water daily is often recommended.
"""

result = model.process(
    question=question,
    context=context,
    threshold=0.5,
    return_sentence_metrics=True,
)

highlighted = result["highlighted_sentences"]
print(f"Highlighted {len(highlighted)} sentences:")
for i, sent in enumerate(highlighted, 1):
    print(f"  {i}. {sent}")
print(f"\nTotal sentences in context: {len(context.strip().split('.')) - 1}")

if "sentence_probabilities" in result:
    probs = result["sentence_probabilities"]
    print(f"\nSentence probabilities: {probs}")