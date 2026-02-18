import os
import nltk

# Setup NLTK (from Colab notebook)
default_nltk_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.environ['NLTK_DATA'] = default_nltk_path

if default_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, default_nltk_path)

# Download punkt if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', download_dir=default_nltk_path)

# Now load the model
from transformers import AutoModel

print("Loading model...")
model = AutoModel.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True
)
print("✓ Model loaded successfully!")

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

print("\nProcessing...")
result = model.process(
    question=question,
    context=context,
    threshold=0.5,
    return_sentence_metrics=True,
)

highlighted = result["highlighted_sentences"]
print(f"\n✓ Highlighted {len(highlighted)} sentences:")
for i, sent in enumerate(highlighted, 1):
    print(f"  {i}. {sent}")

if "sentence_probabilities" in result:
    probs = result["sentence_probabilities"]
    print(f"\n✓ Sentence probabilities:")
    for i, prob in enumerate(probs, 1):
        print(f"  Sentence {i}: {prob:.4f}")
