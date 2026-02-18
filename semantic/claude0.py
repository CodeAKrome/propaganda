from transformers import AutoModel, AutoTokenizer
import warnings

# Suppress the trust_remote_code warning since we're explicitly setting it
warnings.filterwarnings('ignore', category=FutureWarning)

# Load the model with a workaround for the tokenizer issue
try:
    # First, try loading the tokenizer separately to ensure it's initialized properly
    tokenizer = AutoTokenizer.from_pretrained(
        "zilliz/semantic-highlight-bilingual-v1",
        trust_remote_code=True
    )
    
    # Then load the model
    model = AutoModel.from_pretrained(
        "zilliz/semantic-highlight-bilingual-v1",
        trust_remote_code=True
    )
    
    # If the model doesn't have the tokenizer properly set, set it manually
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        model.tokenizer = tokenizer
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: downgrade transformers or use a different loading method
    import sys
    print(f"Current transformers version: {__import__('transformers').__version__}")
    print("\nThis model may require transformers<4.36.0")
    print("Try: pip install 'transformers<4.36.0'")
    sys.exit(1)

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
    # language="en",  # Language can be auto-detected, or explicitly specified
    return_sentence_metrics=True,  # Enable sentence probabilities
)

highlighted = result["highlighted_sentences"]
print(f"Highlighted {len(highlighted)} sentences:")
for i, sent in enumerate(highlighted, 1):
    print(f"  {i}. {sent}")
print(f"\nTotal sentences in context: {len(context.strip().split('.')) - 1}")

# Print sentence probabilities if available
if "sentence_probabilities" in result:
    probs = result["sentence_probabilities"]
    print(f"\nSentence probabilities: {probs}")
