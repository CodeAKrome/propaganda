#!/usr/bin/env python
"""
Diagnostic tool for bias detector model issues
"""

import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

def diagnose_model(model_path: str = './bias-detector-output', base_model: str = 't5-large'):
    """Run diagnostic checks on the bias detector model."""
    
    print("="*80)
    print("BIAS DETECTOR MODEL DIAGNOSTICS")
    print("="*80)
    
    # 1. Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n1. Device: {device}")
    
    # 2. Load model
    print(f"\n2. Loading model from: {model_path}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(base_model)
        base = T5ForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)
        model.to(device)
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # 3. Test with simple input
    print("\n3. Testing with simple article...")
    test_article = "The Senate voted on the bill today."
    input_text = f"classify political bias as json: {test_article}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    print(f"   Input: {test_article}")
    print(f"   Tokenized length: {inputs['input_ids'].shape[1]} tokens")
    
    # 4. Test different generation strategies
    print("\n4. Testing generation strategies...")
    
    strategies = [
        ("Default (Greedy)", {"max_length": 300, "num_beams": 1}),
        ("Beam Search (n=3)", {"max_length": 300, "num_beams": 3}),
        ("Beam Search (n=5)", {"max_length": 300, "num_beams": 5}),
        ("Beam Search + No Repeat", {"max_length": 300, "num_beams": 5, "no_repeat_ngram_size": 3}),
    ]
    
    for name, params in strategies:
        print(f"\n   Strategy: {name}")
        print(f"   Parameters: {params}")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **params)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Raw output: {result[:200]}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(result)
            print(f"   ✓ Valid JSON!")
            print(f"   Structure: {json.dumps(parsed, indent=6)}")
        except json.JSONDecodeError as e:
            print(f"   ✗ JSON parsing failed: {e}")
            print(f"   Full output: {result}")
    
    # 5. Check training data format
    print("\n5. Checking expected output format...")
    expected = {
        "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
        "deg": {"L": 0.7, "M": 0.25, "H": 0.05},
        "reason": "Example reason"
    }
    print(f"   Expected format: {json.dumps(expected, indent=6)}")
    
    # 6. Test with training-style input
    print("\n6. Testing with longer, realistic article...")
    long_article = """
    The Senate passed a comprehensive infrastructure bill today with bipartisan support.
    Republicans and Democrats both praised the legislation, which includes funding for
    roads, bridges, and broadband internet access. The bill now moves to the House.
    """
    
    input_text = f"classify political bias as json: {long_article.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Raw output: {result[:300]}")
    
    try:
        parsed = json.loads(result)
        print(f"   ✓ Valid JSON!")
        print(f"   Result: {json.dumps(parsed, indent=6)}")
    except:
        print(f"   ✗ Not valid JSON")
        print(f"   Full output: {result}")
    
    # 7. Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("\nIf you're seeing article text instead of JSON:")
    print("  1. Model may not be properly trained")
    print("  2. Training data format might be incorrect")
    print("  3. Model might be overfitting to input")
    print("  4. LoRA weights might not have loaded correctly")
    print("\nRecommended fixes:")
    print("  1. Re-train with more epochs (15-30)")
    print("  2. Verify training data has correct format")
    print("  3. Try larger model (t5-large instead of t5-small)")
    print("  4. Use num_beams=5 and no_repeat_ngram_size=3")
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else './bias-detector-output'
    base_model = sys.argv[2] if len(sys.argv) > 2 else 't5-large'
    
    diagnose_model(model_path, base_model)
