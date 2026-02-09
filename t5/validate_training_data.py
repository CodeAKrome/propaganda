#!/usr/bin/env python
"""
Quick validation script to verify train.json compatibility with the MPS-optimized bias detector
"""

import json
from pathlib import Path

def validate_training_data(filepath):
    """Validate that the JSON file matches the expected format."""
    
    print("="*80)
    print("TRAINING DATA VALIDATION")
    print("="*80)
    
    # Load the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ File loaded successfully: {filepath}")
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON format: {e}")
        return False
    
    # Check it's a list
    if not isinstance(data, list):
        print(f"✗ Expected JSON array, got {type(data).__name__}")
        return False
    
    print(f"✓ Valid JSON array with {len(data)} entries")
    
    # Validate each entry
    errors = []
    warnings = []
    
    for i, entry in enumerate(data):
        # Check structure
        if not isinstance(entry, dict):
            errors.append(f"Entry {i}: Not an object")
            continue
            
        # Check article
        if "article" not in entry:
            errors.append(f"Entry {i}: Missing 'article' field")
            continue
        if not isinstance(entry["article"], str) or not entry["article"].strip():
            errors.append(f"Entry {i}: 'article' must be non-empty string")
            continue
            
        # Check label
        if "label" not in entry:
            errors.append(f"Entry {i}: Missing 'label' field")
            continue
        if not isinstance(entry["label"], dict):
            errors.append(f"Entry {i}: 'label' must be an object")
            continue
            
        label = entry["label"]
        
        # Check required fields
        for field in ["dir", "deg", "reason"]:
            if field not in label:
                errors.append(f"Entry {i}: Missing '{field}' in label")
                continue
        
        # Validate 'dir' probabilities
        if "dir" in label:
            dir_probs = label["dir"]
            if not isinstance(dir_probs, dict):
                errors.append(f"Entry {i}: 'dir' must be an object")
            else:
                required_keys = ["L", "C", "R"]
                for key in required_keys:
                    if key not in dir_probs:
                        errors.append(f"Entry {i}: 'dir' missing '{key}' key")
                
                # Check sum (should be ~1.0)
                if all(k in dir_probs for k in required_keys):
                    total = sum(dir_probs.values())
                    if abs(total - 1.0) > 0.1:
                        warnings.append(f"Entry {i}: 'dir' probabilities sum to {total:.2f} (expected ~1.0)")
        
        # Validate 'deg' probabilities
        if "deg" in label:
            deg_probs = label["deg"]
            if not isinstance(deg_probs, dict):
                errors.append(f"Entry {i}: 'deg' must be an object")
            else:
                required_keys = ["L", "M", "H"]
                for key in required_keys:
                    if key not in deg_probs:
                        errors.append(f"Entry {i}: 'deg' missing '{key}' key")
                
                # Check sum
                if all(k in deg_probs for k in required_keys):
                    total = sum(deg_probs.values())
                    if abs(total - 1.0) > 0.1:
                        warnings.append(f"Entry {i}: 'deg' probabilities sum to {total:.2f} (expected ~1.0)")
        
        # Check reason
        if "reason" in label and not isinstance(label["reason"], str):
            errors.append(f"Entry {i}: 'reason' must be a string")
    
    # Print results
    print("\n" + "-"*80)
    print("VALIDATION RESULTS")
    print("-"*80)
    
    if errors:
        print(f"\n✗ Found {len(errors)} ERROR(S):")
        for err in errors[:10]:  # Show first 10
            print(f"  • {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✓ No errors found!")
    
    if warnings:
        print(f"\n⚠ Found {len(warnings)} WARNING(S):")
        for warn in warnings[:10]:  # Show first 10
            print(f"  • {warn}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    # Sample data preview
    if data and not errors:
        print("\n" + "-"*80)
        print("SAMPLE DATA PREVIEW (First Entry)")
        print("-"*80)
        print(f"Article: {data[0]['article'][:150]}...")
        print(f"\nLabel:")
        print(f"  Direction: {data[0]['label']['dir']}")
        print(f"  Degree:    {data[0]['label']['deg']}")
        print(f"  Reason:    {data[0]['label']['reason'][:100]}...")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total entries:    {len(data)}")
    print(f"Valid entries:    {len(data) - len(errors)}")
    print(f"Errors:           {len(errors)}")
    print(f"Warnings:         {len(warnings)}")
    
    if errors:
        print("\n⚠ Your data has errors that will cause entries to be skipped during training.")
        print("   The script will continue but skip invalid entries.")
    else:
        print("\n✓ Your data is fully compatible with the bias detector!")
    
    print("="*80 + "\n")
    
    return len(errors) == 0


if __name__ == "__main__":
    # Test with the provided sample
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to looking for train.json
        filepath = "train.json"
        if not Path(filepath).exists():
            print("Usage: python validate_data.py <path_to_train.json>")
            print("Or place train.json in the current directory")
            sys.exit(1)
    
    validate_training_data(filepath)
