#!/usr/bin/env python3
"""
Bias MCP Server - Standalone Test Script

This script tests the bias-mcp-server installation and demonstrates usage.

INSTALLATION:
    cd t5/mcp_bias_server
    pip install -e .

RUN:
    python test_bias_server.py

Or with custom model path:
    BIAS_MODEL_PATH=/path/to/bias-detector-output python test_bias_server.py

EXPECTED OUTPUT:
    - Model loads successfully
    - Single text analysis returns direction/degree/reason
    - Batch analysis processes multiple texts
    - Model info shows device and configuration
"""

import os
import sys
import json
from pathlib import Path


def find_model_path():
    """Find the bias-detector-output directory."""
    # Check environment variable first
    env_path = os.getenv("BIAS_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    
    # Try relative paths from this script
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "bias-detector-output",
        script_dir.parent / "bias-detector-output",
        script_dir / "src" / "mcp_bias_server" / "bias-detector-output",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # Default fallback
    return str(script_dir.parent / "bias-detector-output")


def test_import():
    """Test that the module can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Module Import")
    print("=" * 60)
    
    try:
        from mcp_bias_server.bias_engine import BiasEngine
        print("  [PASS] BiasEngine imported successfully")
        return True, BiasEngine
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        print("\n  FIX: Install the package:")
        print("    cd t5/mcp_bias_server")
        print("    pip install -e .")
        return False, None


def test_model_path(model_path):
    """Test that the model path exists and has required files."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Files")
    print("=" * 60)
    print(f"  Model path: {model_path}")
    
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    
    path = Path(model_path)
    if not path.exists():
        print(f"  [FAIL] Path does not exist: {model_path}")
        print("\n  FIX: Set BIAS_MODEL_PATH environment variable:")
        print(f"    export BIAS_MODEL_PATH=/path/to/bias-detector-output")
        return False
    
    all_exist = True
    for filename in required_files:
        file_path = path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  [PASS] {filename} ({size:,} bytes)")
        else:
            print(f"  [FAIL] Missing: {filename}")
            all_exist = False
    
    return all_exist


def test_single_analysis(BiasEngine, model_path):
    """Test single text analysis."""
    print("\n" + "=" * 60)
    print("TEST 3: Single Text Analysis")
    print("=" * 60)
    
    try:
        engine = BiasEngine(model_path=model_path, lazy_load=True)
        
        test_text = "The president announced new economic policies today."
        print(f"  Input: \"{test_text}\"")
        print("  Loading model (first inference may take 10-30 seconds)...")
        
        result = engine.analyze(test_text)
        
        print(f"  [PASS] Analysis complete")
        print(f"\n  Direction: L={result.direction['L']:.2f}, C={result.direction['C']:.2f}, R={result.direction['R']:.2f}")
        print(f"  Degree:    L={result.degree['L']:.2f}, M={result.degree['M']:.2f}, H={result.degree['H']:.2f}")
        print(f"  Reason:    {result.reason[:80]}...")
        
        return True, engine
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batch_analysis(engine):
    """Test batch analysis."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Analysis")
    print("=" * 60)
    
    if engine is None:
        print("  [SKIP] No engine available")
        return False
    
    try:
        texts = [
            "The senator's proposal received bipartisan support.",
            "The radical left continues to push their extreme agenda.",
            "Markets rallied on positive economic news.",
        ]
        
        print(f"  Input: {len(texts)} texts")
        results = engine.analyze_batch(texts)
        
        print("  [PASS] Batch analysis complete\n")
        for i, (text, result) in enumerate(zip(texts, results), 1):
            print(f"  [{i}] {text[:50]}...")
            print(f"      Dir: L={result.direction['L']:.1f} C={result.direction['C']:.1f} R={result.direction['R']:.1f}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_model_info(engine):
    """Test model info retrieval."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Information")
    print("=" * 60)
    
    if engine is None:
        print("  [SKIP] No engine available")
        return False
    
    try:
        info = engine.get_model_info()
        print("  [PASS] Model info retrieved\n")
        print(json.dumps(info, indent=4))
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("BIAS MCP SERVER - INSTALLATION TEST")
    print("=" * 60)
    
    model_path = find_model_path()
    print(f"\nUsing model path: {model_path}")
    
    results = []
    
    # Test 1: Import
    success, BiasEngine = test_import()
    results.append(("Module Import", success))
    if not success:
        print("\n" + "=" * 60)
        print("ABORTING: Cannot proceed without module import")
        print("=" * 60)
        sys.exit(1)
    
    # Test 2: Model files
    success = test_model_path(model_path)
    results.append(("Model Files", success))
    
    # Test 3: Single analysis
    success, engine = test_single_analysis(BiasEngine, model_path)
    results.append(("Single Analysis", success))
    
    # Test 4: Batch analysis
    success = test_batch_analysis(engine)
    results.append(("Batch Analysis", success))
    
    # Test 5: Model info
    success = test_model_info(engine)
    results.append(("Model Info", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  All tests passed! The bias detector is working correctly.")
        return 0
    else:
        print("\n  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())