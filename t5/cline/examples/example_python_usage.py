#!/usr/bin/env python3
"""
Example: Using bias-mcp-server as a Python package.

This demonstrates how to use the BiasEngine directly from Python code
without the MCP server layer.

Installation:
    pip install bias-mcp-server

Usage:
    python example_python_usage.py
"""

import os
import json
from pathlib import Path
from mcp_bias_server.bias_engine import BiasEngine


def main():
    # Configure the engine via environment variables or constructor
    # Default path is relative to the t5/ directory
    script_dir = Path(__file__).parent.parent  # Go up from examples/ to mcp_bias_server/
    project_root = script_dir.parent  # Go up to t5/
    default_model_path = project_root / "bias-detector-output"
    
    model_path = os.getenv("BIAS_MODEL_PATH", str(default_model_path))
    base_model = os.getenv("BIAS_BASE_MODEL", "t5-large")
    
    print("=" * 60)
    print("Bias MCP Server - Python Usage Example")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Base model: {base_model}")
    print()
    
    # Create the engine (model loads lazily on first inference)
    engine = BiasEngine(
        model_path=model_path,
        base_model_name=base_model,
        lazy_load=True
    )
    
    # Example 1: Analyze a single text
    print("-" * 60)
    print("Example 1: Single Text Analysis")
    print("-" * 60)
    
    sample_text = """
    The president announced today a sweeping new economic policy that critics 
    argue favors large corporations over working families. Supporters say the 
    measure will create jobs and stimulate growth, while opponents contend it 
    will widen the wealth gap and undermine environmental protections.
    """
    
    print(f"Text: {sample_text.strip()[:100]}...")
    print()
    
    result = engine.analyze(sample_text)
    
    print("Result:")
    print(json.dumps(result.to_dict(), indent=2))
    print()
    
    # Example 2: Batch analysis
    print("-" * 60)
    print("Example 2: Batch Analysis")
    print("-" * 60)
    
    texts = [
        "The senator's proposal received bipartisan support in Congress today.",
        "The radical left continues to push their extreme agenda on taxpayers.",
        "Markets rallied as investors reacted positively to the Fed's announcement."
    ]
    
    results = engine.analyze_batch(texts)
    
    for i, (text, result) in enumerate(zip(texts, results), 1):
        print(f"\nText {i}: {text[:60]}...")
        print(f"  Direction: L={result.direction['L']:.2f}, C={result.direction['C']:.2f}, R={result.direction['R']:.2f}")
        print(f"  Degree:    L={result.degree['L']:.2f}, M={result.degree['M']:.2f}, H={result.degree['H']:.2f}")
    
    # Example 3: Get model information
    print()
    print("-" * 60)
    print("Example 3: Model Information")
    print("-" * 60)
    
    info = engine.get_model_info()
    print(json.dumps(info, indent=2))
    
    print()
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
