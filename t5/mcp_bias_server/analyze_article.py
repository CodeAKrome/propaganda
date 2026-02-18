#!/usr/bin/env python3
"""Analyze the right.txt article for bias."""

import os
import sys

# Set model path
os.environ["BIAS_MODEL_PATH"] = "/Users/kyle/hub/propaganda/t5/bias-detector-output"

from mcp_bias_server.bias_engine import BiasEngine

# Read the article
article_path = "/Users/kyle/hub/propaganda/llm/prompt/right.txt"
with open(article_path, "r") as f:
    text = f.read()

print("=" * 60)
print("BIAS ANALYSIS: llm/prompt/right.txt")
print("=" * 60)
print(f"Article length: {len(text)} characters")
print()

# Analyze
engine = BiasEngine()
result = engine.analyze(text)

print("DIRECTION SCORES (Political Leaning):")
print(f"  Left (L):   {result.direction['L']:.2f}")
print(f"  Center (C): {result.direction['C']:.2f}")
print(f"  Right (R):  {result.direction['R']:.2f}")
print()

print("DEGREE SCORES (Bias Intensity):")
print(f"  Low (L):    {result.degree['L']:.2f}")
print(f"  Medium (M): {result.degree['M']:.2f}")
print(f"  High (H):   {result.degree['H']:.2f}")
print()

print("REASONING:")
print(f"  {result.reason}")
print()

# Interpretation
dir_max = max(result.direction, key=result.direction.get)
deg_max = max(result.degree, key=result.degree.get)

dir_label = {"L": "Left", "C": "Center", "R": "Right"}[dir_max]
deg_label = {"L": "Low", "M": "Medium", "H": "High"}[deg_max]

print("INTERPRETATION:")
print(f"  Primary leaning: {dir_label} ({result.direction[dir_max]:.0%})")
print(f"  Bias intensity:  {deg_label} ({result.degree[deg_max]:.0%})")
print("=" * 60)