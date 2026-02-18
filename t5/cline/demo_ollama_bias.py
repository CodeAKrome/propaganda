#!/usr/bin/env python3
"""
demo_ollama_bias.py - Install and use bias detection with Ollama for inference.

This script demonstrates how to:
1. Install required packages
2. Use Ollama for bias detection inference
3. Run bias analysis from Python

Usage:
    python demo_ollama_bias.py

Requirements:
    pip install ollama
    ollama serve (or have Ollama running)
    ollama pull llama3.2 (or your preferred model)
"""

import json
import os
import sys
import subprocess
from typing import Optional
from dataclasses import dataclass


@dataclass
class BiasResult:
    """Bias analysis result."""
    direction: dict  # {"L": float, "C": float, "R": float}
    degree: dict     # {"L": float, "M": float, "H": float}
    reason: str
    model: str
    
    def to_dict(self) -> dict:
        return {
            "dir": self.direction,
            "deg": self.degree,
            "reason": self.reason,
            "model": self.model
        }


class OllamaBiasAnalyzer:
    """
    Bias analyzer using Ollama for inference.
    
    Uses a local LLM via Ollama to analyze text for political bias.
    """
    
    BIAS_PROMPT = """Analyze the following text for political bias. 

Text to analyze:
{text}

Provide your analysis in the following JSON format ONLY (no other text):
{{
    "direction": {{
        "L": <0.0-1.0 score for Left/progressive leaning>,
        "C": <0.0-1.0 score for Center/neutral>,
        "R": <0.0-1.0 score for Right/conservative leaning>
    }},
    "degree": {{
        "L": <0.0-1.0 score for Low bias>,
        "M": <0.0-1.0 score for Medium bias>,
        "H": <0.0-1.0 score for High bias>
    }},
    "reason": "<brief explanation of the classification>"
}}

The direction scores should sum to approximately 1.0.
The degree scores should sum to approximately 1.0.

Consider:
- Language framing and word choice
- Presence of partisan terminology
- Balance of perspectives presented
- Factual vs opinion content
- Emotional language and rhetoric

JSON response only:"""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        """
        Initialize the Ollama bias analyzer.
        
        Args:
            model: Ollama model name (default: llama3.2)
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self._client = None
        
    def _get_client(self):
        """Get or create the Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host)
        return self._client
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            models = client.list()
            model_names = [m['model'] for m in models.get('models', [])]
            
            # Check if our model or a base version is available
            model_base = self.model.split(':')[0]
            available = any(m.startswith(model_base) for m in model_names)
            
            if not available:
                print(f"  Model '{self.model}' not found. Available models: {model_names}")
                print(f"  Run: ollama pull {self.model}")
                return False
            
            return True
        except Exception as e:
            print(f"  Error connecting to Ollama: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return False
    
    def analyze(self, text: str) -> BiasResult:
        """
        Analyze text for political bias.
        
        Args:
            text: The text to analyze
            
        Returns:
            BiasResult with direction, degree, and reasoning
        """
        client = self._get_client()
        
        # Call Ollama
        response = client.generate(
            model=self.model,
            prompt=self.BIAS_PROMPT.format(text=text),
            format="json",  # Request JSON output
            options={
                "temperature": 0.1,  # Low temperature for consistent results
                "num_predict": 256,  # Limit output length
            }
        )
        
        # Parse the response
        response_text = response.get('response', '{}')
        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text}")
        
        # Normalize scores
        direction = data.get('direction', {'L': 0.33, 'C': 0.34, 'R': 0.33})
        degree = data.get('degree', {'L': 0.33, 'M': 0.34, 'H': 0.33})
        reason = data.get('reason', 'No reasoning provided')
        
        # Ensure scores sum to ~1.0
        direction = self._normalize_scores(direction)
        degree = self._normalize_scores(degree)
        
        return BiasResult(
            direction=direction,
            degree=degree,
            reason=reason,
            model=self.model
        )
    
    def analyze_batch(self, texts: list) -> list[BiasResult]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of BiasResult objects
        """
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results
    
    def _normalize_scores(self, scores: dict) -> dict:
        """Normalize scores to sum to 1.0."""
        total = sum(scores.values())
        if total > 0:
            return {k: round(v / total, 2) for k, v in scores.items()}
        return scores


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Install ollama package
    try:
        import ollama
        print("  ollama package already installed")
    except ImportError:
        print("  Installing ollama package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
        print("  Installed ollama successfully")
    
    print()


def main():
    """Main demo function."""
    print("=" * 70)
    print("Bias Detection with Ollama - Python Demo")
    print("=" * 70)
    print()
    
    # Step 1: Install dependencies
    print("Step 1: Installing dependencies...")
    install_dependencies()
    
    # Step 2: Initialize analyzer
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    print(f"Step 2: Initializing Ollama bias analyzer...")
    print(f"  Model: {model}")
    print(f"  Host: http://localhost:11434")
    print()
    
    analyzer = OllamaBiasAnalyzer(model=model)
    
    # Step 3: Check Ollama
    print("Step 3: Checking Ollama connection...")
    if not analyzer.check_ollama():
        print()
        print("  Please ensure Ollama is running and the model is available:")
        print("    1. Start Ollama: ollama serve")
        print(f"    2. Pull the model: ollama pull {model}")
        print()
        print("  Continuing with demo (may fail)...")
    else:
        print("  Ollama is running and model is available!")
    print()
    
    # Step 4: Analyze sample texts
    print("Step 4: Analyzing sample texts...")
    print()
    
    sample_texts = [
        "The president announced today a sweeping new economic policy that critics argue favors large corporations over working families. Supporters say the measure will create jobs and stimulate growth.",
        
        "The radical left continues to push their extreme agenda on hardworking taxpayers, destroying our economy and undermining traditional values.",
        
        "Markets rallied today as investors reacted positively to the Federal Reserve's decision to maintain current interest rates. The S&P 500 gained 1.2% on the day.",
        
        "In a stunning display of incompetence, the administration once again failed to address the crisis at our southern border, leaving American communities to deal with the consequences.",
        
        "The study found that 62% of respondents supported the new policy, with support crossing party lines. Researchers noted that the findings suggest a shift in public opinion."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}:")
        print(f"  \"{text[:80]}...\"")
        print()
        
        try:
            result = analyzer.analyze(text)
            
            print(f"  Direction:")
            print(f"    Left (L):   {result.direction['L']:.2f}")
            print(f"    Center (C): {result.direction['C']:.2f}")
            print(f"    Right (R):  {result.direction['R']:.2f}")
            print()
            print(f"  Degree:")
            print(f"    Low (L):    {result.degree['L']:.2f}")
            print(f"    Medium (M): {result.degree['M']:.2f}")
            print(f"    High (H):   {result.degree['H']:.2f}")
            print()
            print(f"  Reason: {result.reason}")
            
        except Exception as e:
            print(f"  Error analyzing text: {e}")
        
        print()
        print("-" * 70)
        print()
    
    # Step 5: Batch analysis
    print("Step 5: Batch analysis example...")
    batch_texts = [
        "Congress passed the bill with bipartisan support.",
        "The corrupt mainstream media refuses to report the truth."
    ]
    
    try:
        results = analyzer.analyze_batch(batch_texts)
        for text, result in zip(batch_texts, results):
            dir_label = max(result.direction.keys(), key=lambda k: result.direction[k])
            deg_label = max(result.degree.keys(), key=lambda k: result.degree[k])
            print(f"  \"{text[:50]}...\"")
            print(f"    -> {dir_label}-leaning, {deg_label} bias")
            print()
    except Exception as e:
        print(f"  Error in batch analysis: {e}")
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("To use with a different model:")
    print(f"  export OLLAMA_MODEL=llama3.1:70b")
    print(f"  python demo_ollama_bias.py")


if __name__ == "__main__":
    main()