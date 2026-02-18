#!/usr/bin/env python3
"""
demo_local_llm_bias.py - Install and use bias detection with local LLMs.

This script demonstrates how to:
1. Install required packages
2. Use Ollama or LM Studio for bias detection inference
3. Run bias analysis from Python

Supported Backends:
- Ollama (default: http://localhost:11434)
- LM Studio (default: http://localhost:1234/v1)

Usage:
    python demo_local_llm_bias.py

Requirements:
    pip install ollama openai
    ollama serve (or have Ollama running)
    ollama pull llama3.2 (or your preferred model)
    
    For LM Studio:
    - Start LM Studio
    - Load a model
    - Start the local server (default port 1234)
"""

import json
import os
import sys
import subprocess
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BiasResult:
    """Bias analysis result."""
    direction: dict  # {"L": float, "C": float, "R": float}
    degree: dict     # {"L": float, "M": float, "H": float}
    reason: str
    model: str
    backend: str
    
    def to_dict(self) -> dict:
        return {
            "dir": self.direction,
            "deg": self.degree,
            "reason": self.reason,
            "model": self.model,
            "backend": self.backend
        }


class BiasAnalyzerBase(ABC):
    """Abstract base class for bias analyzers."""
    
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

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if the backend is available."""
        pass
    
    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Generate response from the LLM."""
        pass
    
    def analyze(self, text: str) -> BiasResult:
        """
        Analyze text for political bias.
        
        Args:
            text: The text to analyze
            
        Returns:
            BiasResult with direction, degree, and reasoning
        """
        prompt = self.BIAS_PROMPT.format(text=text)
        response_text = self._generate(prompt)
        
        # Parse the response
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
            model=self.model,
            backend=self.backend_name
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
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        pass


class OllamaBiasAnalyzer(BiasAnalyzerBase):
    """
    Bias analyzer using Ollama for inference.
    
    Uses a local LLM via Ollama to analyze text for political bias.
    """
    
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        """
        Initialize the Ollama bias analyzer.
        
        Args:
            model: Ollama model name (default: llama3.2)
            host: Ollama server URL
        """
        self._model = model
        self.host = host
        self._client = None
        
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def backend_name(self) -> str:
        return "ollama"
    
    def _get_client(self):
        """Get or create the Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host)
        return self._client
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            models = client.list()
            model_names = [m['model'] for m in models.get('models', [])]
            
            # Check if our model or a base version is available
            model_base = self._model.split(':')[0]
            available = any(m.startswith(model_base) for m in model_names)
            
            if not available:
                print(f"  Model '{self._model}' not found. Available models: {model_names}")
                print(f"  Run: ollama pull {self._model}")
                return False
            
            return True
        except Exception as e:
            print(f"  Error connecting to Ollama: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return False
    
    def _generate(self, prompt: str) -> str:
        """Generate response from Ollama."""
        client = self._get_client()
        
        response = client.generate(
            model=self._model,
            prompt=prompt,
            format="json",  # Request JSON output
            options={
                "temperature": 0.1,  # Low temperature for consistent results
                "num_predict": 256,  # Limit output length
            }
        )
        
        return response.get('response', '{}')


class LMStudioBiasAnalyzer(BiasAnalyzerBase):
    """
    Bias analyzer using LM Studio for inference.
    
    LM Studio provides an OpenAI-compatible API at http://localhost:1234/v1
    """
    
    def __init__(self, model: str = "local-model", host: str = "http://localhost:1234/v1"):
        """
        Initialize the LM Studio bias analyzer.
        
        Args:
            model: Model name (LM Studio uses 'local-model' by default)
            host: LM Studio server URL (default: http://localhost:1234/v1)
        """
        self._model = model
        self.host = host
        self._client = None
        
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def backend_name(self) -> str:
        return "lm-studio"
    
    def _get_client(self):
        """Get or create the OpenAI client for LM Studio."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.host,
                api_key="lm-studio"  # LM Studio doesn't require a real key
            )
        return self._client
    
    def check_connection(self) -> bool:
        """Check if LM Studio is running."""
        try:
            client = self._get_client()
            # Try to list models
            models = client.models.list()
            return True
        except Exception as e:
            print(f"  Error connecting to LM Studio: {e}")
            print("  Make sure LM Studio is running with a model loaded")
            print("  and the local server is started (default port 1234)")
            return False
    
    def _generate(self, prompt: str) -> str:
        """Generate response from LM Studio."""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"}  # Request JSON output
        )
        
        return response.choices[0].message.content


def create_analyzer(backend: str = "ollama", **kwargs) -> BiasAnalyzerBase:
    """
    Factory function to create a bias analyzer.
    
    Args:
        backend: "ollama" or "lm-studio"
        **kwargs: Additional arguments passed to the analyzer
        
    Returns:
        BiasAnalyzerBase instance
    """
    if backend == "ollama":
        return OllamaBiasAnalyzer(**kwargs)
    elif backend == "lm-studio":
        return LMStudioBiasAnalyzer(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'lm-studio'")


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
    
    # Install openai package (for LM Studio)
    try:
        import openai
        print("  openai package already installed")
    except ImportError:
        print("  Installing openai package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        print("  Installed openai successfully")
    
    print()


def main():
    """Main demo function."""
    print("=" * 70)
    print("Bias Detection with Local LLMs - Python Demo")
    print("=" * 70)
    print()
    
    # Step 1: Install dependencies
    print("Step 1: Installing dependencies...")
    install_dependencies()
    
    # Step 2: Determine which backend to use
    backend = os.getenv("BIAS_BACKEND", "ollama").lower()
    model = os.getenv("BIAS_MODEL", "llama3.2" if backend == "ollama" else "local-model")
    host = os.getenv("BIAS_HOST", "http://localhost:11434" if backend == "ollama" else "http://localhost:1234/v1")
    
    print(f"Step 2: Initializing {backend} bias analyzer...")
    print(f"  Backend: {backend}")
    print(f"  Model: {model}")
    print(f"  Host: {host}")
    print()
    
    # Create the analyzer
    try:
        analyzer = create_analyzer(backend, model=model, host=host)
    except ValueError as e:
        print(f"  Error: {e}")
        return
    
    # Step 3: Check connection
    print(f"Step 3: Checking {backend} connection...")
    if not analyzer.check_connection():
        print()
        if backend == "ollama":
            print("  Please ensure Ollama is running and the model is available:")
            print("    1. Start Ollama: ollama serve")
            print(f"    2. Pull the model: ollama pull {model}")
        else:
            print("  Please ensure LM Studio is running:")
            print("    1. Open LM Studio")
            print("    2. Load a model")
            print("    3. Start the local server (port 1234)")
        print()
        print("  Trying alternative backend...")
        
        # Try the other backend
        alt_backend = "lm-studio" if backend == "ollama" else "ollama"
        print(f"  Switching to {alt_backend}...")
        try:
            analyzer = create_analyzer(alt_backend)
            if not analyzer.check_connection():
                print("  No backend available. Exiting.")
                return
        except Exception as e:
            print(f"  Error: {e}")
            return
    else:
        print(f"  {backend} is running and model is available!")
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
            print(f"  Backend: {result.backend} ({result.model})")
            
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
            print(f"    -> {dir_label}-leaning, {deg_label} bias ({result.backend})")
            print()
    except Exception as e:
        print(f"  Error in batch analysis: {e}")
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("To use with a different backend:")
    print("  # For Ollama:")
    print("  export BIAS_BACKEND=ollama")
    print("  export BIAS_MODEL=llama3.2")
    print()
    print("  # For LM Studio:")
    print("  export BIAS_BACKEND=lm-studio")
    print("  export BIAS_MODEL=local-model")
    print("  export BIAS_HOST=http://localhost:1234/v1")
    print()
    print("Then run: python demo_local_llm_bias.py")


if __name__ == "__main__":
    main()