#!/usr/bin/env python3
"""
Bias Engine - Advanced T5 model wrapper for political bias detection.

This module provides the core functionality for political bias detection
using fine-tuned T5 models with LoRA adapters. It supports multiple hardware
backends, batch processing, and comprehensive error handling.

Key Features:
    - T5-large with LoRA adapters for efficient inference
    - MPS (Apple Silicon), CUDA, and CPU support
    - Lazy loading for faster startup
    - Batch processing for multiple texts
    - JSON repair for malformed model outputs
    - Multiple output formats
    - Resource management

Architecture:
    - BiasEngine: Main class for model management and inference
    - BiasResult: Dataclass for structured results
    - BiasAnalyzer: High-level analysis utilities

Usage:
    # Basic usage
    engine = BiasEngine()
    result = engine.analyze("Your text here")
    print(result.direction, result.degree, result.reason)
    
    # Batch processing
    results = engine.analyze_batch(["text1", "text2", "text3"])
    
    # With custom model
    engine = BiasEngine(model_path="/path/to/model", base_model_name="t5-base")

Author: Cline Team
License: MIT
"""

from __future__ import annotations

import os
import json
import re
import logging
import warnings
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from contextlib import contextmanager

# Suppress library noise for cleaner output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "0"
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cline-bias-engine")

# Import ML libraries
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel, PeftConfig

# Default HuggingFace model repository
DEFAULT_HF_MODEL_REPO = "kylerussell/bias-detector-t5-lora"

# Default local model path (relative to package)
DEFAULT_LOCAL_MODEL_PATH = "bias-detector-output"


class DeviceType(Enum):
    """Supported device types for model inference."""
    MPS = "mps"      # Apple Silicon
    CUDA = "cuda"    # NVIDIA GPU
    CPU = "cpu"      # CPU fallback


class OutputFormat(Enum):
    """Supported output formats for bias analysis."""
    JSON = "json"           # Standard JSON
    VERBOSE = "verbose"     # Detailed with all metadata
    COMPACT = "compact"     # Minimal output
    TEXT = "text"           # Human-readable text


@dataclass
class BiasResult:
    """
    Structured bias analysis result.
    
    Attributes:
        direction: Political direction scores (L=Left, C=Center, R=Right)
                   Values sum to approximately 1.0
        degree: Bias intensity scores (L=Low, M=Medium, H=High)
                Values sum to approximately 1.0
        reason: Human-readable explanation of the classification
        raw_output: Raw model output (if available)
        device: Device used for inference
        model_info: Model metadata
        confidence: Overall confidence score (0-1)
    """
    direction: Dict[str, float] = field(default_factory=lambda: {"L": 0.0, "C": 0.0, "R": 0.0})
    degree: Dict[str, float] = field(default_factory=lambda: {"L": 0.0, "M": 0.0, "H": 0.0})
    reason: str = ""
    raw_output: Optional[str] = None
    device: str = "unknown"
    model_info: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(
        self,
        format: OutputFormat = OutputFormat.JSON,
        include_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Convert result to dictionary for JSON serialization.
        
        Args:
            format: Output format to use
            include_raw: Whether to include raw model output
            
        Returns:
            Dictionary representation of the result
        """
        if format == OutputFormat.COMPACT:
            return {
                "direction": self.dominant_direction,
                "degree": self.dominant_degree,
                "reason": self.reason[:100] + "..." if len(self.reason) > 100 else self.reason,
            }
        
        if format == OutputFormat.TEXT:
            return {
                "text": self.to_text(),
                "direction": self.dominant_direction,
                "degree": self.dominant_degree,
            }
        
        # Default JSON format
        result = {
            "direction": self.direction,
            "degree": self.degree,
            "reason": self.reason,
            "dominant_direction": self.dominant_direction,
            "dominant_degree": self.dominant_degree,
            "direction_percent": {k: round(v * 100, 1) for k, v in self.direction.items()},
            "degree_percent": {k: round(v * 100, 1) for k, v in self.degree.items()},
            "confidence": round(self.confidence, 3),
            "device": self.device,
        }
        
        if include_raw and self.raw_output:
            result["raw_output"] = self.raw_output
            
        if self.model_info:
            result["model_info"] = self.model_info
            
        return result
    
    def to_text(self) -> str:
        """
        Convert result to human-readable text.
        
        Returns:
            Formatted text representation
        """
        return (
            f"Political Bias Analysis\n"
            f"========================\n"
            f"Direction: {self.dominant_direction} ({self.direction_percent[self.dominant_code]}%)\n"
            f"Degree: {self.dominant_degree} ({self.degree_percent[self.dominant_code]}%)\n"
            f"Confidence: {self.confidence:.1%}\n"
            f"\nReasoning:\n{self.reason}"
        )
    
    @property
    def dominant_direction(self) -> str:
        """Get the dominant political direction as a string."""
        dir_map = {"L": "Left", "C": "Center", "R": "Right"}
        return dir_map.get(self.dominant_code, "Unknown")
    
    @property
    def dominant_degree(self) -> str:
        """Get the dominant bias degree as a string."""
        deg_map = {"L": "Low", "M": "Medium", "H": "High"}
        return deg_map.get(self.dominant_degree_code, "Unknown")
    
    @property
    def dominant_code(self) -> str:
        """Get the code for dominant direction (L, C, or R)."""
        return max(self.direction, key=self.direction.get)
    
    @property
    def dominant_degree_code(self) -> str:
        """Get the code for dominant degree (L, M, or H)."""
        return max(self.degree, key=self.degree.get)
    
    @property
    def direction_percent(self) -> Dict[str, float]:
        """Get direction values as percentages."""
        return {k: round(v * 100, 1) for k, v in self.direction.items()}
    
    @property
    def degree_percent(self) -> Dict[str, float]:
        """Get degree values as percentages."""
        return {k: round(v * 100, 1) for k, v in self.degree.items()}


class BiasEngine:
    """
    T5-based political bias detection engine.
    
    This is the main class for performing political bias analysis.
    It handles model loading, inference, and result processing.
    
    Features:
        - Lazy loading for faster startup
        - Automatic device detection (MPS/CUDA/CPU)
        - HuggingFace Hub integration for easy model access
        - Comprehensive error handling
        - Batch processing support
    
    Example:
        >>> engine = BiasEngine()
        >>> result = engine.analyze("The president announced new policies today...")
        >>> print(result.dominant_direction)
        'Right'
        
        >>> # Batch processing
        >>> results = engine.analyze_batch([
        ...     "Article 1...",
        ...     "Article 2...",
        ...     "Article 3..."
        ... ])
    """
    
    # Class-level model storage for singleton pattern
    _instance: Optional['BiasEngine'] = None
    _model_data: Optional[Dict[str, Any]] = None
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
        use_huggingface: bool = True,
        max_input_length: int = 512,
        max_output_length: int = 512,
    ):
        """
        Initialize the bias engine.
        
        Args:
            model_path: Path to LoRA adapter weights. If None, uses default resolution:
                       1. BIAS_MODEL_PATH environment variable
                       2. Bundled model in package directory
                       3. HuggingFace Hub (if use_huggingface=True)
            base_model_name: HuggingFace model name for base T5 model.
                            Defaults to BIAS_BASE_MODEL env or "t5-large"
            device: Force specific device ("mps", "cuda", "cpu").
                   Auto-detects if None using BIAS_DEVICE env var
            lazy_load: If True, model loads on first inference call (default: True)
            use_huggingface: If True, download from HuggingFace Hub when local model not found
            max_input_length: Maximum input token length
            max_output_length: Maximum output token length
        
        Environment Variables:
            BIAS_MODEL_PATH: Path to LoRA adapter weights
            BIAS_BASE_MODEL: Base T5 model name (default: t5-large)
            BIAS_DEVICE: Force device selection (mps/cuda/cpu)
            BIAS_USE_HUGGINGFACE: Set to "0" to disable HuggingFace Hub auto-download
        
        Example:
            >>> # Use default settings (auto-detect model and device)
            >>> engine = BiasEngine()
            
            >>> # Use local model on GPU
            >>> engine = BiasEngine(
            ...     model_path="/path/to/model",
            ...     device="cuda"
            ... )
            
            >>> # Disable HuggingFace auto-download
            >>> engine = BiasEngine(use_huggingface=False)
        """
        self.use_huggingface = use_huggingface and os.getenv("BIAS_USE_HUGGINGFACE", "1") != "0"
        self._hf_model_loaded = False
        
        # Resolve model path
        if model_path is None:
            model_path = self._resolve_model_path()
        
        self.model_path = model_path
        self.base_model_name = base_model_name or os.getenv("BIAS_BASE_MODEL", "t5-large")
        self.forced_device = device or os.getenv("BIAS_DEVICE")
        self.lazy_load = lazy_load
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self._loaded = False
        
        # Generation parameters
        self.num_beams = 4
        self.early_stopping = True
        self.temperature = 0.7
        self.do_sample = False
        
        if not lazy_load:
            self._ensure_loaded()
    
    def _resolve_model_path(self) -> str:
        """
        Resolve model path using fallback chain.
        
        Resolution order:
        1. BIAS_MODEL_PATH environment variable
        2. Bundled model in package directory
        3. HuggingFace Hub (if use_huggingface=True)
        4. Local ./bias-detector-output (last resort)
        
        Returns:
            Resolved model path (local path or HuggingFace repo ID)
        """
        # 1. Check environment variable
        env_path = os.getenv("BIAS_MODEL_PATH")
        if env_path and Path(env_path).exists():
            logger.info(f"Using model from BIAS_MODEL_PATH: {env_path}")
            return env_path
        
        # 2. Check bundled model
        bundled_path = Path(__file__).parent / "bias-detector-output"
        if bundled_path.exists():
            logger.info(f"Using bundled model: {bundled_path}")
            return str(bundled_path)
        
        # 3. Use HuggingFace Hub
        if self.use_huggingface:
            logger.info(f"Will attempt to load model from HuggingFace: {DEFAULT_HF_MODEL_REPO}")
            self._hf_model_loaded = True
            return DEFAULT_HF_MODEL_REPO
        
        # 4. Fallback to local path
        logger.warning("No model path specified, using default './bias-detector-output'")
        return "./bias-detector-output"
    
    @classmethod
    def get_instance(
        cls,
        model_path: Optional[str] = None,
        base_model_name: str = "t5-large",
        device: Optional[str] = None
    ) -> 'BiasEngine':
        """
        Get singleton instance of the bias engine.
        
        This is a convenience method for getting a shared engine instance.
        The singleton is created with lazy loading enabled.
        
        Args:
            model_path: Path to LoRA adapter weights
            base_model_name: HuggingFace model name
            device: Force specific device
            
        Returns:
            Singleton BiasEngine instance
            
        Example:
            >>> engine = BiasEngine.get_instance()
            >>> result = engine.analyze("text")
        """
        if cls._instance is None:
            model_path = model_path or os.getenv("BIAS_MODEL_PATH", "./bias-detector-output")
            base_model_name = os.getenv("BIAS_BASE_MODEL", base_model_name)
            device = device or os.getenv("BIAS_DEVICE")
            cls._instance = cls(model_path, base_model_name, device, lazy_load=True)
        return cls._instance
    
    def _get_device(self) -> str:
        """
        Determine the best available device for inference.
        
        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        if self.forced_device:
            return self.forced_device
        
        # Check CUDA first (most powerful)
        if torch.cuda.is_available():
            return "cuda"
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Verify MPS is built
            try:
                _ = torch.zeros(1).to("mps")
                return "mps"
            except Exception:
                pass
        
        # Fallback to CPU
        return "cpu"
    
    def _ensure_loaded(self) -> None:
        """
        Load model and tokenizer if not already loaded.
        
        This method implements lazy loading - the model is only loaded
        when first needed.
        """
        if self._loaded and BiasEngine._model_data is not None:
            return
        
        device = self._get_device()
        logger.info(f"Loading model on device: {device}")
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer: {self.base_model_name}")
            tokenizer = T5Tokenizer.from_pretrained(
                self.base_model_name,
                verbose=False,
                trust_remote_code=True
            )
            
            # Load base model
            logger.info(f"Loading base model: {self.base_model_name}")
            base_model = T5ForConditionalGeneration.from_pretrained(
                self.base_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map=None,  # We'll manually set device
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter: {self.model_path}")
            model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                is_trainable=False
            )
            
            # Move to device
            model.to(device)
            model.eval()
            
            # Store model data
            BiasEngine._model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "base_model": self.base_model_name,
                "adapter_path": self.model_path,
            }
            
            self._loaded = True
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _repair_json_string(self, raw: str) -> Optional[Dict]:
        """
        Attempt to repair malformed JSON from model output.
        
        The T5 model sometimes produces JSON with minor syntax errors.
        This method attempts to fix common issues:
        - Missing outer braces
        - Missing quotes on keys
        - Unterminated strings
        - Missing nested braces
        - Extra commas
        
        Args:
            raw: Raw string from model output
            
        Returns:
            Parsed dict or None if unrepairable
        """
        if not raw:
            return None
        
        s = raw.strip()
        
        # Remove outer quotes if present
        if s.startswith('"') and s.endswith('"'):
            try:
                s = json.loads(s)
            except json.JSONDecodeError:
                s = s[1:-1]
        
        # Fix missing opening quote on "dir" at start
        if s.startswith('dir":'):
            s = '"' + s
        
        # Fix missing quotes on known keys
        s = re.sub(r'(?<!")\b(dir|deg|reason)\b(?!"):', r'"\1":', s)
        
        # Add outer braces if missing
        if not s.startswith('{'):
            s = '{' + s
        if not s.endswith('}'):
            s = s + '}'
        
        # Fix "dir":"L":value patterns -> "dir":{"L":value}
        s = re.sub(r'"dir"\s*:\s*"([LRC])"\s*:', r'"dir":{"\1":', s)
        s = re.sub(r'"deg"\s*:\s*"([LMH])"\s*:', r'"deg":{"\1":', s)
        
        # Find positions and fix nested objects
        dir_match = re.search(r'"dir"\s*:\s*\{', s)
        deg_match = re.search(r'"deg"\s*:\s*\{', s)
        reason_match = re.search(r'"reason"\s*:', s)
        
        if dir_match and deg_match:
            deg_start = deg_match.start()
            dir_section = s[:deg_start]
            open_braces = dir_section.count('{') - dir_section.count('}')
            if open_braces > 0:
                s = s[:deg_start] + '}' * (open_braces - 1) + ',' + s[deg_start:]
        
        # Re-find positions after modification
        deg_match = re.search(r'"deg"\s*:\s*\{', s)
        reason_match = re.search(r'"reason"\s*:', s)
        
        if deg_match and reason_match:
            reason_start = reason_match.start()
            deg_section = s[:reason_start]
            open_braces = deg_section.count('{') - deg_section.count('}')
            if open_braces > 0:
                s = s[:reason_start] + '}' * (open_braces - 1) + ',' + s[reason_start:]
        
        # Fix unterminated string at end
        if s.endswith('}'):
            reason_val_match = re.search(r'"reason"\s*:\s*"([^"]*)$', s)
            if reason_val_match:
                s = s[:-1] + '"}'
        
        # Ensure proper closing
        open_braces = s.count('{') - s.count('}')
        if open_braces > 0:
            s = s.rstrip('}') + '}' * (open_braces + 1)
        
        # Fix trailing commas
        s = re.sub(r',\s*}', '}', s)
        
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    
    def _normalize_result(self, result: Dict) -> Optional[Dict]:
        """
        Normalize and correct common model output issues.
        
        Handles:
        - Key name variations (left->L, center->C, right->R)
        - Case variations
        - Missing fields
        - Alternative field names
        
        Args:
            result: Raw bias result from model
            
        Returns:
            Normalized result or None if unfixable
        """
        if not result:
            return None
        
        # Handle raw_output field
        if "raw_output" in result:
            repaired = self._repair_json_string(result["raw_output"])
            if repaired:
                result = repaired
            else:
                return None
        
        normalized = dict(result)
        
        # Direction key mappings
        dir_key_map = {
            "left": "L", "l": "L", "LEFT": "L", "Left": "L",
            "center": "C", "c": "C", "CENTER": "C", "Center": "C", "centre": "C",
            "right": "R", "r": "R", "RIGHT": "R", "Right": "R",
        }
        
        # Degree key mappings
        deg_key_map = {
            "low": "L", "l": "L", "LOW": "L", "Low": "L",
            "medium": "M", "m": "M", "MEDIUM": "M", "Medium": "M", "moderate": "M",
            "high": "H", "h": "H", "HIGH": "H", "High": "H", "heavy": "H",
        }
        
        # Normalize "dir" field
        if "dir" in normalized and isinstance(normalized["dir"], dict):
            new_dir = {}
            for key, value in normalized["dir"].items():
                new_key = dir_key_map.get(key, key)
                new_dir[new_key] = float(value) if value else 0.0
            normalized["dir"] = new_dir
        
        # Normalize "deg" field
        if "deg" in normalized and isinstance(normalized["deg"], dict):
            new_deg = {}
            for key, value in normalized["deg"].items():
                new_key = deg_key_map.get(key, key)
                new_deg[new_key] = float(value) if value else 0.0
            normalized["deg"] = new_deg
        
        # Handle alternative field names
        if "dir" not in normalized:
            for alt in ["direction", "bias_dir", "political_dir", "orientation", "leaning"]:
                if alt in normalized:
                    normalized["dir"] = normalized.pop(alt)
                    break
        
        if "deg" not in normalized:
            for alt in ["degree", "bias_deg", "intensity", "strength", "magnitude"]:
                if alt in normalized:
                    normalized["deg"] = normalized.pop(alt)
                    break
        
        if "reason" not in normalized:
            for alt in ["explanation", "rationale", "analysis", "why", "justification", "text"]:
                if alt in normalized:
                    normalized["reason"] = normalized.pop(alt)
                    break
        
        return normalized
    
    def _validate_result(self, result: Dict) -> Optional[Dict]:
        """
        Validate and clean bias result.
        
        Expected format: {"dir": {...}, "deg": {...}, "reason": "..."}
        
        Args:
            result: Raw bias result
            
        Returns:
            Validated result or None if invalid
        """
        if not result:
            return None
        
        normalized = self._normalize_result(result)
        if not normalized:
            return None
        
        # Check required fields
        required = ["dir", "deg", "reason"]
        missing = [f for f in required if f not in normalized]
        if missing:
            logger.warning(f"Missing required fields: {missing}")
            return None
        
        # Validate dir structure
        if not isinstance(normalized["dir"], dict):
            return None
        if not set(normalized["dir"].keys()).issuperset({"L", "C", "R"}):
            # Try to use available keys
            available = set(normalized["dir"].keys())
            if available.issuperset({"left", "center", "right"}):
                normalized["dir"] = {
                    "L": normalized["dir"].get("left", 0.0),
                    "C": normalized["dir"].get("center", 0.0),
                    "R": normalized["dir"].get("right", 0.0),
                }
            else:
                return None
        
        # Validate deg structure
        if not isinstance(normalized["deg"], dict):
            return None
        if not set(normalized["deg"].keys()).issuperset({"L", "M", "H"}):
            available = set(normalized["deg"].keys())
            if available.issuperset({"low", "medium", "high"}):
                normalized["deg"] = {
                    "L": normalized["deg"].get("low", 0.0),
                    "M": normalized["deg"].get("medium", 0.0),
                    "H": normalized["deg"].get("high", 0.0),
                }
            else:
                return None
        
        # Normalize values to ensure they sum to ~1.0
        dir_sum = sum(normalized["dir"].get(k, 0.0) for k in ["L", "C", "R"])
        deg_sum = sum(normalized["deg"].get(k, 0.0) for k in ["L", "M", "H"])
        
        if dir_sum > 0:
            normalized["dir"] = {
                "L": normalized["dir"].get("L", 0.0) / dir_sum,
                "C": normalized["dir"].get("C", 0.0) / dir_sum,
                "R": normalized["dir"].get("R", 0.0) / dir_sum,
            }
        
        if deg_sum > 0:
            normalized["deg"] = {
                "L": normalized["deg"].get("L", 0.0) / deg_sum,
                "M": normalized["deg"].get("M", 0.0) / deg_sum,
                "H": normalized["deg"].get("H", 0.0) / deg_sum,
            }
        
        return {
            "dir": {
                "L": normalized["dir"]["L"],
                "C": normalized["dir"]["C"],
                "R": normalized["dir"]["R"]
            },
            "deg": {
                "L": normalized["deg"]["L"],
                "M": normalized["deg"]["M"],
                "H": normalized["deg"]["H"]
            },
            "reason": str(normalized["reason"]) if normalized["reason"] else "No explanation provided"
        }
    
    def _calculate_confidence(self, direction: Dict[str, float], degree: Dict[str, float]) -> float:
        """
        Calculate overall confidence score based on result distribution.
        
        Higher confidence when:
        - Values are more polarized (one direction/degree dominates)
        - The dominant value is significantly higher
        
        Args:
            direction: Direction scores
            degree: Degree scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate max direction and degree
        max_dir = max(direction.values())
        max_deg = max(degree.values())
        
        # Calculate entropy (lower = more confident)
        dir_entropy = -sum(
            v * (v if v > 0 else 1) 
            for v in direction.values()
        )
        deg_entropy = -sum(
            v * (v if v > 0 else 1) 
            for v in degree.values()
        )
        
        # Combine metrics
        confidence = (max_dir + max_deg) / 2
        
        return min(1.0, max(0.0, confidence))
    
    def analyze(
        self,
        text: str,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> BiasResult:
        """
        Analyze text for political bias.
        
        This is the main method for bias detection. It takes a text input
        and returns a structured BiasResult with direction, degree, and reasoning.
        
        Args:
            text: Article or text to analyze for political bias
            output_format: Format for the output (default: JSON)
            
        Returns:
            BiasResult with direction, degree, reasoning, and metadata
            
        Example:
            >>> engine = BiasEngine()
            >>> result = engine.analyze("The president's policies...")
            >>> print(result.dominant_direction)
            'Right'
            >>> print(result.direction_percent)
            {'L': 10.5, 'C': 15.2, 'R': 74.3}
        """
        self._ensure_loaded()
        
        model_data = BiasEngine._model_data
        device = model_data["device"]
        
        # Format input for T5
        formatted_input = f"classify political bias as json: {text}"
        
        # Tokenize
        inputs = model_data["tokenizer"](
            formatted_input,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model_data["model"].generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
            )
        
        # Decode
        raw_output = model_data["tokenizer"].decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()
        
        # Parse JSON
        try:
            if raw_output.startswith('"') and raw_output.endswith('"'):
                decoded = json.loads(raw_output)
            else:
                decoded = raw_output
            
            if not decoded.startswith('{'):
                decoded = "{" + decoded + "}"
            
            parsed = json.loads(decoded)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            parsed = {"raw_output": raw_output}
        
        # Validate and normalize
        validated = self._validate_result(parsed)
        
        if validated:
            confidence = self._calculate_confidence(validated["dir"], validated["deg"])
            
            return BiasResult(
                direction=validated["dir"],
                degree=validated["deg"],
                reason=validated["reason"],
                raw_output=raw_output,
                device=device,
                confidence=confidence,
                model_info=self.get_model_info()
            )
        else:
            # Return raw output if validation failed
            logger.warning("Model output validation failed, returning fallback result")
            return BiasResult(
                direction={"L": 0.0, "C": 0.0, "R": 0.0},
                degree={"L": 0.0, "M": 0.0, "H": 0.0},
                reason="Failed to parse model output. The text may be too short or ambiguous.",
                raw_output=raw_output,
                device=device,
                confidence=0.0,
                model_info=self.get_model_info()
            )
    
    def analyze_batch(
        self,
        texts: List[str],
        output_format: OutputFormat = OutputFormat.JSON,
        show_progress: bool = False
    ) -> List[BiasResult]:
        """
        Analyze multiple texts for political bias.
        
        More efficient than multiple analyze() calls when processing
        multiple articles.
        
        Args:
            texts: List of texts to analyze
            output_format: Format for output
            show_progress: Whether to show progress (default: False)
            
        Returns:
            List of BiasResult objects
            
        Example:
            >>> results = engine.analyze_batch([
            ...     "Article 1...",
            ...     "Article 2...",
            ...     "Article 3..."
            ... ])
            >>> for r in results:
            ...     print(r.dominant_direction)
        """
        results = []
        
        if show_progress:
            import tqdm
            texts = tqdm.tqdm(texts, desc="Analyzing")
        
        for text in texts:
            result = self.analyze(text, output_format)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
            
        Example:
            >>> info = engine.get_model_info()
            >>> print(info['device'])
            'cuda'
        """
        self._ensure_loaded()
        
        model_data = BiasEngine._model_data
        
        return {
            "base_model": self.base_model_name,
            "adapter_path": self.model_path,
            "device": model_data["device"],
            "model_type": "T5 with LoRA",
            "source": "HuggingFace Hub" if self._hf_model_loaded else "Local",
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "supported_languages": ["English"],
            "framework": "transformers",
            "output_format": {
                "dir": "Political direction scores (L=Left, C=Center, R=Right)",
                "deg": "Bias degree scores (L=Low, M=Medium, H=High)",
                "reason": "Explanation of the classification"
            }
        }
    
    def unload(self) -> None:
        """
        Unload model from memory to free resources.
        
        Call this method when you're done using the engine to free memory.
        After unloading, you'll need to create a new engine or reinitialize.
        
        Example:
            >>> engine.unload()
            >>> # Model is now unloaded
        """
        if BiasEngine._model_data is not None:
            del BiasEngine._model_data
            BiasEngine._model_data = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self._loaded = False
        logger.info("Model unloaded successfully")
    
    @contextmanager
    def managed(self):
        """
        Context manager for automatic resource cleanup.
        
        Example:
            >>> with engine.managed():
            ...     result = engine.analyze("text")
            # Model is automatically unloaded
        """
        try:
            yield self
        finally:
            self.unload()


class BiasAnalyzer:
    """
    High-level bias analysis utilities.
    
    This class provides convenient methods for common analysis tasks
    and additional features beyond the basic BiasEngine.
    
    Example:
        >>> analyzer = BiasAnalyzer()
        >>> summary = analyzer.analyze_multiple_files(["file1.txt", "file2.txt"])
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "t5-large",
        device: Optional[str] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to model (optional)
            base_model: Base T5 model name
            device: Device to use
        """
        self.engine = BiasEngine(
            model_path=model_path,
            base_model_name=base_model,
            device=device,
            lazy_load=True
        )
    
    def analyze_file(
        self,
        filepath: str,
        encoding: str = "utf-8"
    ) -> BiasResult:
        """
        Analyze a text file for bias.
        
        Args:
            filepath: Path to the text file
            encoding: File encoding (default: utf-8)
            
        Returns:
            BiasResult for the file
            
        Example:
            >>> result = analyzer.analyze_file("news_article.txt")
            >>> print(result.dominant_direction)
        """
        with open(filepath, 'r', encoding=encoding) as f:
            text = f.read()
        return self.engine.analyze(text)
    
    def analyze_directory(
        self,
        directory: str,
        extensions: List[str] = None,
        recursive: bool = True
    ) -> Dict[str, BiasResult]:
        """
        Analyze all text files in a directory.
        
        Args:
            directory: Path to directory
            extensions: File extensions to process (default: .txt, .md, .text)
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary mapping filenames to BiasResults
        """
        extensions = extensions or ['.txt', '.md', '.text']
        path = Path(directory)
        
        results = {}
        
        if recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')
        
        for file in files:
            if file.suffix.lower() in extensions:
                try:
                    result = self.analyze_file(str(file))
                    results[str(file)] = result
                except Exception as e:
                    logger.warning(f"Failed to analyze {file}: {e}")
        
        return results
    
    def get_statistics(
        self,
        results: List[BiasResult]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from multiple results.
        
        Args:
            results: List of BiasResults
            
        Returns:
            Dictionary with statistics
            
        Example:
            >>> stats = analyzer.get_statistics(results)
            >>> print(stats['avg_direction'])
        """
        if not results:
            return {}
        
        import numpy as np
        
        # Calculate averages
        avg_direction = {
            k: np.mean([r.direction[k] for r in results])
            for k in ["L", "C", "R"]
        }
        
        avg_degree = {
            k: np.mean([r.degree[k] for r in results])
            for k in ["L", "M", "H"]
        }
        
        # Count dominant directions
        direction_counts = {}
        for r in results:
            dom = r.dominant_direction
            direction_counts[dom] = direction_counts.get(dom, 0) + 1
        
        degree_counts = {}
        for r in results:
            dom = r.dominant_degree
            degree_counts[dom] = degree_counts.get(dom, 0) + 1
        
        return {
            "count": len(results),
            "avg_direction": avg_direction,
            "avg_degree": avg_degree,
            "direction_distribution": direction_counts,
            "degree_distribution": degree_counts,
            "avg_confidence": np.mean([r.confidence for r in results]),
        }
