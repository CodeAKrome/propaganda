#!/usr/bin/env python3
"""
Bias Engine - T5 model wrapper for political bias detection.

Handles model loading, inference, and result normalization.
Supports MPS (Apple Silicon), CUDA, and CPU devices.
"""

import os
import json
import re
import logging
import warnings
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# Suppress library noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# Default HuggingFace model repository for LoRA adapters
DEFAULT_HF_MODEL_REPO = "kylerussell/bias-detector-t5-lora"


@dataclass
class BiasResult:
    """Structured bias analysis result."""
    direction: Dict[str, float]  # {"L": 0.2, "C": 0.5, "R": 0.3}
    degree: Dict[str, float]     # {"L": 0.3, "M": 0.5, "H": 0.2}
    reason: str
    raw_output: Optional[str] = None
    device: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "dir": self.direction,
            "deg": self.degree,
            "reason": self.reason
        }
        if self.raw_output:
            result["raw_output"] = self.raw_output
        result["device"] = self.device
        return result


class BiasEngine:
    """
    T5-based political bias detection engine.
    
    Uses a fine-tuned T5-large model with LoRA adapters for efficient inference.
    Supports multiple hardware backends: MPS (Apple Silicon), CUDA, and CPU.
    
    Example:
        engine = BiasEngine(model_path="./bias-detector-output")
        result = engine.analyze("The president announced new policies today...")
        print(result.to_dict())
    """
    
    _instance: Optional['BiasEngine'] = None
    _model_data: Optional[Dict[str, Any]] = None
    
    def __init__(
        self,
        model_path: str = None,
        base_model_name: str = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
        use_huggingface: bool = True
    ):
        """
        Initialize the bias engine.
        
        Args:
            model_path: Path to LoRA adapter weights (default: auto-resolved via _resolve_model_path)
            base_model_name: HuggingFace model name for base T5 model (default: BIAS_BASE_MODEL env or t5-large)
            device: Force specific device ("mps", "cuda", "cpu"). Auto-detects if None.
            lazy_load: If True, model loads on first inference call
            use_huggingface: If True, download from HuggingFace Hub when local model not found
        """
        self.use_huggingface = use_huggingface
        self._hf_model_loaded = False
        
        # Determine model path with fallback chain
        if model_path is None:
            model_path = self._resolve_model_path()
        
        self.model_path = model_path
        self.base_model_name = base_model_name or os.getenv("BIAS_BASE_MODEL", "t5-large")
        self.forced_device = device or os.getenv("BIAS_DEVICE")
        self.lazy_load = lazy_load
        self._loaded = False
        
        if not lazy_load:
            self._ensure_loaded()
    
    def _resolve_model_path(self) -> str:
        """
        Resolve model path using fallback chain:
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
            return env_path
        
        # 2. Check bundled model
        bundled_path = Path(__file__).parent / "bias-detector-output"
        if bundled_path.exists():
            return str(bundled_path)
        
        # 3. Use HuggingFace Hub
        if self.use_huggingface:
            self._hf_model_loaded = True
            return DEFAULT_HF_MODEL_REPO
        
        # 4. Fallback to local path
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
        
        Uses environment variables for defaults:
        - BIAS_MODEL_PATH: Path to LoRA adapter
        - BIAS_BASE_MODEL: Base T5 model name
        - BIAS_DEVICE: Force device selection
        
        Args:
            model_path: Path to LoRA adapter weights
            base_model_name: HuggingFace model name
            device: Force specific device
            
        Returns:
            Singleton BiasEngine instance
        """
        if cls._instance is None:
            model_path = model_path or os.getenv("BIAS_MODEL_PATH", "./bias-detector-output")
            base_model_name = os.getenv("BIAS_BASE_MODEL", base_model_name)
            device = device or os.getenv("BIAS_DEVICE")
            cls._instance = cls(model_path, base_model_name, device, lazy_load=True)
        return cls._instance
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.forced_device:
            return self.forced_device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    
    def _ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._loaded and BiasEngine._model_data is not None:
            return
        
        device = self._get_device()
        
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(self.base_model_name, verbose=False)
        
        # Load base model
        base_model = T5ForConditionalGeneration.from_pretrained(
            self.base_model_name,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.to(device)
        model.eval()
        
        BiasEngine._model_data = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
        self._loaded = True
    
    def _repair_json_string(self, raw: str) -> Optional[Dict]:
        """
        Attempt to repair malformed JSON from model output.
        
        Handles common issues:
        - Missing outer braces
        - Missing nested braces
        - Missing quotes on keys
        - Unterminated strings
        - Extra quotes around JSON
        
        Args:
            raw: Raw string from model
            
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
        
        # Find positions of dir and deg objects
        dir_match = re.search(r'"dir"\s*:\s*\{', s)
        deg_match = re.search(r'"deg"\s*:\s*\{', s)
        reason_match = re.search(r'"reason"\s*:', s)
        
        if dir_match and deg_match:
            deg_start = deg_match.start()
            dir_section = s[:deg_start]
            open_braces = dir_section.count('{') - dir_section.count('}')
            if open_braces > 0:
                s = s[:deg_start] + '}' * (open_braces - 1) + ',' + s[deg_start:]
        
        # Re-find positions after potential modification
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
        - Raw output that needs JSON repair
        
        Args:
            result: Raw bias result
            
        Returns:
            Normalized result or None if unfixable
        """
        if not result:
            return None
        
        # Handle raw_output
        if "raw_output" in result:
            repaired = self._repair_json_string(result["raw_output"])
            if repaired:
                result = repaired
            else:
                return None
        
        normalized = dict(result)
        
        # Key mappings
        dir_key_map = {
            "left": "L", "l": "L", "LEFT": "L", "Left": "L",
            "center": "C", "c": "C", "CENTER": "C", "Center": "C", "centre": "C",
            "right": "R", "r": "R", "RIGHT": "R", "Right": "R",
        }
        
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
                new_dir[new_key] = value
            normalized["dir"] = new_dir
        
        # Normalize "deg" field
        if "deg" in normalized and isinstance(normalized["deg"], dict):
            new_deg = {}
            for key, value in normalized["deg"].items():
                new_key = deg_key_map.get(key, key)
                new_deg[new_key] = value
            normalized["deg"] = new_deg
        
        # Handle alternative field names
        if "dir" not in normalized:
            for alt in ["direction", "bias_dir", "political_dir", "orientation"]:
                if alt in normalized:
                    normalized["dir"] = normalized.pop(alt)
                    break
        
        if "deg" not in normalized:
            for alt in ["degree", "bias_deg", "intensity", "strength"]:
                if alt in normalized:
                    normalized["deg"] = normalized.pop(alt)
                    break
        
        if "reason" not in normalized:
            for alt in ["explanation", "rationale", "analysis", "why", "justification"]:
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
            return None
        
        # Validate dir structure
        if not isinstance(normalized["dir"], dict):
            return None
        if not set(normalized["dir"].keys()).issuperset({"L", "C", "R"}):
            return None
        
        # Validate deg structure
        if not isinstance(normalized["deg"], dict):
            return None
        if not set(normalized["deg"].keys()).issuperset({"L", "M", "H"}):
            return None
        
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
            "reason": normalized["reason"]
        }
    
    def analyze(self, text: str) -> BiasResult:
        """
        Analyze text for political bias.
        
        Args:
            text: Article or text to analyze
            
        Returns:
            BiasResult with direction, degree, and reasoning
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
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model_data["model"].generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        raw_output = model_data["tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()
        
        # Parse JSON
        try:
            if raw_output.startswith('"') and raw_output.endswith('"'):
                decoded = json.loads(raw_output)
            else:
                decoded = raw_output
            
            if not decoded.startswith('{'):
                decoded = "{" + decoded + "}"
            
            parsed = json.loads(decoded)
        except Exception:
            parsed = {"raw_output": raw_output}
        
        # Validate and normalize
        validated = self._validate_result(parsed)
        
        if validated:
            return BiasResult(
                direction=validated["dir"],
                degree=validated["deg"],
                reason=validated["reason"],
                device=device
            )
        else:
            # Return raw output if validation failed
            return BiasResult(
                direction={"L": 0.0, "C": 0.0, "R": 0.0},
                degree={"L": 0.0, "M": 0.0, "H": 0.0},
                reason="Failed to parse model output",
                raw_output=raw_output,
                device=device
            )
    
    def analyze_batch(self, texts: List[str]) -> List[BiasResult]:
        """
        Analyze multiple texts for political bias.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of BiasResult objects
        """
        return [self.analyze(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._ensure_loaded()
        
        model_data = BiasEngine._model_data
        
        return {
            "base_model": self.base_model_name,
            "adapter_path": self.model_path,
            "device": model_data["device"],
            "model_type": "T5 with LoRA",
            "source": "HuggingFace Hub" if self._hf_model_loaded else "Local",
            "max_input_length": 512,
            "max_output_length": 512,
            "supported_languages": ["English"],
            "output_format": {
                "dir": "Political direction scores (L=Left, C=Center, R=Right)",
                "deg": "Bias degree scores (L=Low, M=Medium, H=High)",
                "reason": "Explanation of the classification"
            }
        }
    
    def unload(self) -> None:
        """Unload model from memory."""
        if BiasEngine._model_data is not None:
            del BiasEngine._model_data
            BiasEngine._model_data = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded = False