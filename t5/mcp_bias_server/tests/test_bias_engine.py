#!/usr/bin/env python3
"""
Tests for the BiasEngine class.

These tests verify the bias detection engine functionality including:
- JSON repair and normalization
- Result validation
- Model loading and inference (requires model files)
"""

import os
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from mcp_bias_server.bias_engine import BiasEngine, BiasResult


class TestBiasResult:
    """Tests for the BiasResult dataclass."""
    
    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        result = BiasResult(
            direction={"L": 0.2, "C": 0.5, "R": 0.3},
            degree={"L": 0.3, "M": 0.5, "H": 0.2},
            reason="Test reason",
            device="cpu"
        )
        
        d = result.to_dict()
        assert d["dir"] == {"L": 0.2, "C": 0.5, "R": 0.3}
        assert d["deg"] == {"L": 0.3, "M": 0.5, "H": 0.2}
        assert d["reason"] == "Test reason"
        assert d["device"] == "cpu"
        assert "raw_output" not in d
    
    def test_to_dict_with_raw_output(self):
        """Test dictionary conversion with raw output."""
        result = BiasResult(
            direction={"L": 0.0, "C": 0.0, "R": 0.0},
            degree={"L": 0.0, "M": 0.0, "H": 0.0},
            reason="Failed to parse",
            raw_output="invalid json",
            device="cpu"
        )
        
        d = result.to_dict()
        assert d["raw_output"] == "invalid json"


class TestBiasEngineJSONRepair:
    """Tests for JSON repair functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a BiasEngine instance for testing."""
        return BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            lazy_load=True
        )
    
    def test_repair_missing_braces(self, engine):
        """Test repair of missing outer braces."""
        result = engine._repair_json_string('"dir":{"L":0.2,"C":0.5,"R":0.3},"deg":{"L":0.3,"M":0.5,"H":0.2},"reason":"test"')
        assert result is not None
        assert "dir" in result
        assert "deg" in result
        assert "reason" in result
    
    def test_repair_missing_quotes_on_keys(self, engine):
        """Test repair of missing quotes on known keys."""
        result = engine._repair_json_string('dir":{"L":0.2},"deg":{"M":0.5},"reason":"test"')
        assert result is not None
        assert "dir" in result
        assert "deg" in result
    
    def test_repair_nested_structure(self, engine):
        """Test repair of nested structure issues."""
        result = engine._repair_json_string('"dir":"L":0.5,"deg":"M":0.5,"reason":"test"')
        assert result is not None
    
    def test_repair_empty_string(self, engine):
        """Test handling of empty string."""
        result = engine._repair_json_string("")
        assert result is None
    
    def test_repair_valid_json(self, engine):
        """Test that valid JSON passes through."""
        valid = '{"dir":{"L":0.2,"C":0.5,"R":0.3},"deg":{"L":0.3,"M":0.5,"H":0.2},"reason":"test"}'
        result = engine._repair_json_string(valid)
        assert result is not None
        assert result["dir"]["L"] == 0.2


class TestBiasEngineNormalization:
    """Tests for result normalization."""
    
    @pytest.fixture
    def engine(self):
        """Create a BiasEngine instance for testing."""
        return BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            lazy_load=True
        )
    
    def test_normalize_direction_keys(self, engine):
        """Test normalization of direction key variations."""
        result = {
            "dir": {"left": 0.2, "center": 0.5, "right": 0.3},
            "deg": {"L": 0.3, "M": 0.5, "H": 0.2},
            "reason": "test"
        }
        
        normalized = engine._normalize_result(result)
        assert normalized is not None
        assert "L" in normalized["dir"]
        assert "C" in normalized["dir"]
        assert "R" in normalized["dir"]
    
    def test_normalize_degree_keys(self, engine):
        """Test normalization of degree key variations."""
        result = {
            "dir": {"L": 0.2, "C": 0.5, "R": 0.3},
            "deg": {"low": 0.3, "medium": 0.5, "high": 0.2},
            "reason": "test"
        }
        
        normalized = engine._normalize_result(result)
        assert normalized is not None
        assert "L" in normalized["deg"]
        assert "M" in normalized["deg"]
        assert "H" in normalized["deg"]
    
    def test_normalize_alternative_field_names(self, engine):
        """Test normalization of alternative field names."""
        result = {
            "direction": {"L": 0.2, "C": 0.5, "R": 0.3},
            "intensity": {"L": 0.3, "M": 0.5, "H": 0.2},
            "explanation": "test"
        }
        
        normalized = engine._normalize_result(result)
        assert normalized is not None
        assert "dir" in normalized
        assert "deg" in normalized
        assert "reason" in normalized
    
    def test_normalize_empty_result(self, engine):
        """Test handling of empty result."""
        result = engine._normalize_result(None)
        assert result is None


class TestBiasEngineValidation:
    """Tests for result validation."""
    
    @pytest.fixture
    def engine(self):
        """Create a BiasEngine instance for testing."""
        return BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            lazy_load=True
        )
    
    def test_validate_valid_result(self, engine):
        """Test validation of a valid result."""
        result = {
            "dir": {"L": 0.2, "C": 0.5, "R": 0.3},
            "deg": {"L": 0.3, "M": 0.5, "H": 0.2},
            "reason": "This is a test reason"
        }
        
        validated = engine._validate_result(result)
        assert validated is not None
        assert validated["dir"]["L"] == 0.2
        assert validated["deg"]["M"] == 0.5
    
    def test_validate_missing_field(self, engine):
        """Test validation fails with missing required field."""
        result = {
            "dir": {"L": 0.2, "C": 0.5, "R": 0.3},
            "deg": {"L": 0.3, "M": 0.5, "H": 0.2}
            # Missing "reason"
        }
        
        validated = engine._validate_result(result)
        assert validated is None
    
    def test_validate_missing_dir_keys(self, engine):
        """Test validation fails with missing direction keys."""
        result = {
            "dir": {"L": 0.2, "C": 0.5},  # Missing "R"
            "deg": {"L": 0.3, "M": 0.5, "H": 0.2},
            "reason": "test"
        }
        
        validated = engine._validate_result(result)
        assert validated is None
    
    def test_validate_missing_deg_keys(self, engine):
        """Test validation fails with missing degree keys."""
        result = {
            "dir": {"L": 0.2, "C": 0.5, "R": 0.3},
            "deg": {"L": 0.3, "H": 0.2},  # Missing "M"
            "reason": "test"
        }
        
        validated = engine._validate_result(result)
        assert validated is None
    
    def test_validate_non_dict_fields(self, engine):
        """Test validation fails when fields are not dicts."""
        result = {
            "dir": "left",
            "deg": {"L": 0.3, "M": 0.5, "H": 0.2},
            "reason": "test"
        }
        
        validated = engine._validate_result(result)
        assert validated is None


class TestBiasEngineDeviceSelection:
    """Tests for device selection logic."""
    
    def test_device_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        engine = BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            device="cpu",
            lazy_load=True
        )
        
        assert engine._get_device() == "cpu"
    
    def test_forced_device(self):
        """Test forced device selection."""
        engine = BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            device="cuda",
            lazy_load=True
        )
        
        assert engine._get_device() == "cuda"


class TestBiasEngineSingleton:
    """Tests for singleton pattern."""
    
    def test_get_instance_creates_singleton(self):
        """Test that get_instance creates a singleton."""
        # Clear any existing instance
        BiasEngine._instance = None
        
        engine1 = BiasEngine.get_instance(model_path="/fake/path1")
        engine2 = BiasEngine.get_instance()
        
        assert engine1 is engine2
        
        # Cleanup
        BiasEngine._instance = None


class TestBiasEngineModelInfo:
    """Tests for model info functionality."""
    
    @patch('mcp_bias_server.bias_engine.BiasEngine._ensure_loaded')
    def test_get_model_info(self, mock_ensure_loaded):
        """Test getting model information."""
        engine = BiasEngine(
            model_path="/fake/path",
            base_model_name="t5-large",
            device="cpu",
            lazy_load=True
        )
        
        # Mock the model data
        BiasEngine._model_data = {
            "model": Mock(),
            "tokenizer": Mock(),
            "device": "cpu"
        }
        engine._loaded = True
        
        info = engine.get_model_info()
        
        assert info["base_model"] == "t5-large"
        assert info["adapter_path"] == "/fake/path"
        assert info["device"] == "cpu"
        assert info["model_type"] == "T5 with LoRA"
        assert "dir" in info["output_format"]
        assert "deg" in info["output_format"]
        
        # Cleanup
        BiasEngine._model_data = None


# Integration tests (require actual model files)
@pytest.mark.integration
class TestBiasEngineIntegration:
    """Integration tests requiring model files."""
    
    @pytest.fixture
    def real_engine(self):
        """Create a real BiasEngine if model files exist."""
        model_path = os.environ.get("BIAS_MODEL_PATH", "./bias-detector-output")
        if not os.path.exists(model_path):
            pytest.skip(f"Model path not found: {model_path}")
        
        return BiasEngine(model_path=model_path, lazy_load=True)
    
    def test_analyze_real_text(self, real_engine):
        """Test analyzing real text with the model."""
        text = "The president announced new economic policies today."
        
        result = real_engine.analyze(text)
        
        assert isinstance(result, BiasResult)
        assert result.device in ["cpu", "mps", "cuda"]
        assert len(result.direction) == 3
        assert len(result.degree) == 3
        assert isinstance(result.reason, str)
    
    def test_analyze_batch_real_texts(self, real_engine):
        """Test batch analysis with real texts."""
        texts = [
            "The president announced new economic policies today.",
            "Critics argue the new legislation goes too far."
        ]
        
        results = real_engine.analyze_batch(texts)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, BiasResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])