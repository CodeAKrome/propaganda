"""Tests for BiasProcessor class."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from bias_detector.processor import BiasProcessor


class TestBiasProcessorJSONRepair:
    """Test JSON repair functionality."""

    @pytest.fixture
    def processor(self):
        """Create a BiasProcessor instance for testing."""
        with patch.dict('os.environ', {'MONGO_USER': 'test', 'MONGO_PASS': 'test'}):
            with patch('bias_detector.processor.MongoClient'):
                return BiasProcessor()

    def test_repair_missing_outer_braces(self, processor):
        """Test adding missing outer braces."""
        raw = '"dir":{"L":0.2,"C":0.6,"R":0.2},"deg":{"L":0.1,"M":0.8,"H":0.1},"reason":"test"'
        result = processor.repair_json_string(raw)
        assert result is not None
        assert "dir" in result
        assert "deg" in result
        assert "reason" in result

    def test_repair_missing_nested_braces(self, processor):
        """Test fixing missing nested braces."""
        raw = '"dir":"L":0.2,"C":0.6,"R":0.1,"deg":"L":0.1,"M":0.2,"H":0.0,"reason":"test"'
        result = processor.repair_json_string(raw)
        assert result is not None
        assert "dir" in result
        assert isinstance(result["dir"], dict)
        assert result["dir"]["L"] == 0.2

    def test_repair_missing_quotes_on_keys(self, processor):
        """Test adding missing quotes on keys."""
        raw = 'dir":{"L":0.2},"deg":{"L":0.1},"reason":"test"'
        result = processor.repair_json_string(raw)
        assert result is not None
        assert "dir" in result

    def test_repair_trailing_commas(self, processor):
        """Test removing trailing commas."""
        raw = '{"dir":{"L":0.2,},"deg":{"L":0.1,},"reason":"test"}'
        result = processor.repair_json_string(raw)
        assert result is not None

    def test_repair_unterminated_string(self, processor):
        """Test fixing unterminated string."""
        raw = '{"dir":{"L":0.2},"deg":{"L":0.1},"reason":"test'
        result = processor.repair_json_string(raw)
        assert result is not None
        assert result["reason"] == "test"

    def test_repair_complete_malformed(self, processor):
        """Test repairing completely malformed JSON."""
        raw = 'dir":"L":0.2,"C":0.7,"R":0.1,"deg":"L":0.1,"M":0.2,"H":0.0,"reason":"The article maintains a neutral tone'
        result = processor.repair_json_string(raw)
        assert result is not None
        assert "dir" in result
        assert "deg" in result
        assert "reason" in result


class TestBiasProcessorNormalization:
    """Test key normalization functionality."""

    @pytest.fixture
    def processor(self):
        """Create a BiasProcessor instance for testing."""
        with patch.dict('os.environ', {'MONGO_USER': 'test', 'MONGO_PASS': 'test'}):
            with patch('bias_detector.processor.MongoClient'):
                return BiasProcessor()

    def test_normalize_dir_keys(self, processor):
        """Test normalizing direction keys."""
        result = {
            "dir": {"left": 0.2, "center": 0.6, "right": 0.2},
            "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
            "reason": "test"
        }
        normalized = processor.normalize_bias_result(result)
        assert normalized is not None
        assert "L" in normalized["dir"]
        assert "C" in normalized["dir"]
        assert "R" in normalized["dir"]

    def test_normalize_deg_keys(self, processor):
        """Test normalizing degree keys."""
        result = {
            "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
            "deg": {"low": 0.1, "medium": 0.8, "high": 0.1},
            "reason": "test"
        }
        normalized = processor.normalize_bias_result(result)
        assert normalized is not None
        assert "L" in normalized["deg"]
        assert "M" in normalized["deg"]
        assert "H" in normalized["deg"]

    def test_normalize_alternative_field_names(self, processor):
        """Test normalizing alternative field names."""
        result = {
            "direction": {"L": 0.2, "C": 0.6, "R": 0.2},
            "degree": {"L": 0.1, "M": 0.8, "H": 0.1},
            "explanation": "test"
        }
        normalized = processor.normalize_bias_result(result)
        assert normalized is not None
        assert "dir" in normalized
        assert "deg" in normalized
        assert "reason" in normalized


class TestBiasProcessorValidation:
    """Test validation functionality."""

    @pytest.fixture
    def processor(self):
        """Create a BiasProcessor instance for testing."""
        with patch.dict('os.environ', {'MONGO_USER': 'test', 'MONGO_PASS': 'test'}):
            with patch('bias_detector.processor.MongoClient'):
                return BiasProcessor()

    def test_validate_valid_result(self, processor):
        """Test validating a valid result."""
        result = {
            "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
            "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
            "reason": "The article is balanced."
        }
        validated = processor.validate_bias_result(result)
        assert validated is not None
        assert validated["dir"]["L"] == 0.2
        assert validated["deg"]["M"] == 0.8

    def test_validate_missing_field(self, processor):
        """Test validation with missing field."""
        result = {
            "dir": {"L": 0.2, "C": 0.6, "R": 0.2},
            "deg": {"L": 0.1, "M": 0.8, "H": 0.1}
        }
        validated = processor.validate_bias_result(result)
        assert validated is None

    def test_validate_missing_keys(self, processor):
        """Test validation with missing keys."""
        result = {
            "dir": {"L": 0.2, "R": 0.8},
            "deg": {"L": 0.1, "M": 0.8, "H": 0.1},
            "reason": "test"
        }
        validated = processor.validate_bias_result(result)
        assert validated is None

    def test_validate_empty_result(self, processor):
        """Test validation with empty result."""
        validated = processor.validate_bias_result({})
        assert validated is None


class TestBiasProcessorOutput:
    """Test output format."""

    @pytest.fixture
    def processor(self):
        """Create a BiasProcessor instance for testing."""
        with patch.dict('os.environ', {'MONGO_USER': 'test', 'MONGO_PASS': 'test'}):
            with patch('bias_detector.processor.MongoClient'):
                return BiasProcessor()

    def test_output_key_order(self, processor):
        """Test that output has correct key order."""
        result = {
            "deg": {"H": 0.1, "L": 0.1, "M": 0.8},
            "dir": {"R": 0.2, "L": 0.2, "C": 0.6},
            "reason": "test"
        }
        validated = processor.validate_bias_result(result)
        assert validated is not None
        
        # Check key order in output
        keys = list(validated.keys())
        assert keys == ["dir", "deg", "reason"]
        
        # Check nested key order
        dir_keys = list(validated["dir"].keys())
        assert dir_keys == ["L", "C", "R"]
        
        deg_keys = list(validated["deg"].keys())
        assert deg_keys == ["L", "M", "H"]
