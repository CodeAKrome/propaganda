"""
Bias Detector - Political bias detection for news articles.

A tool for processing news articles from MongoDB and detecting political bias
using a T5-based machine learning model.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from bias_detector.processor import BiasProcessor
from bias_detector.validator import BiasValidator
from bias_detector.cli import detect_bias

__all__ = ["BiasProcessor", "BiasValidator", "detect_bias", "__version__"]
