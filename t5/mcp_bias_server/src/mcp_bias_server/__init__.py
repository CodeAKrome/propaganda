"""
Bias MCP Server - Political bias detection using fine-tuned T5 model.

Out-of-the-box experience:
    pip install bias-mcp-server
    bias "Your text here"
    
Or in Python:
    from mcp_bias_server import quick_analyze
    result = quick_analyze("Your text")
    print(result['direction'])
"""

__version__ = "0.1.2"
__author__ = "Bias Detector Team"
__email__ = "bias-detector@example.com"

from mcp_bias_server.bias_engine import BiasEngine, BiasResult
from mcp_bias_server.cli import main as cli_main
from mcp_bias_server.server import run_server

# Singleton engine for quick_analyze
_engine = None


def quick_analyze(text: str, use_huggingface: bool = True) -> dict:
    """
    One-line bias analysis - the simplest possible API.
    
    Args:
        text: Text to analyze
        use_huggingface: If True, auto-download from HuggingFace if no local model
        
    Returns:
        dict with keys:
            - direction: {"L": float, "C": float, "R": float}
            - degree: {"L": float, "M": float, "H": float}
            - reason: str
            - dominant_direction: "Left" | "Center" | "Right"
            - dominant_degree: "Low" | "Medium" | "High"
    
    Example:
        >>> result = quick_analyze("The president announced new policies.")
        >>> print(result['dominant_direction'])
        'Right'
    """
    global _engine
    
    if _engine is None:
        _engine = BiasEngine(use_huggingface=use_huggingface)
    
    result = _engine.analyze(text)
    
    # Find dominant values
    dom_dir = max(result.direction, key=result.direction.get)
    dom_deg = max(result.degree, key=result.degree.get)
    
    dir_labels = {"L": "Left", "C": "Center", "R": "Right"}
    deg_labels = {"L": "Low", "M": "Medium", "H": "High"}
    
    return {
        "direction": result.direction,
        "degree": result.degree,
        "reason": result.reason,
        "dominant_direction": dir_labels[dom_dir],
        "dominant_degree": deg_labels[dom_deg],
        "direction_percent": {k: v * 100 for k, v in result.direction.items()},
        "degree_percent": {k: v * 100 for k, v in result.degree.items()},
    }


def analyze_file(filepath: str, use_huggingface: bool = True) -> dict:
    """
    Analyze bias in a text file.
    
    Args:
        filepath: Path to text file
        use_huggingface: If True, auto-download from HuggingFace if no local model
        
    Returns:
        Same as quick_analyze()
    """
    with open(filepath, 'r') as f:
        text = f.read()
    return quick_analyze(text, use_huggingface)


__all__ = [
    "BiasEngine",
    "BiasResult", 
    "quick_analyze",
    "analyze_file",
    "cli_main",
    "run_server",
    "__version__",
]
