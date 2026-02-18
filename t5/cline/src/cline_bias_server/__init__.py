"""
Cline Bias Server - MCP server for political bias detection.

A comprehensive Model Context Protocol (MCP) server that provides
political bias detection using fine-tuned T5 models with LoRA adapters.

Quick Start:
    # Install
    pip install cline-bias-server
    
    # CLI usage
    bias "Your text here"
    
    # Python API
    from cline_bias_server import quick_analyze
    result = quick_analyze("Your text")

MCP Server Usage:
    # Start the MCP server
    cline-bias-server

Features:
    - Political bias detection (Left/Center/Right)
    - Bias intensity analysis (Low/Medium/High)
    - Batch processing support
    - Multiple output formats (JSON, text, detailed)
    - Resource management for efficient memory usage
    - Comprehensive error handling
    - Full MCP protocol support (tools, resources, prompts)
"""

__version__ = "1.0.0"
__author__ = "Cline Team"
__email__ = "cline-bias@example.com"
__license__ = "MIT"

# Lazy imports to avoid circular import issues
# The bias_engine and server modules are imported on-demand

_engine = None


def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name in ("BiasEngine", "BiasResult", "BiasAnalyzer"):
        from cline_bias_server import bias_engine
        return getattr(bias_engine, name)
    elif name == "quick_analyze":
        return _get_quick_analyze()
    elif name == "analyze_file":
        return _get_analyze_file()
    elif name == "analyze_batch":
        return _get_analyze_batch()
    elif name == "reset_engine":
        return _get_reset_engine()
    elif name == "run_server":
        from cline_bias_server.server import run_server
        return run_server
    elif name == "create_server":
        from cline_bias_server.server import create_server
        return create_server
    elif name == "cli_main":
        from cline_bias_server.cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_quick_analyze():
    """Get quick_analyze function with proper closure over _engine."""
    global _engine
    
    def quick_analyze(
        text: str,
        use_huggingface: bool = True,
        model_path: str = None,
        base_model: str = "t5-large"
    ) -> dict:
        global _engine
        from cline_bias_server.bias_engine import BiasEngine
        
        if _engine is None:
            _engine = BiasEngine(
                model_path=model_path,
                base_model_name=base_model,
                use_huggingface=use_huggingface
            )
        
        result = _engine.analyze(text)
        
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
            "direction_percent": {k: round(v * 100, 1) for k, v in result.direction.items()},
            "degree_percent": {k: round(v * 100, 1) for k, v in result.degree.items()},
            "raw_output": result.raw_output,
            "device": result.device,
        }
    
    return quick_analyze


def _get_analyze_file():
    """Get analyze_file function."""
    def analyze_file(
        filepath: str,
        use_huggingface: bool = True,
        model_path: str = None
    ) -> dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        quick_analyze = _get_quick_analyze()
        return quick_analyze(text, use_huggingface, model_path)
    
    return analyze_file


def _get_analyze_batch():
    """Get analyze_batch function."""
    global _engine
    
    def analyze_batch(
        texts: list,
        use_huggingface: bool = True,
        model_path: str = None
    ) -> list:
        global _engine
        from cline_bias_server.bias_engine import BiasEngine
        
        if _engine is None:
            _engine = BiasEngine(
                model_path=model_path,
                use_huggingface=use_huggingface
            )
        
        return _engine.analyze_batch(texts)
    
    return analyze_batch


def _get_reset_engine():
    """Get reset_engine function."""
    global _engine
    
    def reset_engine() -> None:
        global _engine
        if _engine is not None:
            _engine.unload()
            _engine = None
    
    return reset_engine


__all__ = [
    "BiasEngine",
    "BiasResult", 
    "BiasAnalyzer",
    "quick_analyze",
    "analyze_file",
    "analyze_batch",
    "reset_engine",
    "run_server",
    "create_server",
    "cli_main",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
