#!/usr/bin/env python3
"""
Cline Bias Server CLI - Command-line interface for bias detection.

This module provides a command-line interface for the bias detection server.
It supports various output formats and options for easy integration.

Usage:
    # Analyze text
    bias "Your text here"
    
    # Analyze file
    bias --file article.txt
    
    # Batch mode
    bias --batch file1.txt file2.txt
    
    # JSON output
    bias --json "Your text"
    
    # Server mode
    bias --server

Author: Cline Team
License: MIT
"""

from __future__ import annotations

import sys
import os
import json
import argparse
from typing import Optional, List
from pathlib import Path

# Import local modules
from cline_bias_server import BiasEngine, BiasResult, BiasAnalyzer, reset_engine


def format_result(result: BiasResult, output_format: str = "json") -> str:
    """
    Format a bias result for display.
    
    Args:
        result: The BiasResult to format
        output_format: Format type (json, text, compact)
        
    Returns:
        Formatted string
    """
    if output_format == "text":
        return result.to_text()
    elif output_format == "compact":
        data = result.to_dict(format="compact")
        return json.dumps(data, indent=2)
    else:  # json
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def analyze_text(
    text: str,
    model_path: Optional[str] = None,
    base_model: str = "t5-large",
    device: Optional[str] = None,
    output_format: str = "json",
    use_huggingface: bool = True
) -> None:
    """
    Analyze text for bias and print result.
    
    Args:
        text: Text to analyze
        model_path: Path to model
        base_model: Base model name
        device: Device to use
        output_format: Output format
        use_huggingface: Whether to use HuggingFace
    """
    engine = BiasEngine(
        model_path=model_path,
        base_model_name=base_model,
        device=device,
        lazy_load=False,
        use_huggingface=use_huggingface
    )
    
    try:
        result = engine.analyze(text)
        print(format_result(result, output_format))
    finally:
        engine.unload()


def analyze_file(
    filepath: str,
    model_path: Optional[str] = None,
    base_model: str = "t5-large",
    device: Optional[str] = None,
    output_format: str = "json",
    encoding: str = "utf-8"
) -> None:
    """
    Analyze a file for bias and print result.
    
    Args:
        filepath: Path to file
        model_path: Path to model
        base_model: Base model name
        device: Device to use
        output_format: Output format
        encoding: File encoding
    """
    with open(filepath, 'r', encoding=encoding) as f:
        text = f.read()
    
    analyze_text(text, model_path, base_model, device, output_format)


def analyze_batch(
    filepaths: List[str],
    model_path: Optional[str] = None,
    base_model: str = "t5-large",
    device: Optional[str] = None,
    output_format: str = "json"
) -> None:
    """
    Analyze multiple files for bias and print results.
    
    Args:
        filepaths: List of file paths
        model_path: Path to model
        base_model: Base model name
        device: Device to use
        output_format: Output format
    """
    engine = BiasEngine(
        model_path=model_path,
        base_model_name=base_model,
        device=device,
        lazy_load=False
    )
    
    try:
        # Load all texts
        texts = []
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        # Analyze batch
        results = engine.analyze_batch(texts)
        
        # Get statistics
        analyzer = BiasAnalyzer(model_path=model_path, base_model=base_model, device=device)
        analyzer.engine = engine
        stats = analyzer.get_statistics(results)
        
        # Format output
        if output_format == "compact":
            formatted_results = [r.to_dict(format="compact") for r in results]
        else:
            formatted_results = [r.to_dict() for r in results]
        
        output = {
            "count": len(results),
            "results": formatted_results,
            "statistics": stats
        }
        
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    finally:
        engine.unload()


def run_server(
    model_path: Optional[str] = None,
    base_model: str = "t5-large",
    device: Optional[str] = None
) -> None:
    """
    Run the MCP server.
    
    Args:
        model_path: Path to model
        base_model: Base model name
        device: Device to use
    """
    from cline_bias_server.server import run_server as mcp_run_server
    
    # Set environment variables if provided
    if model_path:
        os.environ["BIAS_MODEL_PATH"] = model_path
    if base_model:
        os.environ["BIAS_BASE_MODEL"] = base_model
    if device:
        os.environ["BIAS_DEVICE"] = device
    
    mcp_run_server()


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="cline-bias",
        description="Political bias detection using T5 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text
  cline-bias "The president's new policy is controversial..."
  
  # Analyze file
  cline-bias --file article.txt
  
  # Batch analysis
  cline-bias --batch article1.txt article2.txt article3.txt
  
  # Text output
  cline-bias --format text "Your text here"
  
  # Use specific model
  cline-bias --model-path ./my-model "Text"
  
  # Run MCP server
  cline-bias --server
        """
    )
    
    # Positional argument for text
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze for political bias"
    )
    
    # Options
    parser.add_argument(
        "-f", "--file",
        help="Path to text file to analyze"
    )
    
    parser.add_argument(
        "-b", "--batch",
        nargs="+",
        help="Analyze multiple files"
    )
    
    parser.add_argument(
        "-m", "--model-path",
        help="Path to LoRA adapter model"
    )
    
    parser.add_argument(
        "--base-model",
        default="t5-large",
        help="Base T5 model name (default: t5-large)"
    )
    
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "-o", "--format",
        choices=["json", "text", "compact"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "-e", "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
    )
    
    parser.add_argument(
        "-s", "--server",
        action="store_true",
        help="Run as MCP server"
    )
    
    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Disable auto-download from HuggingFace Hub"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version information"
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.
    
    Args:
        argv: Command-line arguments (default: sys.argv)
        
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Version check
    if args.version:
        from cline_bias_server import __version__
        print(f"cline-bias-server v{__version__}")
        return 0
    
    # Server mode
    if args.server:
        run_server(args.model_path, args.base_model, args.device)
        return 0
    
    # Determine what to analyze
    text = args.text
    filepath = args.file
    batch = args.batch
    
    # Validation
    if not text and not filepath and not batch:
        parser.print_help()
        return 1
    
    if text and (filepath or batch):
        print("Error: Cannot specify both text and file/batch arguments", file=sys.stderr)
        return 1
    
    # Prepare options
    model_path = args.model_path or os.getenv("BIAS_MODEL_PATH")
    device = args.device or os.getenv("BIAS_DEVICE")
    use_huggingface = not args.no_huggingface
    
    try:
        if batch:
            analyze_batch(
                batch,
                model_path=model_path,
                base_model=args.base_model,
                device=device,
                output_format=args.format
            )
        elif filepath:
            analyze_file(
                filepath,
                model_path=model_path,
                base_model=args.base_model,
                device=device,
                output_format=args.format,
                encoding=args.encoding
            )
        else:
            analyze_text(
                text,
                model_path=model_path,
                base_model=args.base_model,
                device=device,
                output_format=args.format,
                use_huggingface=use_huggingface
            )
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
