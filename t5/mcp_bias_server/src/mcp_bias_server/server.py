#!/usr/bin/env python3
"""
MCP Bias Server - Model Context Protocol server for political bias analysis.

Provides tools for detecting political bias in text using a fine-tuned T5 model.
This server implements the MCP protocol for integration with AI assistants.

Usage:
    python -m mcp_bias_server
    bias-mcp-server

Environment Variables:
    BIAS_MODEL_PATH: Path to LoRA adapter (default: ./bias-detector-output)
    BIAS_BASE_MODEL: Base T5 model name (default: t5-large)
    BIAS_DEVICE: Force device selection (auto/mps/cuda/cpu)
"""

import os
import sys
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("bias-mcp-server")

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Local imports
from mcp_bias_server.bias_engine import BiasEngine, BiasResult

# Create MCP server instance
server = Server("bias-mcp-server")

# Global engine instance
_engine: Optional[BiasEngine] = None


def get_engine() -> BiasEngine:
    """Get or create the bias engine singleton."""
    global _engine
    if _engine is None:
        # Let BiasEngine use bundled model by default (or BIAS_MODEL_PATH env)
        model_path = os.getenv("BIAS_MODEL_PATH")
        base_model = os.getenv("BIAS_BASE_MODEL", "t5-large")
        device = os.getenv("BIAS_DEVICE")
        _engine = BiasEngine(model_path, base_model, device, lazy_load=True)
    return _engine


@server.list_tools()
async def list_tools() -> List[Tool]:
    """
    List available MCP tools.
    
    Returns:
        List of Tool objects representing available bias analysis tools.
    """
    return [
        Tool(
            name="analyze_bias",
            description="""Analyze text for political bias using a fine-tuned T5 model.

Returns a structured analysis with:
- Direction scores (L=Left, C=Center, R=Right) - values sum to ~1.0
- Degree scores (L=Low, M=Medium, H=High) - values sum to ~1.0  
- Reasoning explanation for the classification

The model is trained on news articles and detects political leaning
in the text content. Best results with news articles, opinion pieces,
or political commentary.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze for political bias. Should be a news article, opinion piece, or political commentary for best results."
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="analyze_batch",
            description="""Analyze multiple texts for political bias in a single call.

More efficient than multiple analyze_bias calls when processing
multiple articles. Returns a list of bias analysis results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze for political bias"
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="get_model_info",
            description="""Get information about the loaded T5 bias detection model.

Returns details about:
- Model architecture (T5-large with LoRA adapters)
- Device being used (MPS/CUDA/CPU)
- Input/output format specifications
- Model capabilities and limitations""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Execute a tool call.
    
    Args:
        name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        List of TextContent objects with results
    """
    try:
        engine = get_engine()
        
        if name == "analyze_bias":
            text = arguments.get("text", "")
            if not text or not text.strip():
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Text cannot be empty",
                        "success": False
                    }, indent=2)
                )]
            
            logger.info(f"Analyzing text of length {len(text)}")
            result = engine.analyze(text)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "result": result.to_dict()
                }, indent=2)
            )]
        
        elif name == "analyze_batch":
            texts = arguments.get("texts", [])
            if not texts:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Texts list cannot be empty",
                        "success": False
                    }, indent=2)
                )]
            
            logger.info(f"Analyzing batch of {len(texts)} texts")
            results = engine.analyze_batch(texts)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(results),
                    "results": [r.to_dict() for r in results]
                }, indent=2)
            )]
        
        elif name == "get_model_info":
            info = engine.get_model_info()
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "model_info": info
                }, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Unknown tool: {name}",
                    "success": False,
                    "available_tools": ["analyze_bias", "analyze_batch", "get_model_info"]
                }, indent=2)
            )]
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Model files not found: {str(e)}",
                "success": False,
                "hint": "Ensure BIAS_MODEL_PATH points to valid LoRA adapter directory"
            }, indent=2)
        )]
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            }, indent=2)
        )]


async def run_async():
    """Run the MCP server asynchronously."""
    logger.info("Starting Bias MCP Server...")
    
    # Log configuration
    model_path = os.getenv("BIAS_MODEL_PATH", "./bias-detector-output")
    base_model = os.getenv("BIAS_BASE_MODEL", "t5-large")
    device = os.getenv("BIAS_DEVICE", "auto")
    logger.info(f"Configuration: model_path={model_path}, base_model={base_model}, device={device}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def run_server():
    """Entry point for running the MCP server."""
    asyncio.run(run_async())


if __name__ == "__main__":
    run_server()