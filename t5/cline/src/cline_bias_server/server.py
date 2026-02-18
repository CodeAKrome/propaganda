#!/usr/bin/env python3
"""
Cline Bias MCP Server - Model Context Protocol server for political bias detection.

This module implements a complete MCP server for political bias detection
using fine-tuned T5 models. It follows the MCP specification and provides
tools for AI assistants to analyze text for political bias.

MCP Protocol Features:
    - Tools: analyze_bias, analyze_batch, get_model_info, analyze_url
    - Resources: model-info, bias-schemas
    - Prompts: analyze-article, compare-bias

Installation:
    pip install cline-bias-server
    
Usage:
    # As MCP server (stdio mode)
    cline-bias-server
    
    # In Python
    from cline_bias_server import run_server
    run_server()

Environment Variables:
    BIAS_MODEL_PATH: Path to LoRA adapter (default: ./bias-detector-output)
    BIAS_BASE_MODEL: Base T5 model name (default: t5-large)
    BIAS_DEVICE: Force device selection (auto/mps/cuda/cpu)
    BIAS_USE_HUGGINGFACE: Set to "0" to disable auto-download from HuggingFace

Author: Cline Team
License: MIT
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("cline-bias-server")

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource, Resource
    from mcp.server.session import Session
except ImportError as e:
    logger.error(f"MCP SDK not installed: {e}")
    logger.info("Install with: pip install mcp")
    sys.exit(1)

# Local imports
from cline_bias_server.bias_engine import BiasEngine, BiasResult, BiasAnalyzer

# Server metadata
SERVER_NAME = "cline-bias-server"
SERVER_VERSION = "1.0.0"
SERVER_DESCRIPTION = "MCP server for political bias detection using T5 with LoRA"


class BiasServer:
    """
    MCP Server for political bias detection.
    
    This class manages the MCP server lifecycle and tool execution.
    It provides a clean interface between the MCP protocol and the BiasEngine.
    
    Attributes:
        server: The MCP Server instance
        engine: The BiasEngine instance
        
    Example:
        >>> bias_server = BiasServer()
        >>> await bias_server.run()
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "t5-large",
        device: Optional[str] = None
    ):
        """
        Initialize the bias server.
        
        Args:
            model_path: Path to LoRA adapter weights
            base_model: Base T5 model name
            device: Force specific device
        """
        self.server = Server(SERVER_NAME)
        self.engine: Optional[BiasEngine] = None
        self.analyzer: Optional[BiasAnalyzer] = None
        
        # Configuration
        self.model_path = model_path or os.getenv("BIAS_MODEL_PATH")
        self.base_model = base_model or os.getenv("BIAS_BASE_MODEL", "t5-large")
        self.device = device or os.getenv("BIAS_DEVICE")
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"Initialized {SERVER_NAME} v{SERVER_VERSION}")
    
    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""
        self.server.list_tools(self._handle_list_tools)
        self.server.call_tool(self._handle_call_tool)
        
        # Optional: Register resources
        try:
            self.server.list_resources(self._handle_list_resources)
            self.server.read_resource(self._handle_read_resource)
        except AttributeError:
            # Resources may not be available in all MCP versions
            pass
        
        # Optional: Register prompts
        try:
            self.server.list_prompts(self._handle_list_prompts)
            self.server.get_prompt(self._handle_get_prompt)
        except AttributeError:
            # Prompts may not be available in all MCP versions
            pass
    
    def _get_engine(self) -> BiasEngine:
        """Get or create the bias engine."""
        if self.engine is None:
            self.engine = BiasEngine(
                model_path=self.model_path,
                base_model_name=self.base_model,
                device=self.device,
                lazy_load=True
            )
        return self.engine
    
    def _get_analyzer(self) -> BiasAnalyzer:
        """Get or create the bias analyzer."""
        if self.analyzer is None:
            self.analyzer = BiasAnalyzer(
                model_path=self.model_path,
                base_model=self.base_model,
                device=self.device
            )
        return self.analyzer
    
    async def _handle_list_tools(self) -> List[Tool]:
        """
        Handle MCP list_tools request.
        
        Returns:
            List of available MCP tools
        """
        return [
            Tool(
                name="analyze_bias",
                description="""Analyze text for political bias using a fine-tuned T5 model.

Returns a structured analysis with:
- Direction scores (L=Left, C=Center, R=Right) - values sum to ~1.0
- Degree scores (L=Low, M=Medium, H=High) - values sum to ~1.0  
- Reasoning explanation for the classification
- Confidence score (0-1)

The model is trained on news articles and detects political leaning
in the text content. Best results with news articles, opinion pieces,
or political commentary.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze for political bias. Should be a news article, opinion piece, or political commentary for best results."
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["json", "compact", "text"],
                            "default": "json",
                            "description": "Output format: json (full), compact (minimal), text (readable)"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="analyze_batch",
                description="""Analyze multiple texts for political bias in a single call.

More efficient than multiple analyze_bias calls when processing
multiple articles. Returns a list of bias analysis results with
summary statistics.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to analyze for political bias"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["json", "compact"],
                            "default": "json",
                            "description": "Output format for results"
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
- Model capabilities and limitations
- Input/output format specifications""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="analyze_file",
                description="""Analyze a text file for political bias.

Reads the file content and analyzes it for political bias.
Supports .txt, .md, and other text formats.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the text file to analyze"
                        },
                        "encoding": {
                            "type": "string",
                            "default": "utf-8",
                            "description": "File encoding (default: utf-8)"
                        }
                    },
                    "required": ["filepath"]
                }
            ),
            Tool(
                name="compare_bias",
                description="""Compare political bias between multiple texts.

Analyzes multiple texts and provides a comparison showing
differences in political direction and degree across the texts.
Useful for comparing coverage of the same topic by different sources.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to compare"
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional labels for each text (e.g., source names)"
                        }
                    },
                    "required": ["texts"]
                }
            ),
            Tool(
                name="unload_model",
                description="""Unload the model from memory.

Use this to free up GPU/system memory when the server is not
actively processing requests. The model will be automatically
reloaded on the next analysis request.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
    
    async def _handle_call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """
        Handle MCP tool call request.
        
        Args:
            name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            List of TextContent objects with results
        """
        try:
            if name == "analyze_bias":
                return await self._handle_analyze_bias(arguments)
            elif name == "analyze_batch":
                return await self._handle_analyze_batch(arguments)
            elif name == "get_model_info":
                return await self._handle_get_model_info()
            elif name == "analyze_file":
                return await self._handle_analyze_file(arguments)
            elif name == "compare_bias":
                return await self._handle_compare_bias(arguments)
            elif name == "unload_model":
                return await self._handle_unload_model()
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown tool: {name}",
                        "success": False,
                        "available_tools": [
                            "analyze_bias",
                            "analyze_batch", 
                            "get_model_info",
                            "analyze_file",
                            "compare_bias",
                            "unload_model"
                        ]
                    }, indent=2)
                )]
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"File not found: {str(e)}",
                    "success": False
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
    
    async def _handle_analyze_bias(
        self,
        arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle analyze_bias tool call."""
        text = arguments.get("text", "")
        output_format = arguments.get("output_format", "json")
        
        if not text or not text.strip():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Text cannot be empty",
                    "success": False
                }, indent=2)
            )]
        
        logger.info(f"Analyzing text of length {len(text)}")
        
        engine = self._get_engine()
        result = engine.analyze(text)
        
        # Format output based on request
        if output_format == "compact":
            output = result.to_dict(format="compact")
        elif output_format == "text":
            output = result.to_text()
        else:
            output = result.to_dict(include_raw=False)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "result": output
            }, indent=2, ensure_ascii=False)
        )]
    
    async def _handle_analyze_batch(
        self,
        arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle analyze_batch tool call."""
        texts = arguments.get("texts", [])
        output_format = arguments.get("output_format", "json")
        
        if not texts:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Texts list cannot be empty",
                    "success": False
                }, indent=2)
            )]
        
        logger.info(f"Analyzing batch of {len(texts)} texts")
        
        engine = self._get_engine()
        results = engine.analyze_batch(texts)
        
        # Get statistics
        analyzer = self._get_analyzer()
        stats = analyzer.get_statistics(results)
        
        # Format results
        if output_format == "compact":
            formatted_results = [r.to_dict(format="compact") for r in results]
        else:
            formatted_results = [r.to_dict() for r in results]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "count": len(results),
                "results": formatted_results,
                "statistics": stats
            }, indent=2, ensure_ascii=False)
        )]
    
    async def _handle_get_model_info(self) -> List[TextContent]:
        """Handle get_model_info tool call."""
        engine = self._get_engine()
        info = engine.get_model_info()
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "server_version": SERVER_VERSION,
                "model_info": info
            }, indent=2)
        )]
    
    async def _handle_analyze_file(
        self,
        arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle analyze_file tool call."""
        filepath = arguments.get("filepath", "")
        encoding = arguments.get("encoding", "utf-8")
        
        if not filepath:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Filepath cannot be empty",
                    "success": False
                }, indent=2)
            )]
        
        logger.info(f"Analyzing file: {filepath}")
        
        analyzer = self._get_analyzer()
        result = analyzer.analyze_file(filepath, encoding)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "filepath": filepath,
                "result": result.to_dict()
            }, indent=2, ensure_ascii=False)
        )]
    
    async def _handle_compare_bias(
        self,
        arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle compare_bias tool call."""
        texts = arguments.get("texts", [])
        labels = arguments.get("labels")
        
        if not texts:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Texts list cannot be empty",
                    "success": False
                }, indent=2)
            )]
        
        # Generate labels if not provided
        if not labels:
            labels = [f"Text {i+1}" for i in range(len(texts))]
        
        if len(labels) != len(texts):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Number of labels must match number of texts",
                    "success": False
                }, indent=2)
            )]
        
        logger.info(f"Comparing bias across {len(texts)} texts")
        
        engine = self._get_engine()
        results = engine.analyze_batch(texts)
        
        # Build comparison
        comparison = {
            "items": [
                {
                    "label": label,
                    "direction": r.direction,
                    "degree": r.degree,
                    "dominant_direction": r.dominant_direction,
                    "dominant_degree": r.dominant_degree,
                    "confidence": r.confidence
                }
                for label, r in zip(labels, results)
            ]
        }
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "comparison": comparison
            }, indent=2, ensure_ascii=False)
        )]
    
    async def _handle_unload_model(self) -> List[TextContent]:
        """Handle unload_model tool call."""
        if self.engine is not None:
            self.engine.unload()
            self.engine = None
            logger.info("Model unloaded")
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "message": "Model unloaded successfully"
            }, indent=2)
        )]
    
    async def _handle_list_resources(self) -> List[Resource]:
        """Handle list_resources request."""
        return [
            Resource(
                uri="bias://model-info",
                name="Model Information",
                description="Information about the loaded bias detection model"
            ),
            Resource(
                uri="bias://schemas/result",
                name="Bias Result Schema",
                "description": "JSON schema for bias analysis results"
            )
        ]
    
    async def _handle_read_resource(self, uri: str) -> str:
        """Handle read_resource request."""
        if uri == "bias://model-info":
            engine = self._get_engine()
            info = engine.get_model_info()
            return json.dumps(info, indent=2)
        elif uri == "bias://schemas/result":
            return json.dumps({
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "object",
                        "properties": {
                            "L": {"type": "number"},
                            "C": {"type": "number"},
                            "R": {"type": "number"}
                        }
                    },
                    "degree": {
                        "type": "object", 
                        "properties": {
                            "L": {"type": "number"},
                            "M": {"type": "number"},
                            "H": {"type": "number"}
                        }
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }, indent=2)
        
        raise ValueError(f"Unknown resource: {uri}")
    
    async def _handle_list_prompts(self) -> List[Dict[str, Any]]:
        """Handle list_prompts request."""
        return [
            {
                "name": "analyze-article",
                "description": "Analyze an article for political bias",
                "arguments": [
                    {"name": "article_text", "required": True}
                ]
            },
            {
                "name": "compare-sources",
                "description": "Compare bias between news sources",
                "arguments": [
                    {"name": "source1", "required": True},
                    {"name": "source2", "required": True}
                ]
            }
        ]
    
    async def _handle_get_prompt(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_prompt request."""
        if name == "analyze-article":
            article_text = arguments.get("article_text", "")
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Please analyze the following article for political bias:

{article_text}

Provide your analysis including:
1. Political direction (Left, Center, or Right)
2. Bias intensity (Low, Medium, or High)
3. Reasoning for your classification"""
                    }
                ]
            }
        
        raise ValueError(f"Unknown prompt: {name}")
    
    async def run(self) -> None:
        """Run the MCP server."""
        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}...")
        
        # Log configuration
        model_path = self.model_path or os.getenv("BIAS_MODEL_PATH", "./bias-detector-output")
        base_model = self.base_model
        device = self.device or "auto"
        
        logger.info(f"Configuration: model_path={model_path}, base_model={base_model}, device={device}")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def create_server(
    model_path: Optional[str] = None,
    base_model: str = "t5-large",
    device: Optional[str] = None
) -> BiasServer:
    """
    Factory function to create a BiasServer instance.
    
    Args:
        model_path: Path to LoRA adapter weights
        base_model: Base T5 model name
        device: Force specific device
        
    Returns:
        Configured BiasServer instance
        
    Example:
        >>> server = create_server(device="cuda")
        >>> asyncio.run(server.run())
    """
    return BiasServer(model_path, base_model, device)


def run_server() -> None:
    """
    Entry point for running the MCP server.
    
    This is the main function called when running the server
    from the command line.
    
    Example:
        >>> # From command line
        >>> cline-bias-server
    """
    server = create_server()
    asyncio.run(server.run())


# Main entry point
if __name__ == "__main__":
    run_server()
