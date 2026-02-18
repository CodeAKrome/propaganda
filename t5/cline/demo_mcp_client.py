#!/usr/bin/env python3
"""
demo_mcp_client.py - Install and use the bias-mcp-server MCP server from Python.

This script demonstrates how to:
1. Install the bias-mcp-server package
2. Connect to the MCP server as a client
3. Call the MCP tools from Python code

Usage:
    python demo_mcp_client.py

Requirements:
    pip install mcp bias-mcp-server
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path

# MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class BiasMCPClient:
    """Client for connecting to the bias-mcp-server MCP server."""
    
    def __init__(self, model_path: str = "./bias-detector-output"):
        """
        Initialize the MCP client.
        
        Args:
            model_path: Path to the LoRA adapter weights
        """
        self.model_path = model_path
        self.session = None
        self.read_stream = None
        self.write_stream = None
        
    async def connect(self):
        """Connect to the MCP server."""
        # Server parameters - runs the bias-mcp-server as a subprocess
        server_params = StdioServerParameters(
            command=sys.executable,  # Use current Python interpreter
            args=["-m", "mcp_bias_server.server"],
            env={
                **os.environ,
                "BIAS_MODEL_PATH": self.model_path,
                "BIAS_BASE_MODEL": "t5-large",
                "BIAS_DEVICE": "auto"
            }
        )
        
        # Create the stdio client
        self.read_stream, self.write_stream = await stdio_client(server_params)
        self.session = ClientSession(self.read_stream, self.write_stream)
        
        # Initialize the session
        await self.session.initialize()
        
        return self
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.close()
    
    async def list_tools(self):
        """List available tools from the server."""
        tools = await self.session.list_tools()
        return tools
    
    async def analyze_bias(self, text: str) -> dict:
        """
        Analyze a single text for political bias.
        
        Args:
            text: The text to analyze
            
        Returns:
            Bias analysis result
        """
        result = await self.session.call_tool(
            "analyze_bias",
            arguments={"text": text}
        )
        return result
    
    async def analyze_batch(self, texts: list) -> dict:
        """
        Analyze multiple texts for political bias.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Batch analysis results
        """
        result = await self.session.call_tool(
            "analyze_batch",
            arguments={"texts": texts}
        )
        return result
    
    async def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        result = await self.session.call_tool(
            "get_model_info",
            arguments={}
        )
        return result


async def main():
    """Main demo function."""
    print("=" * 70)
    print("Bias MCP Server - Python Client Demo")
    print("=" * 70)
    print()
    
    # Check if bias-mcp-server is installed
    print("Step 1: Checking installation...")
    try:
        import mcp_bias_server
        print("  bias-mcp-server is installed")
    except ImportError:
        print("  bias-mcp-server not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bias-mcp-server"])
        print("  Installed successfully!")
    print()
    
    # Get model path
    model_path = os.getenv("BIAS_MODEL_PATH", "./bias-detector-output")
    print(f"Step 2: Model path: {model_path}")
    
    if not Path(model_path).exists():
        print(f"  WARNING: Model path does not exist!")
        print(f"  Please set BIAS_MODEL_PATH or train the model first.")
        print()
        print("  Continuing with demo (will fail if model not found)...")
    print()
    
    # Connect to the MCP server
    print("Step 3: Connecting to MCP server...")
    client = BiasMCPClient(model_path=model_path)
    
    try:
        await client.connect()
        print("  Connected successfully!")
        print()
        
        # List available tools
        print("Step 4: Listing available tools...")
        tools = await client.list_tools()
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        print()
        
        # Get model info
        print("Step 5: Getting model information...")
        try:
            info = await client.get_model_info()
            print(json.dumps(info.model_dump(), indent=2))
        except Exception as e:
            print(f"  Error: {e}")
        print()
        
        # Analyze sample texts
        print("Step 6: Analyzing sample texts...")
        sample_texts = [
            "The president announced today a sweeping new economic policy that critics argue favors large corporations over working families.",
            "The radical left continues to push their extreme agenda on hardworking taxpayers.",
            "Markets rallied as investors reacted positively to the Federal Reserve's announcement."
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n  Text {i}: {text[:60]}...")
            try:
                result = await client.analyze_bias(text)
                print(f"  Result: {json.dumps(result.model_dump(), indent=4)}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print()
        print("=" * 70)
        print("Demo complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"  Error connecting to server: {e}")
        print()
        print("  Make sure the model is trained and BIAS_MODEL_PATH is set correctly.")
        raise
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())