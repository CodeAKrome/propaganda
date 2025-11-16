import argparse
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
from typing import Optional

# Parse command line arguments
parser = argparse.ArgumentParser(description='Ollama FastAPI Microservice')
parser.add_argument('--port', type=int, default=8101, help='Port to run the service on (default: 8101)')
parser.add_argument('--model', type=str, default='gpt-oss:20b', help='Ollama model to use (default: gpt-oss:20b)')
parser.add_argument('--ollama-url', type=str, default='http://host.docker.internal:11434', help='Ollama API URL (default: http://host.docker.internal:11434)')
args = parser.parse_args()

app = FastAPI(
    title="Ollama Microservice",
    description="FastAPI microservice for Ollama completions",
    version="1.0.0"
)

# Request model
class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# Response model
class PromptResponse(BaseModel):
    response: str
    model: str
    done: bool

@app.get("/")
async def root():
    return {
        "message": "Ollama FastAPI Microservice",
        "default_model": args.model,
        "endpoints": {
            "/generate": "POST - Generate completion from prompt",
            "/health": "GET - Health check",
            "/models": "GET - List available models"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the service and Ollama are running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{args.ollama_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "ollama": "connected"}
            else:
                return {"status": "degraded", "ollama": "unreachable"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{args.ollama_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=502, detail="Failed to fetch models from Ollama")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")

@app.post("/generate", response_model=PromptResponse)
async def generate_completion(request: PromptRequest):
    """Generate a completion from a prompt using Ollama"""
    model = request.model or args.model
    
    # Build Ollama request
    ollama_request = {
        "model": model,
        "prompt": request.prompt,
        "stream": False
    }
    
    # Add optional parameters if provided
    if request.temperature is not None:
        ollama_request["temperature"] = request.temperature
    
    options = {}
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    
    if options:
        ollama_request["options"] = options
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{args.ollama_url}/api/generate",
                json=ollama_request,
                timeout=300.0  # 5 minute timeout for generation
            )
            
            if response.status_code == 200:
                result = response.json()
                return PromptResponse(
                    response=result.get("response", ""),
                    model=result.get("model", model),
                    done=result.get("done", True)
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama error: {response.text}"
                )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    print(f"Starting Ollama FastAPI Microservice on port {args.port}")
    print(f"Default model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

