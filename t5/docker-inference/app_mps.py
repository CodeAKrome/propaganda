#!/usr/bin/env python
"""
FastAPI wrapper for BiasDetectorInference (MPS version)
"""

import os
import sys
import json
import torch
import logging
import warnings
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Suppress library noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# Global model instance
model_instance = None

class PredictRequest(BaseModel):
    text: str
    model_path: Optional[str] = "./bias-detector-output"
    base_model_name: Optional[str] = "t5-large"

class PredictResponse(BaseModel):
    result: dict
    device: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model_instance
    # Model will be loaded lazily on first request
    yield
    # Cleanup
    if model_instance is not None:
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(
    title="Bias Detector API (MPS)",
    description="Political bias detection using T5 with LoRA adapters - MPS version",
    version="1.0.0",
    lifespan=lifespan
)

def get_model(model_path: str, base_model_name: str):
    """Get or create model instance"""
    global model_instance
    
    if model_instance is None:
        # Device selection: CUDA > MPS (if available) > CPU
        # Note: MPS requires macOS with Apple Silicon and proper PyTorch build
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
        
        tokenizer = T5Tokenizer.from_pretrained(base_model_name, verbose=False)
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_name, 
            low_cpu_mem_usage=True
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)
        model.eval()
        
        model_instance = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
    
    return model_instance

def predict_bias(text: str, model_path: str, base_model_name: str) -> dict:
    """Run bias prediction"""
    instance = get_model(model_path, base_model_name)
    
    formatted_input = f"classify political bias as json: {text}"
    
    inputs = instance["tokenizer"](
        formatted_input, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(instance["device"])
    
    with torch.no_grad():
        outputs = instance["model"].generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    raw_result = instance["tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()
    
    # Clean and parse JSON
    try:
        if raw_result.startswith('"') and raw_result.endswith('"'):
            decoded_string = json.loads(raw_result)
        else:
            decoded_string = raw_result
        
        if not decoded_string.startswith('{'):
            decoded_string = "{" + decoded_string + "}"
        
        final_json = json.loads(decoded_string)
        
    except Exception:
        cleaned = raw_result.replace('\\"', '"').strip('"')
        if not cleaned.startswith('{'):
            cleaned = "{" + cleaned + "}"
        try:
            final_json = json.loads(cleaned)
        except json.JSONDecodeError:
            final_json = {"raw_output": raw_result}
    
    return final_json

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0", "mode": "MPS"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict political bias for given text.
    
    Returns JSON with bias classification.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predict_bias(
            request.text, 
            request.model_path or "./bias-detector-output", 
            request.base_model_name or "t5-large"
        )
        instance = get_model(
            request.model_path or "./bias-detector-output", 
            request.base_model_name or "t5-large"
        )
        
        return PredictResponse(
            result=result,
            device=instance["device"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/raw")
async def predict_raw(request: PredictRequest):
    """
    Predict political bias and return raw JSON response.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predict_bias(
            request.text, 
            request.model_path or "./bias-detector-output", 
            request.base_model_name or "t5-large"
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
