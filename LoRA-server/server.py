#!/usr/bin/env python3

"""
LoRA Server - Serve Fine-tuned Models via FastAPI
==================================================
Serves LoRA-adapted models locally with FastAPI + Uvicorn.

Supports:
- LoRA adapters merged with base models
- PEFT format models
- Dynamic model loading/unloading

Usage:
    python server.py --model-path ./lora-output --port 8080
    python server.py --model-path ./lora-output --port 1337 --host 0.0.0.0
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


model_instance = None
tokenizer_instance = None
model_config = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    max_tokens: Optional[int] = Field(512, ge=1, le=2048)
    temperature: Optional[float] = Field(0.3, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    result: Dict[str, Any]
    device: str
    model_path: str


class BatchRequest(BaseModel):
    texts: List[str]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: Optional[str] = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_path: str):
    global model_instance, tokenizer_instance, model_config

    logger.info(f"Loading model from {model_path}")

    device = get_device()
    logger.info(f"Using device: {device}")

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    logger.info(f"Base model: {base_model_name}")

    if peft_config.task_type == "CAUSAL_LM":
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_instance = model
    tokenizer_instance = tokenizer
    model_config = {
        "base_model": base_model_name,
        "task_type": peft_config.task_type,
        "device": device,
    }

    logger.info("Model loaded successfully!")


def generate(
    prompt: str, max_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9
) -> str:
    if model_instance is None or tokenizer_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer_instance(
        prompt, return_tensors="pt", max_length=2048, truncation=True
    )

    if model_config["device"] != "cpu":
        inputs = {k: v.to(model_config["device"]) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_instance.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer_instance.pad_token_id,
            eos_token_id=tokenizer_instance.eos_token_id,
        )

    generated = tokenizer_instance.decode(outputs[0], skip_special_tokens=True)

    if prompt in generated:
        return generated[len(prompt) :].strip()
    return generated.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance, tokenizer_instance
    yield
    if model_instance is not None:
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="LoRA Model Server",
    description="Serve fine-tuned LoRA models locally",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        model_loaded=model_instance is not None,
        device=get_device(),
        model_path=model_config.get("base_model") if model_config else None,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_instance else "error",
        model_loaded=model_instance is not None,
        device=get_device(),
        model_path=model_config.get("base_model") if model_config else None,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        logger.info(f"Analyzing text of length: {len(request.text)}")

        prompt = f"""Analyze the political bias of this article. 
Return JSON with direction (L/C/R with probabilities) and degree (L/M/H with probabilities).

Article: {request.text[:2000]}

Response:"""

        result = generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = {"raw_response": result, "error": "Failed to parse JSON"}

        return AnalyzeResponse(
            result=parsed,
            device=model_config["device"] if model_config else get_device(),
            model_path=model_config["base_model"] if model_config else "unknown",
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def analyze_batch(request: BatchRequest):
    results = []

    for text in request.texts:
        try:
            prompt = f"""Analyze the political bias of this article. 
Return JSON with direction and degree.

Article: {text[:2000]}

Response:"""

            result = generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = {"raw": result, "error": "parse_failed"}

            results.append(parsed)

        except Exception as e:
            results.append({"error": str(e)})

    return JSONResponse(content=results)


@app.post("/generate")
async def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.3):
    try:
        result = generate(prompt, max_tokens, temperature)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    return {
        "loaded": model_config.get("base_model") if model_config else None,
        "device": model_config.get("device") if model_config else None,
        "task_type": model_config.get("task_type") if model_config else None,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Server")
    parser.add_argument("--model-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    args = parser.parse_args()

    import uvicorn

    logger.info(f"Loading model: {args.model_path}")
    load_model(args.model_path)

    logger.info(f"Starting server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
