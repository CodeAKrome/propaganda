#!/usr/bin/env python3
"""
MLX Model Download and Benchmark Script
Downloads MLX models and tests them against Iran bias detection task.
Uses mlx2 conda environment.
"""

import os
import sys
import time
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

CONDA_SHELL = "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV = "mlx2"

MODELS_DIR = Path.home() / "Models" / "mlx"

MODELS = {
    "Qwen3.5-35B-A3B-4bit": {
        "repo": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "size_gb": 20,
        "type": "MoE",
        "active_params": "3B/35B",
    },
    "Hermes-4-70B-MLX-6bit": {
        "repo": "lmstudio-community/Hermes-4-70B-MLX-6bit",
        "size_gb": 57.3,
        "type": "Dense",
        "params": "71B",
    },
    "Qwen3.6-27B-OptiQ-4bit": {
        "repo": "mlx-optiq/Qwen3.6-27B-OptiQ-4bit",
        "size_gb": 15.7,
        "type": "Dense",
        "params": "27B",
    },
    "Gemma-4-31B-OptiQ-4bit": {
        "repo": "mlx-optiq/gemma-4-31B-it-OptiQ-4bit",
        "size_gb": 18.1,
        "type": "Dense",
        "params": "31B",
    },
    "Qwen2.5-32B-4bit": {
        "repo": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "size_gb": 20,
        "type": "Dense",
        "params": "32B",
    },
}

def conda_cmd(cmd, timeout=7200):
    """Run command in mlx2 conda environment"""
    full_cmd = f'source {CONDA_SHELL} && conda activate {CONDA_ENV} && {cmd}'
    result = subprocess.run(
        ["zsh", "-c", full_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent)
    )
    return result

def download_model(model_name: str, repo: str) -> dict:
    """Download a model using mlx_lm.convert"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / model_name.replace("-", "_").replace(".", "_")

    if model_path.exists():
        size = shutil.disk_usage(model_path).used / (1024**3)
        print(f"  Model already exists at {model_path} (~{size:.1f} GB)")
        return {"success": True, "skipped": True, "size_gb": size}

    cmd = f'mlx_lm.convert --hf-path {repo} -q --mlx-path "{model_path}"'

    start = time.time()
    result = conda_cmd(cmd, timeout=7200)
    elapsed = time.time() - start

    if result.returncode == 0:
        size = shutil.disk_usage(model_path).used / (1024**3) if model_path.exists() else 0
        print(f"  Downloaded in {elapsed:.1f}s (~{size:.1f} GB)")
        return {"success": True, "time_s": elapsed, "size_gb": size}
    else:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:500]}

def benchmark_model(model_name: str, model_path: Path, prompt: str, max_tokens: int = 300) -> dict:
    """Benchmark a single model using mlx_lm.generate"""
    print(f"\n  Benchmarking {model_name}...")

    model_path_str = str(model_path)

    python_code = f'''
import mlx_lm
import time

model_path = "{model_path_str}"
prompt = """{prompt.replace('"', '\\"')}"""

load_start = time.time()
model, tokenizer = mlx_lm.load(model_path)
load_time = time.time() - load_start

gen_start = time.time()
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt,
    verbose=False,
    max_tokens={max_tokens},
    temp=0.7,
)
gen_time = time.time() - gen_start

tokens = len(tokenizer.encode(response))
tps = tokens / gen_time if gen_time > 0 else 0

print(f"LOAD:{load_time:.2f}")
print(f"GEN:{gen_time:.2f}")
print(f"TOKENS:{tokens}")
print(f"TPS:{tps:.1f}")
print(f"RESPONSE:{response[:300]}...")
'''

    start_time = time.time()
    result = conda_cmd(f'python -c "{python_code}"', timeout=600)

    if result.returncode == 0:
        output = result.stdout
        lines = output.split("\n")
        data = {}
        for line in lines:
            if ":" in line and not line.startswith("RESPONSE"):
                key, val = line.split(":", 1)
                data[key] = val

        return {
            "success": True,
            "load_time_s": float(data.get("LOAD", 0)),
            "gen_time_s": float(data.get("GEN", 0)),
            "total_time_s": time.time() - start_time,
            "tokens": int(data.get("TOKENS", 0)),
            "tokens_per_sec": float(data.get("TPS", 0)),
            "response_preview": data.get("RESPONSE", "")[:200],
        }
    else:
        return {
            "success": False,
            "error": result.stderr[:500],
            "total_time_s": time.time() - start_time,
        }

def run_test_prompt(prompt: str, max_tokens: int = 300):
    """Test prompt for bias detection"""
    return prompt

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX Model Download and Benchmark")
    parser.add_argument("--download-only", action="store_true", help="Only download models")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for name, config in MODELS.items():
            print(f"  {name:<30} {config['repo']}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter models if specific one requested
    models_to_test = MODELS
    if args.model:
        models_to_test = {args.model: MODELS.get(args.model)}
        if not models_to_test[args.model]:
            print(f"Model {args.model} not found")
            return

    download_results = {}
    benchmark_results = {}

    # Download phase
    print("\n" + "="*70)
    print("PHASE 1: DOWNLOADING MLX MODELS")
    print("="*70)
    print(f"Target directory: {MODELS_DIR}")
    print(f"Models to download: {len(models_to_test)}")

    for name, config in models_to_test.items():
        result = download_model(name, config["repo"])
        download_results[name] = result

    if args.download_only:
        print_summary(download_results, {})
        return

    # Benchmark phase
    print("\n" + "="*70)
    print("PHASE 2: BENCHMARKING MODELS")
    print("="*70)

    bias_test_prompt = """Analyze the political bias of this news article:

"US forces conducted precision strikes against Iranian military targets in response to threats against American personnel. Iran condemned the attacks as illegal aggression. The strikes were limited and targeted, with no civilian casualties reported."

Provide a JSON response with:
- dir: direction scores {"L": confidence_left, "C": confidence_center, "R": confidence_right}
- deg: degree scores {"L": low, "M": medium, "H": high}
- reason: brief explanation of the bias detected

Output ONLY valid JSON, no additional text."""

    for name, config in models_to_test.items():
        model_path = MODELS_DIR / name.replace("-", "_").replace(".", "_")

        if download_results.get(name, {}).get("success"):
            print(f"\n--- Testing {name} ---")
            result = benchmark_model(name, model_path, bias_test_prompt)
            benchmark_results[name] = result

            if result.get("success"):
                print(f"  Load: {result['load_time_s']:.1f}s | Gen: {result['gen_time_s']:.1f}s | {result['tokens_per_sec']:.1f} tok/s")
                print(f"  Response: {result['response_preview'][:100]}...")
            else:
                print(f"  Error: {result.get('error', 'Unknown')[:100]}")
        else:
            benchmark_results[name] = {"success": False, "error": "Download failed"}

    print_summary(download_results, benchmark_results)
    save_results(download_results, benchmark_results)

def print_summary(download_results, benchmark_results):
    """Print summary table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Model':<28} {'Type':<6} {'Size':<7} {'Status':<9} {'Load':<8} {'Gen':<8} {'Tokens/s':<10}")
    print("-" * 80)

    for name, config in MODELS.items():
        bench = benchmark_results.get(name, {})
        dl = download_results.get(name, {})

        if dl.get("skipped"):
            status = "cached"
        elif dl.get("success"):
            status = "ok"
        else:
            status = "failed"

        load = f"{bench.get('load_time_s', 0):.1f}s" if bench.get("success") else "-"
        gen = f"{bench.get('gen_time_s', 0):.1f}s" if bench.get("success") else "-"
        tps = f"{bench.get('tokens_per_sec', 0):.1f}" if bench.get("success") else "-"

        print(f"{name:<28} {config['type']:<6} {config['size_gb']:<7} {status:<9} {load:<8} {gen:<8} {tps:<10}")

    print("-" * 80)

def save_results(download_results, benchmark_results):
    """Save results to JSON file"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "platform": "Apple Silicon M3 Max 128GB",
        "environment": "conda mlx2",
        "download_results": download_results,
        "benchmark_results": benchmark_results,
    }

    output_path = Path(__file__).parent.parent / "mlx_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()