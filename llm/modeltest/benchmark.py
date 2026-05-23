#!/usr/bin/env python3
"""
MLX Model Download and Benchmark Script
Downloads MLX models and tests them against bias detection task.
Uses mlx2 conda environment.
Stores results in llm/modeltest/
"""

import os
import sys
import time
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR
MODELS_DIR = SCRIPT_DIR / "models"

CONDA_SHELL = "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV = "mlx2"

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
        "repo": "mlx-community/Qwen3.6-27B-OptiQ-4bit",
        "size_gb": 16.5,
        "type": "Dense",
        "params": "27B",
    },
    "Gemma-4-31B-OptiQ-4bit": {
        "repo": "mlx-community/gemma-4-31B-it-OptiQ-4bit",
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

BIAS_TEST_PROMPT = """Analyze the political bias of this news article:

"US forces conducted precision strikes against Iranian military targets in response to threats against American personnel. Iran condemned the attacks as illegal aggression. The strikes were limited and targeted, with no civilian casualties reported."

Provide a JSON response with:
- dir: direction scores {"L": confidence_left, "C": confidence_center, "R": confidence_right}
- deg: degree scores {"L": low, "M": medium, "H": high}
- reason: brief explanation of the bias detected

Output ONLY valid JSON, no additional text."""

SUMMARY_TEST_PROMPT = """Summarize this article in 3 sentences:

The Iranian government announced today that it will resume nuclear negotiations with Western powers following months of tensions. Officials from the United States, France, Germany, and Britain expressed cautious optimism about the talks. The discussions will focus on limiting Iran's uranium enrichment capacity in exchange for lifted sanctions."""

CATEGORIZATION_TEST = """Categorize this news headline into one of: POLITICS, ECONOMY, SECURITY, DIPLOMACY, HUMANITARIAN

Headline: "US and China reach preliminary trade agreement after months of negotiations"

Output ONLY the category name."""

def conda_cmd_script(script_path: Path, timeout=600):
    """Run a Python script file in mlx2 conda environment"""
    script_path_str = str(script_path)
    full_cmd = f'source {CONDA_SHELL} && conda activate {CONDA_ENV} && python {script_path_str}'
    result = subprocess.run(
        ["zsh", "-c", full_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(SCRIPT_DIR)
    )
    return result

def get_model_size(model_path: Path) -> float:
    """Get total size of model directory in GB"""
    if not model_path.exists():
        return 0
    total = 0
    for f in model_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)

def download_model(model_name: str, repo: str) -> dict:
    """Download a model using mlx_lm.convert"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / model_name.replace("-", "_").replace(".", "_")

    if model_path.exists():
        size = get_model_size(model_path)
        print(f"  Model already exists (~{size:.1f} GB)")
        return {"success": True, "skipped": True, "size_gb": size}

    cmd = f'mlx_lm.convert --hf-path {repo} -q --mlx-path "{model_path}"'
    full_cmd = f'source {CONDA_SHELL} && conda activate {CONDA_ENV} && {cmd}'

    start = time.time()
    result = subprocess.run(
        ["zsh", "-c", full_cmd],
        capture_output=True,
        text=True,
        timeout=7200,
        cwd=str(SCRIPT_DIR)
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        size = get_model_size(model_path) if model_path.exists() else 0
        print(f"  Downloaded in {elapsed:.1f}s (~{size:.1f} GB)")
        return {"success": True, "time_s": elapsed, "size_gb": size}
    else:
        print(f"  Error: {result.stderr[:500]}")
        return {"success": False, "error": result.stderr[:500], "time_s": elapsed}

def create_benchmark_script(model_path: str, prompt: str, max_tokens: int, output_file: str):
    """Create a Python script to benchmark a model"""
    # Escape the prompt for Python string
    prompt_escaped = prompt.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    script = f'''#!/usr/bin/env python3
import mlx_lm
import time

model_path = "{model_path}"
prompt = """{prompt_escaped}"""

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
)
gen_time = time.time() - gen_start

tokens = len(tokenizer.encode(response))
tps = tokens / gen_time if gen_time > 0 else 0

with open("{output_file}", "w") as f:
    f.write(f"LOAD:{{load_time:.2f}}\\n")
    f.write(f"GEN:{{gen_time:.2f}}\\n")
    f.write(f"TOKENS:{{tokens}}\\n")
    f.write(f"TPS:{{tps:.1f}}\\n")
    f.write(f"RESPONSE:{{response}}\\n")
'''

    script_path = SCRIPT_DIR / f"_bench_{Path(output_file).stem}.py"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path

def benchmark_model(model_name: str, model_path: Path, prompt: str, max_tokens: int = 300) -> dict:
    """Benchmark a single model"""
    model_path_str = str(model_path)
    output_file = SCRIPT_DIR / f"result_{model_name.replace('-', '_')}.txt"

    script_path = create_benchmark_script(model_path_str, prompt, max_tokens, str(output_file))

    start_time = time.time()
    result = conda_cmd_script(script_path, timeout=600)

    # Clean up script
    script_path.unlink(missing_ok=True)

    if result.returncode == 0 and output_file.exists():
        with open(output_file) as f:
            lines = f.read().split("\n")

        data = {}
        for line in lines:
            if line.startswith("LOAD:"):
                data["load_time"] = float(line.split(":")[1])
            elif line.startswith("GEN:"):
                data["gen_time"] = float(line.split(":")[1])
            elif line.startswith("TOKENS:"):
                data["tokens"] = int(line.split(":")[1])
            elif line.startswith("TPS:"):
                data["tps"] = float(line.split(":")[1])
            elif line.startswith("RESPONSE:"):
                data["response"] = line[9:]

        output_file.unlink(missing_ok=True)

        return {
            "success": True,
            "load_time_s": data.get("load_time", 0),
            "gen_time_s": data.get("gen_time", 0),
            "total_time_s": time.time() - start_time,
            "tokens": data.get("tokens", 0),
            "tokens_per_sec": data.get("tps", 0),
            "response": data.get("response", ""),
        }
    else:
        output_file.unlink(missing_ok=True)
        return {
            "success": False,
            "error": result.stderr[:500] if result.stderr else result.stdout[:500],
            "total_time_s": time.time() - start_time,
        }

def run_all_tests(model_name: str, model_path: Path) -> dict:
    """Run all benchmark tests on a model"""
    results = {
        "bias_detection": benchmark_model(model_name, model_path, BIAS_TEST_PROMPT, max_tokens=400),
        "summarization": benchmark_model(model_name, model_path, SUMMARY_TEST_PROMPT, max_tokens=200),
        "categorization": benchmark_model(model_name, model_path, CATEGORIZATION_TEST, max_tokens=50),
    }
    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX Model Benchmark")
    parser.add_argument("--download-only", action="store_true", help="Only download models")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--quick", action="store_true", help="Quick test single prompt")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for name, config in MODELS.items():
            print(f"  {name:<30} {config['repo']}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_test = MODELS
    if args.model:
        models_to_test = {args.model: MODELS.get(args.model)}
        if not models_to_test[args.model]:
            print(f"Model {args.model} not found")
            return

    download_results = {}
    benchmark_results = {}

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

    print("\n" + "="*70)
    print("PHASE 2: BENCHMARKING MODELS")
    print("="*70)

    for name, config in models_to_test.items():
        model_path = MODELS_DIR / name.replace("-", "_").replace(".", "_")

        if download_results.get(name, {}).get("success"):
            print(f"\n{'='*50}")
            print(f"Testing: {name}")
            print(f"{'='*50}")

            results = run_all_tests(name, model_path)
            benchmark_results[name] = results

            for task, result in results.items():
                if result.get("success"):
                    print(f"  {task:<20}: {result['gen_time_s']:.1f}s | {result['tokens_per_sec']:.1f} tok/s")
                    print(f"    Response: {result['response'][:80]}...")
                else:
                    print(f"  {task:<20}: FAILED - {result.get('error', 'Unknown')[:80]}")
        else:
            benchmark_results[name] = {"error": "Download failed"}

    print_summary(download_results, benchmark_results)
    save_results(download_results, benchmark_results)

def print_summary(download_results, benchmark_results):
    """Print summary table"""
    print("\n" + "="*90)
    print("MODEL COMPARISON SUMMARY")
    print("="*90)

    header = f"{'Model':<28} {'Type':<6} {'GB':<5} {'Status':<8} {'Bias':<12} {'Summ':<12} {'Cat':<12} {'Avg TPS':<10}"
    print(f"\n{header}")
    print("-" * 90)

    for name, config in MODELS.items():
        bench = benchmark_results.get(name, {})
        dl = download_results.get(name, {})

        if dl.get("skipped"):
            status = "cached"
        elif dl.get("success"):
            status = "ready"
        else:
            status = "failed"

        bias_result = bench.get('bias_detection', {}) if isinstance(bench, dict) else {}
        summ_result = bench.get('summarization', {}) if isinstance(bench, dict) else {}
        cat_result = bench.get('categorization', {}) if isinstance(bench, dict) else {}

        bias_time = f"{bias_result.get('gen_time_s', 0):.1f}s" if bias_result.get('success') else "-"
        summ_time = f"{summ_result.get('gen_time_s', 0):.1f}s" if summ_result.get('success') else "-"
        cat_time = f"{cat_result.get('gen_time_s', 0):.1f}s" if cat_result.get('success') else "-"

        all_tps = []
        for r in [bias_result, summ_result, cat_result]:
            if isinstance(r, dict) and r.get('success'):
                all_tps.append(r.get('tokens_per_sec', 0))

        avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0

        print(f"{name:<28} {config['type']:<6} {config['size_gb']:<5} {status:<8} {bias_time:<12} {summ_time:<12} {cat_time:<12} {avg_tps:<10.1f}")

    print("-" * 90)

def save_results(download_results, benchmark_results):
    """Save results to JSON file"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "platform": "Apple Silicon M3 Max 128GB",
        "environment": "conda mlx2",
        "models_directory": str(MODELS_DIR),
        "download_results": download_results,
        "benchmark_results": benchmark_results,
    }

    output_path = RESULTS_DIR / "mlx_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()