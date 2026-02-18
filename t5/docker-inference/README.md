# Bias Detector Docker Inference

Dockerized FastAPI service for political bias detection using T5 with LoRA adapters.

## Structure

```
docker-inference/
├── app_cpu.py           # FastAPI app for CPU/CUDA (old Mac x86)
├── app_mps.py           # FastAPI app for MPS (Apple Silicon)
├── Dockerfile.cpu       # Docker image for CPU/CUDA version
├── Dockerfile.mps       # Docker image for MPS version
├── requirements_cpu.txt # Pinned dependencies (old Mac x86)
├── requirements_mps.txt # Standard dependencies
├── docker-compose.yml   # Compose file with profiles
└── README.md

../server_mps.py         # Native MPS server (run directly on macOS)
```

## Prerequisites

- Docker and Docker Compose installed (for Docker versions)
- Model weights in `t5/bias-detector-output/` directory
- (Optional) NVIDIA Container Toolkit for GPU support in Docker
- (Native MPS) macOS with Apple Silicon and PyTorch with MPS support

## Quick Start

### Native MPS Server (Recommended for Apple Silicon)

For MPS GPU acceleration on Apple Silicon, run the native server directly:

```bash
cd t5
pip install -r requirements.txt
python server_mps.py
```

Or with uvicorn:
```bash
cd t5
uvicorn server_mps:app --host 0.0.0.0 --port 8000
```

### Docker Versions

### Build and Run CPU/CUDA Version (Old Mac x86)

```bash
# Build the image
docker-compose build bias-detector-cpu

# Run with CPU only
docker-compose --profile cpu up bias-detector-cpu

# Run with GPU support (requires NVIDIA Container Toolkit)
docker-compose --profile cpu up bias-detector-cpu
```

### Build and Run MPS Version (Standard)

```bash
# Build the image
docker-compose build bias-detector-mps

# Run the container
docker-compose --profile mps up bias-detector-mps
```

## Alternative: Direct Docker Commands

### CPU/CUDA Version

```bash
# Build
docker build -f Dockerfile.cpu -t bias-detector:cpu .

# Run (CPU only)
docker run -p 8000:8000 -v ../bias-detector-output:/app/bias-detector-output:ro bias-detector:cpu

# Run with GPU
docker run --gpus all -p 8000:8000 -v ../bias-detector-output:/app/bias-detector-output:ro bias-detector:cpu
```

### MPS Version

```bash
# Build
docker build -f Dockerfile.mps -t bias-detector:mps .

# Run
docker run -p 8000:8000 -v ../bias-detector-output:/app/bias-detector-output:ro bias-detector:mps
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Bias

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The government announced new policies today."}'
```

Response:
```json
{
  "result": {
    "bias": "center",
    "confidence": 0.85
  },
  "device": "cuda"
}
```

### Raw Prediction

```bash
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d '{"text": "The government announced new policies today."}'
```

### Using Text Files

```bash
# Read text from file and send to API
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$(cat prompt/left.txt | tr '\n' ' ' | tr -d '\"')\"}"

# Alternative: Using jq for proper JSON escaping
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{text: .}' < prompt/left.txt)"

# Using a temporary JSON file
cat prompt/left.txt | jq -Rs '{text: .}' > /tmp/payload.json
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d @/tmp/payload.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./bias-detector-output` | Path to LoRA adapter weights |
| `BASE_MODEL_NAME` | `t5-large` | Base T5 model name |

## Volume Mounts

The model weights directory must be mounted at `/app/bias-detector-output`:

```bash
-v /path/to/bias-detector-output:/app/bias-detector-output:ro
```

## GPU Support

For CUDA GPU support, ensure:

1. NVIDIA drivers installed on host
2. NVIDIA Container Toolkit installed
3. Add `--gpus all` flag to docker run

## Notes

- **CPU Version**: Uses pinned package versions for compatibility with older hardware (old Mac x86)
- **MPS Version**: Uses latest compatible versions. Note: MPS (Apple Silicon GPU) is **not available inside Docker containers** - the app will automatically fallback to CPU. For native MPS support, run the Python script directly on macOS.
- Both versions expose the same API interface
- Model weights are loaded lazily on first request to reduce memory usage
