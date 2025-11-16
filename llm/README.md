Features:

Command Line Arguments:

--port: Port to run on (default: 8101)
--model: Default Ollama model (default: gpt-oss:20b)
--ollama-url: Ollama API URL (default: http://localhost:11434)


Endpoints:

POST /generate - Send a prompt and get a completion
GET /health - Check service and Ollama connection status
GET /models - List available Ollama models
GET / - Service info and endpoint list


Request Parameters:

prompt (required) - Your text prompt
model (optional) - Override default model
temperature (optional) - Control randomness
max_tokens (optional) - Limit response length

# Install dependencies first
pip install fastapi uvicorn httpx pydantic

# Run with defaults
python ollama_service.py

# Run on custom port
python ollama_service.py --port 8080

# Use different model
python ollama_service.py --model llama2:7b

# Example request
curl -X POST "http://localhost:8101/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?"}'
  