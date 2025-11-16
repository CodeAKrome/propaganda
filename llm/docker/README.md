Option 1: Using Docker directly
# Build the image
docker build -t ollama-fastapi .

# Run with defaults (connects to Ollama on host machine)
docker run -p 8101:8101 ollama-fastapi

# Run with custom port
docker run -p 8080:8080 ollama-fastapi --port 8080

# Run with custom model
docker run -p 8101:8101 ollama-fastapi --model llama2:7b

# Run with custom Ollama URL (if Ollama is in another container)
docker run -p 8101:8101 ollama-fastapi --ollama-url http://ollama:11434

---

Option 2: Using Docker Compose

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

Test it:

curl -X POST "http://localhost:8101/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain Docker in one sentence"}'

