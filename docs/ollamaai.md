# ollamaai.py

Ollama client for text and vision models with fire CLI.

## Overview

Provides a command-line interface to Ollama for generating text or analyzing images. Supports prompt from argument, file, or stdin. Merged from `ai_ollama.py`.

## Usage

```bash
python db/ollamaai.py [prompt] [options]
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `prompt` | Prompt string, filename, or `-` for stdin | Required |
| `model` | Ollama model to use | `qwen3:14b` |
| `system_prompt` | System prompt | "You are a helpful assistant..." |
| `tokens` | Max tokens to generate | `128000` |
| `temperature` | Temperature setting | `0.3` |
| `image` | Path/URL to image for vision models | None |
| `prompt_file` | Path to file with prompt | None |
| `system_prompt_file` | Path to file with system prompt | None |

## Examples

### Basic Prompt

```bash
python db/ollamaai.py "What is the capital of France?"
```

### Custom Model

```bash
python db/ollamaai.py "Explain quantum computing" --model llama3.1:8b
```

### Read from File

```bash
python db/ollamaai.py prompt.txt
```

### Read from Stdin

```bash
echo "Summarize this" | python db/ollamaai.py -
```

### Custom System Prompt

```bash
python db/ollamaai.py "Explain like I'm 5" --system_prompt "You are a friendly teacher"
```

### System Prompt from File

```bash
python db/ollamaai.py "Explain this code" --system_prompt_file system.txt
```

### Vision (Image Analysis)

```bash
python db/ollamaai.py "Describe what's in this image" --image photo.jpg
```

### Vision from URL

```bash
python db/ollamaai.py "What does this chart show?" --image https://example.com/chart.png
```

### Low Temperature (more deterministic)

```bash
python db/ollamaai.py "Calculate 2+2" --temperature 0.1
```

### High Temperature (more creative)

```bash
python db/ollamaai.py "Write a story" --temperature 0.9
```

### Fewer Tokens

```bash
python db/ollamaai.py "Quick summary" --tokens 100
```

### Combine Options

```bash
python db/ollamaai.py prompt.txt \
  --model llava:13b \
  --system_prompt "You are a document analyzer" \
  --temperature 0.2 \
  --tokens 500
```

## Alternative: Command Line

You can also call it as a module:

```bash
python -m db.ollamaai "Your prompt" --model llama3.1:8b
```

## Using with Pipes

### Output to File

```bash
python db/ollamaai.py "Write a poem" > poem.txt
```

### Chain with Other Tools

```bash
python db/ollamaai.py "List 5 key points" < article.txt | tee summary.txt
```

## Python API

```python
from db.ollamaai import OllamaAI

# Initialize
ai = OllamaAI(
    model="llama3.1:8b",
    system_prompt="You are a helpful assistant.",
    max_tokens=1000,
    temperature=0.3
)

# Generate text
response = ai.says("What is machine learning?")
print(response)

# Generate with image
response = ai.says("Describe this image", image_path_or_url="photo.jpg")
print(response)
```

## Image Loading

Supports three image input formats:
- **Path**: Local file path (`photo.jpg`)
- **URL**: Web URL (`https://example.com/image.png`)
- **Base64**: Data URL (`data:image/png;base64,...`)

## Environment Variables

```bash
OLLAMA_HOST=localhost:11434  # Default Ollama host
```

## Available Models

Common Ollama models:
- `llama3.1:8b`, `llama3.1:70b`
- `qwen3:14b`, `qwen3:32b`
- `llava:7b`, `llava:13b` (vision)
- `mistral`, `mixtral`
- `gemma:7b`
- `phi3`, `phi3.5`

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No response generated |