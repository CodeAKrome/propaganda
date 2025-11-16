#!/usr/bin/env python3
"""
LM Studio API Client for Text Inference
Connects to LM Studio's local API server for text generation.
"""

import argparse
import sys
import os
import base64
import json
from pathlib import Path
import requests


def read_prompt(prompt_arg):
    """Read prompt from stdin, file, or treat as direct string."""
    if prompt_arg == "-":
        # Read from stdin
        return sys.stdin.read().strip()
    elif os.path.isfile(prompt_arg):
        # Read from file
        with open(prompt_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        # Treat as direct prompt string
        return prompt_arg


def read_system_prompt(system_arg):
    """Read system prompt from file or treat as direct string."""
    if system_arg is None:
        return None
    elif os.path.isfile(system_arg):
        # Read from file
        with open(system_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        # Treat as direct system prompt string
        return system_arg


def encode_image(image_path):
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path):
    """Determine MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(ext, "image/jpeg")


def send_inference_request(
    prompt,
    system_prompt=None,
    top_n=10,
    temperature=0.3,
    image_path=None,
    api_url="http://localhost:1234/v1/chat/completions",
):
    """Send inference request to LM Studio API."""

    # Build messages array
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message content
    if image_path:
        # If image is provided, use multimodal format
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_data = encode_image(image_path)
        mime_type = get_image_mime_type(image_path)

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ],
            }
        )
    else:
        # Text-only message
        messages.append({"role": "user", "content": prompt})

    # Prepare request payload
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": top_n if top_n > 0 else -1,  # LM Studio uses max_tokens
        "stream": False,
    }

    # Send request
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to LM Studio API.", file=sys.stderr)
        print(
            "Make sure LM Studio is running and the API server is started.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Connect to LM Studio for text inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is the capital of France?"
  %(prog)s prompt.txt --system "You are a helpful assistant"
  %(prog)s - --temperature 0.7 --top_n 100
  %(prog)s "Describe this image" --image photo.jpg
  echo "Tell me a joke" | %(prog)s -
        """,
    )

    parser.add_argument(
        "prompt", help='Prompt filename, "-" for stdin, or direct prompt string'
    )

    parser.add_argument(
        "--system",
        help="System prompt filename or direct system prompt string",
        default=None,
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Maximum number of tokens to generate (default: 10, use -1 for unlimited)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for sampling (default: 0.3)",
    )

    parser.add_argument(
        "--image", help="Path to image file for multimodal inference", default=None
    )

    parser.add_argument(
        "--api-url",
        default="http://localhost:1234/v1/chat/completions",
        help="LM Studio API URL (default: http://localhost:1234/v1/chat/completions)",
    )

    args = parser.parse_args()

    # Read prompt
    try:
        prompt = read_prompt(args.prompt)
    except Exception as e:
        print(f"Error reading prompt: {e}", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        print("Error: Empty prompt provided", file=sys.stderr)
        sys.exit(1)

    # Read system prompt if provided
    system_prompt = read_system_prompt(args.system)

    # Send inference request
    result = send_inference_request(
        prompt=prompt,
        system_prompt=system_prompt,
        top_n=args.top_n,
        temperature=args.temperature,
        image_path=args.image,
        api_url=args.api_url,
    )

    # Extract and print the response
    try:
        response_text = result["choices"][0]["message"]["content"]
        print(response_text)
    except (KeyError, IndexError) as e:
        print(f"Error parsing response: {e}", file=sys.stderr)
        print(f"Raw response: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
