#!/usr/bin/env python
import fire
from openai import OpenAI


def chat(prompt=None, file=None, model="default", base_url="http://localhost:8001/v1"):
    """
    Send a chat completion to a local OpenAI-compatible endpoint.

    Args:
        prompt: The prompt text to send directly.
        file: Path to a text file containing the prompt.
        model: Model name to use (default: "default").
        base_url: API base URL (default: http://localhost:8001/v1).
    """
    if file:
        with open(file, "r") as f:
            content = f.read()
    elif prompt:
        content = prompt
    else:
        raise ValueError("Provide either --prompt or --file")

    client = OpenAI(base_url=base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    fire.Fire(chat)
    
