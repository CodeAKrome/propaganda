import ollama
from json import loads
import sys
import base64
from PIL import Image
from io import BytesIO
import requests
import os

# Example usage:
# python ollamaai.py "Capital of Spain"
# python ollamaai.py prompt.txt
# python ollamaai.py - < prompt.txt
# python ollamaai.py "Capital of Spain" --model llama3.1:8b --temperature 0.5


class OllamaAI:
    def __init__(
        self, system_prompt=None, model=None, max_tokens=3000, temperature=0.1
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.set_system(system_prompt)

    def set_system(self, prompt):
        self.system = (
            prompt
            or "You are a helpful assistant. You reply with short, accurate answers."
        )

    def load_image(self, PathUrlBase64):
        """
        Returns an image object from a path, URL or base64 encoded image data.
        """
        if PathUrlBase64.startswith(("http://", "https://")):
            response = requests.get(PathUrlBase64)
            img = Image.open(BytesIO(response.content))
        elif PathUrlBase64.startswith("data:image"):
            img_data = PathUrlBase64.split(",")[1]
            img = Image.open(BytesIO(base64.b64decode(img_data)))
        else:
            img = Image.open(PathUrlBase64)
        return img

    def says(self, prompt, image_path_or_url=None):
        message = {"role": "user", "content": prompt}

        if image_path_or_url:
            # Load and encode the image
            image = self.load_image(image_path_or_url)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            # Add image to the message
            message["images"] = [img_str]

        try:
            response = ollama.chat(
                model=self.model,
                messages=[message],
                stream=False,
                options={
                    "temperature": self.temperature,
                    "system": self.system,
                    "num_predict": self.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            sys.stderr.write(f"Error generating content: {e}\n")
            return None


def load_from_file(file_path):
    """Load content from a file."""
    with open(file_path, "r") as file:
        return file.read().strip()


def get_prompt(prompt_input):
    """
    Get prompt from various sources:
    - If "-", read from stdin
    - If it's a file that exists, read from file
    - Otherwise, treat as the prompt string itself
    """
    if prompt_input == "-":
        return sys.stdin.read().strip()
    elif os.path.isfile(prompt_input):
        return load_from_file(prompt_input)
    else:
        return prompt_input


def get_system_prompt(system_input):
    """
    Get system prompt from various sources:
    - If it's a file that exists, read from file
    - Otherwise, treat as the system prompt string itself
    """
    if system_input and os.path.isfile(system_input):
        return load_from_file(system_input)
    else:
        return system_input


def main(
    prompt_input,
    model="qwen3:14b",
    system_prompt=None,
    tokens=128000,
    temperature=0.3,
    image=None,
):
    """
    Interact with OllamaAI.

    Args:
        prompt_input (str, required): Prompt string, filename, or "-" for stdin
        model (str, optional): Model to use. Default: qwen3:14b
        system_prompt (str, optional): System prompt string or filename
        tokens (int, optional): Max tokens to generate. Default: 10
        temperature (float, optional): Temperature setting. Default: 0.3
        image (str, optional): Path or URL to an image
    """
    # Get the prompt
    prompt = get_prompt(prompt_input)

    if not prompt:
        raise ValueError("Prompt cannot be empty")

    # Get the system prompt if provided
    system = get_system_prompt(system_prompt)

    # Create OllamaAI instance
    ollama_ai = OllamaAI(
        system_prompt=system, model=model, max_tokens=tokens, temperature=temperature
    )

    # Generate response
    if image:
        response = ollama_ai.says(prompt, image)
    else:
        response = ollama_ai.says(prompt)

    if response:
        print(response)
    else:
        sys.stderr.write("No response generated\n")
        sys.exit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
