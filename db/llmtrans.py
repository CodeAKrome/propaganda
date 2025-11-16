#!/usr/bin/env python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"


def grafix_device():
    """Return device to use for GPU"""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def mps_built():
    """Is mps support available"""
    if torch.backends.mps.is_built():
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    else:
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )


thisdevice = grafix_device()
print(f"Using {thisdevice} device")


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to the selected device
model = model.to(thisdevice)

# Configurable parameters
temperature = 0.7  # Controls randomness (0.0 = deterministic, higher = more random)
max_tokens = 100000  # Maximum number of tokens to generate

messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(thisdevice)

outputs = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=temperature,
    do_sample=True,  # Required for temperature to have effect
)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
