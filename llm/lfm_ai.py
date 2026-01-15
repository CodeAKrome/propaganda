#!/usr/bin/env python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-bf16")

prompt = "What is the capital of France?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

sampler = make_sampler(temp=0.1, top_k=50, top_p=0.1)
logits_processors = make_logits_processors(repetition_penalty=1.05)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    sampler=sampler,
    logits_processors=logits_processors,
    verbose=True,
)

print(response)
