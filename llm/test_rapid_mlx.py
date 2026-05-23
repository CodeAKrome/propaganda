#!/usr/bin/env python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")  # any value works, no real key needed

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)
