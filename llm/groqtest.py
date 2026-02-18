#!/usr/bin/env python
import os

from groq import Groq

MODEL="llama-3.3-70b-versatile"
MODEL="openai/gpt-oss-120b"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model=MODEL,
)

print(chat_completion.choices[0].message.content)
