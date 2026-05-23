#!/usr/bin/env python3

import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True


headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "mistralai/mistral-medium-3.5-128b",
  "reasoning_effort": "high",
  "messages": [{"role":"user","content":"Hello."}],
  "max_tokens": 16384,
  "temperature": 0.70,
  "top_p": 1.00,
  "stream": stream
}



response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())

