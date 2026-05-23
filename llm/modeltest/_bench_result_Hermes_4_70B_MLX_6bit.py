#!/usr/bin/env python3
import mlx_lm
import time

model_path = "/Users/kyle/hub/propaganda/llm/modeltest/models/Hermes_4_70B_MLX_6bit"
prompt = """Summarize this article in 3 sentences:\n\nThe Iranian government announced today that it will resume nuclear negotiations with Western powers following months of tensions. Officials from the United States, France, Germany, and Britain expressed cautious optimism about the talks. The discussions will focus on limiting Iran's uranium enrichment capacity in exchange for lifted sanctions."""

load_start = time.time()
model, tokenizer = mlx_lm.load(model_path)
load_time = time.time() - load_start

gen_start = time.time()
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt,
    verbose=False,
    max_tokens=200,
)
gen_time = time.time() - gen_start

tokens = len(tokenizer.encode(response))
tps = tokens / gen_time if gen_time > 0 else 0

with open("/Users/kyle/hub/propaganda/llm/modeltest/result_Hermes_4_70B_MLX_6bit.txt", "w") as f:
    f.write(f"LOAD:{load_time:.2f}\n")
    f.write(f"GEN:{gen_time:.2f}\n")
    f.write(f"TOKENS:{tokens}\n")
    f.write(f"TPS:{tps:.1f}\n")
    f.write(f"RESPONSE:{response}\n")
