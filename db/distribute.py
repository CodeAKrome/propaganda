#!/usr/bin/env python

"""
Distributed Inference Dispatcher
Distributes inference tasks to multiple worker nodes (e.g., Macs running a server)
"""

import asyncio
import aiohttp
import time
import json
import sys

# --- CONFIGURATION ---
WORKER_NODES = [
    "http://127.0.0.1:11434",
    "http://chico.local:11434",
    "http://harpo.local:11434",
]

# ---------------------


def log(msg):
    """Helper to print logs to STDERR so they don't corrupt the STDOUT data stream."""
    print(msg, file=sys.stderr)


async def run_inference(session, worker_url, input_id, model_name, prompt_text):
    url = f"{worker_url}/api/generate"
    payload = {"model": model_name, "prompt": prompt_text, "stream": False}

    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "id": input_id,
                    "model": model_name,
                    "prompt": prompt_text,
                    "response": result.get("response", ""),
                    "worker": worker_url,
                }
            else:
                error_msg = await response.text()
                log(f"⚠️ Error {response.status} from {worker_url}: {error_msg}")
                return None
    except Exception as e:
        log(f"❌ Connection error on {worker_url}: {e}")
        return None


async def worker_loop(worker_name, worker_url, queue):
    async with aiohttp.ClientSession() as session:
        while not queue.empty():
            task = await queue.get()
            input_id, model_name, prompt_text = task

            start_time = time.time()
            result = await run_inference(
                session, worker_url, input_id, model_name, prompt_text
            )

            if result:
                duration = time.time() - start_time
                log(f"✅ ID: {input_id} finished by {worker_name} in {duration:.2f}s")

                # --- OUTPUT TO STDOUT ---
                # We use json.dumps ensures it is valid JSON.
                # flush=True ensures it pipes immediately.
                print(json.dumps(result), file=sys.stdout, flush=True)
            else:
                log(f"Re-queueing ID: {input_id} due to failure...")
                await queue.put(task)

            queue.task_done()


async def main():
    queue = asyncio.Queue()

    # Check if data is being piped in
    if sys.stdin.isatty():
        log("Waiting for input via stdin (paste data + Ctrl-D)...")

    input_lines = sys.stdin.readlines()

    if not input_lines:
        log("No input provided.")
        return

    log(f"Loading {len(input_lines)} items into queue...")

    valid_count = 0
    for line in input_lines:
        line = line.strip()
        if not line:
            continue

        # Split: ID [TAB] MODEL [TAB] PROMPT
        parts = line.split("\t", 2)

        if len(parts) == 3:
            queue.put_nowait((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            valid_count += 1
        else:
            log(f"Skipping malformed line: {line[:30]}...")

    if valid_count == 0:
        return

    start_global = time.time()

    tasks = []
    for i, node_url in enumerate(WORKER_NODES):
        task_name = f"Mac-{i+1}"
        tasks.append(asyncio.create_task(worker_loop(task_name, node_url, queue)))

    await queue.join()

    for task in tasks:
        task.cancel()

    total_time = time.time() - start_global
    log(f"\n🎉 Batch complete in {total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
