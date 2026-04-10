#!/usr/bin/env python3

"""
MongoDB to LoRA Training Data Exporter
=======================================
Extracts balanced training data from MongoDB for fine-tuning local models.

Supports models:
- T5 (text-to-text)
- Llama/GLM (instruction-tuning format)
- Qwen (chat format)
- Custom JSON formats

Usage:
    python mongo2lora.py -o train.json --target-samples 10000
    python mongo2lora.py -o train.json --model-type llama --task bias_detection
"""

import json
import random
import argparse
import os
from collections import defaultdict
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pymongo


MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rssnews")
MONGO_COLL = os.getenv("MONGO_COLL", "articles")


def get_mongo_client(uri: Optional[str] = None) -> pymongo.MongoClient:
    if uri is None:
        uri = MONGO_URI
    return pymongo.MongoClient(uri)


def parse_date_arg(date_str: str) -> datetime:
    if date_str.startswith("-") and date_str[1:].isdigit():
        return datetime.now() + timedelta(days=int(date_str))
    return datetime.fromisoformat(date_str)


def load_buckets(
    collection,
    bias_field: str = "bias",
    min_dominant: float = 0.5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, List]:
    query = {bias_field: {"$exists": True, "$ne": None}}

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = parse_date_arg(start_date)
        if end_date:
            date_filter["$lte"] = parse_date_arg(end_date)
        query["published"] = date_filter

    docs = list(collection.find(query))

    buckets = defaultdict(list)

    for doc in docs:
        bias = doc.get(bias_field, {})
        if isinstance(bias, str):
            continue

        dir_vals = bias.get("dir", bias.get("direction", {}))
        deg_vals = bias.get("deg", bias.get("degree", {}))

        if not dir_vals or not deg_vals:
            continue

        dominant_dir = max(dir_vals, key=dir_vals.get)
        dominant_deg = max(deg_vals, key=deg_vals.get)

        if (
            dir_vals.get(dominant_dir, 0) >= min_dominant
            and deg_vals.get(dominant_deg, 0) >= min_dominant
        ):
            combo = f"{dominant_dir}-{dominant_deg}"
            buckets[combo].append(
                {
                    "text": doc.get("article", "") or doc.get("title", ""),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "published": str(doc.get("published", "")),
                    "bias": bias,
                }
            )

    return buckets


def allocate_samples(
    buckets: Dict[str, List], min_samples: int, target_samples: Optional[int] = None
) -> Dict[str, int]:
    bins = sorted(buckets.keys())
    counts = {}

    for combo in bins:
        counts[combo] = min(len(buckets[combo]), min_samples)

    allocated = sum(counts.values())

    if target_samples and allocated < target_samples:
        remaining = target_samples - allocated
        for combo in bins:
            if len(buckets[combo]) > min_samples:
                extra = min(remaining, len(buckets[combo]) - min_samples)
                counts[combo] += extra
                remaining -= extra
                if remaining <= 0:
                    break

    return counts


def format_for_model(data: List[Dict], model_type: str, task: str) -> List[Dict]:
    formatted = []

    for item in data:
        text = item["text"]
        bias = item["bias"]

        dir_vals = bias.get(
            "dir", bias.get("direction", {"L": 0.33, "C": 0.34, "R": 0.33})
        )
        deg_vals = bias.get(
            "deg", bias.get("degree", {"L": 0.33, "M": 0.34, "H": 0.33})
        )

        output_json = json.dumps({"direction": dir_vals, "degree": deg_vals})

        if model_type == "t5":
            formatted.append(
                {
                    "input": f"classify political bias as json: {text[:1000]}",
                    "output": output_json,
                }
            )

        elif model_type in ["llama", "qwen", "glm"]:
            if model_type == "llama":
                formatted.append(
                    {
                        "instruction": "Analyze the political bias of this article. Return JSON with direction (L/C/R probabilities) and degree (L/M/H probabilities).",
                        "input": text[:2000],
                        "output": output_json,
                    }
                )
            elif model_type == "qwen":
                formatted.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a political bias analysis assistant.",
                            },
                            {
                                "role": "user",
                                "content": f"Analyze the political bias:\n\n{text[:2000]}",
                            },
                            {"role": "assistant", "content": output_json},
                        ]
                    }
                )
            else:  # glm
                formatted.append(
                    {
                        "prompt": f"Analyze political bias: {text[:2000]}",
                        "response": output_json,
                    }
                )

        elif model_type == "raw":
            formatted.append({"text": text, "bias": bias})

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Export balanced training data for LoRA"
    )
    parser.add_argument("-o", "--output", default="train.json", help="Output file")
    parser.add_argument("--collection", default=MONGO_COLL, help="MongoDB collection")
    parser.add_argument("--bias-field", default="bias", help="Bias field name")
    parser.add_argument(
        "--min-samples", type=int, default=100, help="Min samples per bin"
    )
    parser.add_argument("--target-samples", type=int, help="Target total samples")
    parser.add_argument(
        "--min-dominant", type=float, default=0.5, help="Min dominant threshold"
    )
    parser.add_argument("--start-date", help="Start date (-7 or ISO)")
    parser.add_argument("--end-date", help="End date (-1 or ISO)")
    parser.add_argument(
        "--model-type",
        default="llama",
        choices=["t5", "llama", "qwen", "glm", "raw"],
        help="Output format",
    )
    parser.add_argument("--task", default="bias_detection", help="Task name")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    client = get_mongo_client()
    db = client[MONGO_DB]
    coll = db[args.collection]

    print(f"Loading bias buckets from {MONGO_DB}.{args.collection}...")
    buckets = load_buckets(
        coll,
        bias_field=args.bias_field,
        min_dominant=args.min_dominant,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if not buckets:
        print("No biased articles found!")
        return

    print(f"Found {len(buckets)} bias combinations:")
    for combo, docs in sorted(buckets.items()):
        print(f"  {combo}: {len(docs)} articles")

    counts = allocate_samples(buckets, args.min_samples, args.target_samples)

    print(f"\nAllocating samples: {sum(counts.values())} total")

    data = []
    for combo, count in counts.items():
        selected = random.sample(buckets[combo], min(count, len(buckets[combo])))
        data.extend(selected)

    if args.shuffle:
        random.shuffle(data)

    formatted = format_for_model(data, args.model_type, args.task)

    with open(args.output, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"\nWrote {len(formatted)} samples to {args.output}")

    for combo, count in sorted(counts.items()):
        print(f"  {combo}: {count}")


if __name__ == "__main__":
    main()
