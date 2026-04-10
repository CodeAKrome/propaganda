#!/usr/bin/env python3
"""
NER Processor - queries MongoDB for articles and calls NER service to extract entities.
"""

import os
import sys
import argparse
import requests
from datetime import datetime, timedelta
import pymongo

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
NER_URL = os.getenv("NER_URL", "http://localhost:8100/extract")


def parse_date_arg(date_str: str) -> datetime:
    """Parse date argument (negative days or ISO format)."""
    if date_str.startswith("-") and date_str[1:].isdigit():
        days_ago = int(date_str)
        return datetime.now() + timedelta(days=days_ago)
    else:
        return datetime.fromisoformat(date_str)


def process_articles_since(date_str: str, limit: int = None):
    """Fetch articles from MongoDB and process them with NER service."""
    client = pymongo.MongoClient(MONGO_URI)
    db = client["rssnews"]
    coll = db["articles"]

    start_date = parse_date_arg(date_str)
    print(f"Processing articles since {start_date.isoformat()}")

    query = {"published": {"$gte": start_date}}
    if limit:
        articles = list(coll.find(query).limit(limit))
    else:
        articles = list(coll.find(query))

    print(f"Found {len(articles)} articles")

    processed = 0
    errors = 0

    for article in articles:
        article_id = article["_id"]
        text = article.get("article", "") or article.get("title", "")

        if not text or len(text) < 50:
            continue

        try:
            response = requests.post(
                NER_URL,
                json={"text": text[:50000]},  # Max 50k chars
                timeout=30,
            )

            if response.status_code == 200:
                ner_result = response.json()
                coll.update_one({"_id": article_id}, {"$set": {"ner": ner_result}})
                processed += 1
                print(f"  [{processed}] {article.get('title', '')[:50]}...")
            else:
                errors += 1
                print(f"  ⚠️ Error on {article_id}: {response.status_code}")

        except Exception as e:
            errors += 1
            print(f"  ⚠️ Exception: {e}")

    print(f"\nDone: {processed} processed, {errors} errors")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process articles with NER")
    parser.add_argument(
        "date", nargs="?", default="-1", help="Date (YYYY-MM-DD or -N days)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of articles")
    args = parser.parse_args()

    process_articles_since(args.date, args.limit)
