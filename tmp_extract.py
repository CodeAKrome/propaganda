#!/usr/bin/env python3
"""Iran/Hormuz news extraction from MongoDB."""

import pymongo
from datetime import datetime, timedelta
import sys

client = pymongo.MongoClient("mongodb://root:example@localhost:27017")
coll = client["rssnews"]["articles"]

days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
output = sys.argv[2] if len(sys.argv) > 2 else "output/iran_hormuz.txt"

start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

query = {
    "published": {"$gte": start},
    "$or": [
        {"title": {"$regex": "Iran", "$options": "i"}},
        {"title": {"$regex": "Hormuz", "$options": "i"}},
    ],
}

print(f"Extracting articles since {start}...")

count = 0
with open(output, "w") as f:
    for doc in coll.find(query).limit(500):
        id_str = str(doc["_id"])
        title = doc.get("title", "")[:200]
        source = doc.get("source", "")

        f.write(f"ID:{id_str}\n")
        f.write(f"SOURCE:{source}\n")
        f.write(f"TITLE:{title}\n")
        f.write("\n")
        count += 1

print(f"Extracted {count} articles to {output}")
