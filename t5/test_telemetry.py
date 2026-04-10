#!/usr/bin/env python3

"""
Module for test_telemetry.py.
"""
"""Quick test to verify MongoDB telemetry is working."""

import pymongo

MONGO_URI = "mongodb://root:example@localhost:27017"
MONGO_DB = "rssnews"
MONGO_TELEMETRY_COLL = "training_telemetry"

def main():
    client = pymongo.MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_TELEMETRY_COLL]
    
    # Count all documents
    total = coll.count_documents({})
    print(f"Total documents in {MONGO_TELEMETRY_COLL}: {total}")
    
    # Get sample documents
    docs = list(coll.find({}).limit(5))
    print(f"\nFirst {len(docs)} documents:")
    for doc in docs:
        print(f"  - run_id: {doc.get('run_id', 'N/A')}")
        print(f"    event: {doc.get('event', 'N/A')}")
        print(f"    step: {doc.get('step', 'N/A')}")
        print(f"    timestamp: {doc.get('timestamp', 'N/A')}")
        if doc.get('metrics'):
            print(f"    metrics: {doc.get('metrics')}")
        print()

if __name__ == "__main__":
    main()
