#!/usr/bin/env python3

"""
Module for mongo2training_balanced.py.
"""
"""
MongoDB to Balanced Training Data Exporter
============================================
Exports bias training data from MongoDB with balanced bins.

Usage:
    python mongo2training_balanced.py -o train.json
    python mongo2training_balanced.py --minsamples 1000 --totsamples 50000
"""

import json
import random
import argparse
from collections import defaultdict
from typing import Optional, Dict, List, Any
from pymongo import MongoClient


def get_mongo_client(uri: Optional[str] = None) -> MongoClient:
    """Get MongoDB client."""
    if uri is None:
        import os
        uri = os.getenv('MONGO_URI', 'mongodb://root:example@localhost:27017')
    return MongoClient(uri)


def load_buckets(collection, min_dominant: float = 0.5) -> Dict[str, List]:
    """Load articles and organize into bias buckets."""
    docs = list(collection.find({
        'bias': {'$exists': True, '$ne': None}
    }))
    
    buckets: Dict[str, List] = defaultdict(list)
    
    for doc in docs:
        bias = doc.get('bias', {})
        if isinstance(bias, str):
            continue
        
        dir_vals = bias.get('dir', {})
        deg_vals = bias.get('deg', {})
        
        if not dir_vals or not deg_vals:
            continue
        
        dominant_dir = max(dir_vals, key=dir_vals.get)
        dominant_deg = max(deg_vals, key=deg_vals.get)
        
        # Filter for dominant >= min_dominant
        if dir_vals[dominant_dir] >= min_dominant and deg_vals[dominant_deg] >= min_dominant:
            combo = f"{dominant_dir}-{dominant_deg}"
            buckets[combo].append(doc)
    
    return buckets


def allocate_samples(buckets: Dict[str, List], minsamples: int, 
                     totsamples: Optional[int] = None) -> Dict[str, int]:
    """Allocate samples per bin."""
    bins = sorted(buckets.keys())
    counts: Dict[str, int] = {}
    
    # First: allocate minsamples per bin (or all available)
    for combo in bins:
        counts[combo] = min(len(buckets[combo]), minsamples)
    
    allocated = sum(counts.values())
    
    # Second: distribute remaining up to totsamples
    if totsamples and allocated < totsamples:
        remaining = totsamples - allocated
        for combo in bins:
            if len(buckets[combo]) > minsamples:
                extra = min(remaining, len(buckets[combo]) - minsamples)
                counts[combo] += extra
                remaining -= extra
                if remaining <= 0:
                    break
    
    return counts


def export_training_data(collection, output: str, minsamples: int = 1000, 
                         totsamples: Optional[int] = None, min_dominant: float = 0.5,
                         seed: int = 42) -> List[Dict]:
    """Export balanced training data from MongoDB."""
    print(f"Loading articles from MongoDB...")
    buckets = load_buckets(collection, min_dominant)
    
    print(f"\nAvailable per bin (dominant >= {min_dominant}):")
    for combo in sorted(buckets.keys()):
        print(f"  {combo}: {len(buckets[combo])} available")
    
    # Calculate allocation
    print(f"\nParameters: minsamples={minsamples}", end="")
    if totsamples:
        print(f", totsamples={totsamples}")
    else:
        print()
    
    counts = allocate_samples(buckets, minsamples, totsamples)
    final_total = sum(counts.values())
    
    print(f"\nFinal distribution:")
    for combo in sorted(counts.keys()):
        available = len(buckets[combo])
        take = counts[combo]
        status = "" if take >= minsamples else f" (only {available} available)"
        print(f"  {combo}: {take} samples{status}")
    
    print(f"\nTotal samples: {final_total}")
    
    # Sample
    random.seed(seed)
    training_data = []
    bins = sorted(buckets.keys())
    
    for combo in bins:
        take = counts[combo]
        samples = random.sample(buckets[combo], take) if len(buckets[combo]) >= take else buckets[combo]
        
        for doc in samples:
            article = doc.get('article', doc.get('title', ''))
            bias = doc.get('bias', {})
            label = {
                'dir': bias.get('dir', {}),
                'deg': bias.get('deg', {}),
                'reason': bias.get('reason', '')
            }
            training_data.append({
                'article': article,
                'label': label
            })
    
    # Shuffle
    random.shuffle(training_data)
    
    # Save
    with open(output, 'w') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(training_data)} samples to {output}")
    return training_data


def main():
    parser = argparse.ArgumentParser(description='Export balanced training data from MongoDB')
    parser.add_argument('-o', '--output', default='train_mongodb.json',
                        help='Output JSON file')
    parser.add_argument('--uri', help='MongoDB connection URI')
    parser.add_argument('--database', default='rssnews',
                        help='Database name')
    parser.add_argument('--collection', default='articles',
                        help='Collection name')
    parser.add_argument('--minsamples', type=int, default=1000,
                        help='Minimum samples per bin (default: 1000)')
    parser.add_argument('--totsamples', type=int, default=None,
                        help='Maximum total samples (default: unlimited)')
    parser.add_argument('--min-dominant', type=float, default=0.5,
                        help='Minimum dominant bias value (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    client = get_mongo_client(args.uri)
    db = client[args.database]
    collection = db[args.collection]
    
    export_training_data(
        collection,
        output=args.output,
        minsamples=args.minsamples,
        totsamples=args.totsamples,
        min_dominant=args.min_dominant,
        seed=args.seed
    )
    
    client.close()


if __name__ == '__main__':
    main()
