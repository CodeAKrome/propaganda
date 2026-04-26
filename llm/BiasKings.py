#!/usr/bin/env python3
"""
BiasKings.py - Find the most biased articles in MongoDB

Queries MongoDB for articles with the highest Left, Center, and Right bias scores.
"""

import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import json


def get_mongo_client():
    """Initialize MongoDB connection using MONGO_URI env var."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        return client
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB: {e}")
        print(f"Ensure MONGO_URI is set correctly (currently: {mongo_uri})")
        sys.exit(1)


def get_top_bias_articles(collection, direction: str, limit: int = 3):
    """
    Get articles with highest bias in specified direction.
    Only considers articles with high degree (H), then sorts by direction.

    Args:
        collection: MongoDB articles collection
        direction: 'L' (Left), 'C' (Center), or 'R' (Right)
        limit: Number of results to return

    Returns:
        List of articles with high degree, sorted by direction
    """
    pipeline = [
        {
            "$match": {
                "bias": {"$exists": True},
                f"bias.dir.{direction}": {"$exists": True, "$gt": 0},
                "bias.deg.H": {"$exists": True, "$gt": 0.5},
            }
        },
        {"$sort": {f"bias.dir.{direction}": -1}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 1,
                "title": 1,
                "source": 1,
                "published": 1,
                "bias.dir": 1,
                "bias.deg": 1,
                "bias.reason": 1,
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def print_results(articles: list, direction: str, label: str):
    """Print formatted results for a bias direction."""
    direction_emoji = {"L": "🟦 LEFT", "C": "⚪ CENTER", "R": "🔴 RIGHT"}

    print(f"\n{'=' * 60}")
    print(f"TOP 3 {direction_emoji.get(direction, direction)} BIAS ARTICLES")
    print(f"{'=' * 60}")

    if not articles:
        print("No articles found with bias data.")
        return

    for i, article in enumerate(articles, 1):
        bias_dir = article.get("bias", {}).get("dir", {})
        bias_deg = article.get("bias", {}).get("deg", {})
        bias_reason = article.get("bias", {}).get("reason", "N/A")

        left = bias_dir.get("L", 0)
        center = bias_dir.get("C", 0)
        right = bias_dir.get("R", 0)

        print(f"\n#{i} | MongoDB ID: {article['_id']}")
        print(f"    Title: {article.get('title', 'N/A')[:80]}...")
        print(f"    Source: {article.get('source', 'N/A')}")
        print(f"    Published: {article.get('published', 'N/A')}")
        print(f"    Bias Scores: L={left:.3f} | C={center:.3f} | R={right:.3f}")
        print(
            f"    Degree: L={bias_deg.get('L', 0):.3f} | M={bias_deg.get('M', 0):.3f} | H={bias_deg.get('H', 0):.3f}"
        )
        print(f"    Reason: {bias_reason[:150]}...")


def main():
    """Main entry point."""
    print("=" * 60)
    print("BIAS KINGS - Finding Most Biased Articles")
    print("=" * 60)

    client = get_mongo_client()
    db = client["rssnews"]
    collection = db["articles"]

    print(f"\nConnected to: {db.name}")

    # Get counts
    total = collection.count_documents({"bias": {"$exists": True}})
    print(f"Total articles with bias data: {total}")

    # Get top left
    left_articles = get_top_bias_articles(collection, "L")
    print_results(left_articles, "L", "Left")

    # Get top center
    center_articles = get_top_bias_articles(collection, "C")
    print_results(center_articles, "C", "Center")

    # Get top right
    right_articles = get_top_bias_articles(collection, "R")
    print_results(right_articles, "R", "Right")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(
        f"{'Rank':<6} {'Direction':<10} {'MongoDB ID':<30} {'L':>6} {'C':>6} {'R':>6}"
    )
    print("-" * 60)

    for i, article in enumerate(left_articles, 1):
        d = article.get("bias", {}).get("dir", {})
        print(
            f"{i:<6} {'LEFT':<10} {str(article['_id'])[:28]:<30} {d.get('L', 0):>6.3f} {d.get('C', 0):>6.3f} {d.get('R', 0):>6.3f}"
        )

    for i, article in enumerate(center_articles, 1):
        d = article.get("bias", {}).get("dir", {})
        print(
            f"{i:<6} {'CENTER':<10} {str(article['_id'])[:28]:<30} {d.get('L', 0):>6.3f} {d.get('C', 0):>6.3f} {d.get('R', 0):>6.3f}"
        )

    for i, article in enumerate(right_articles, 1):
        d = article.get("bias", {}).get("dir", {})
        print(
            f"{i:<6} {'RIGHT':<10} {str(article['_id'])[:28]:<30} {d.get('L', 0):>6.3f} {d.get('C', 0):>6.3f} {d.get('R', 0):>6.3f}"
        )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")

    client.close()


if __name__ == "__main__":
    main()
