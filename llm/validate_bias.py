#!/usr/bin/env python3
"""
Validate bias detection results by comparing stored MongoDB values
with fresh predictions from the T5 server.

Usage:
    python validate_bias.py [--sample N] [--api-url URL]
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId


class BiasValidator:
    """Validate stored bias results against fresh predictions."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize MongoDB connection and API endpoint.

        Args:
            api_url: URL of the T5 bias detection server
        """
        mongo_user = os.getenv("MONGO_USER")
        mongo_pass = os.getenv("MONGO_PASS")

        if not mongo_user or not mongo_pass:
            raise ValueError(
                "MONGO_USER and MONGO_PASS environment variables must be set"
            )

        uri = f"mongodb://{mongo_user}:{mongo_pass}@localhost:27017"
        self.client = MongoClient(uri)
        self.db = self.client["rssnews"]
        self.collection = self.db["articles"]
        self.api_url = api_url.rstrip("/")

    def get_articles_with_bias(self, sample_size: int = 10):
        """
        Get a sample of articles that have bias field populated.

        Args:
            sample_size: Number of articles to sample

        Returns:
            List of articles with bias field
        """
        query = {
            "bias": {"$exists": True, "$ne": None, "$ne": ""},
            "article": {"$exists": True, "$ne": None, "$ne": ""},
        }

        # Get random sample using $sample aggregation
        pipeline = [
            {"$match": query},
            {"$sample": {"size": sample_size}}
        ]

        return list(self.collection.aggregate(pipeline))

    def count_articles_with_bias(self) -> int:
        """
        Count total articles with bias field populated.

        Returns:
            Count of articles with bias
        """
        query = {
            "bias": {"$exists": True, "$ne": None, "$ne": ""},
        }
        return self.collection.count_documents(query)

    def parse_stored_bias(self, bias_str: str) -> dict | None:
        """
        Parse stored bias string to dict.

        Args:
            bias_str: Stored bias value (may be JSON string or dict)

        Returns:
            Parsed dict or None if invalid
        """
        if not bias_str:
            return None

        if isinstance(bias_str, dict):
            return bias_str

        try:
            return json.loads(bias_str)
        except json.JSONDecodeError:
            return None

    def detect_bias(self, text: str) -> dict | None:
        """
        Call the T5 bias detection API.

        Args:
            text: Article text to analyze

        Returns:
            Dictionary with bias detection results, or None on error
        """
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"text": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def compare_bias(self, stored: dict, predicted: dict) -> dict:
        """
        Compare stored and predicted bias values.

        Args:
            stored: Stored bias dict
            predicted: Newly predicted bias dict

        Returns:
            Comparison result dict
        """
        result = {
            "match": True,
            "dir_diff": {},
            "deg_diff": {},
            "reason_match": False,
        }

        # Compare dir values
        if "dir" in stored and "dir" in predicted:
            for key in ["L", "C", "R"]:
                stored_val = stored["dir"].get(key, 0)
                pred_val = predicted["dir"].get(key, 0)
                diff = abs(stored_val - pred_val)
                if diff > 0.05:  # 5% tolerance
                    result["match"] = False
                result["dir_diff"][key] = {
                    "stored": stored_val,
                    "predicted": pred_val,
                    "diff": round(diff, 3)
                }

        # Compare deg values
        if "deg" in stored and "deg" in predicted:
            for key in ["L", "M", "H"]:
                stored_val = stored["deg"].get(key, 0)
                pred_val = predicted["deg"].get(key, 0)
                diff = abs(stored_val - pred_val)
                if diff > 0.05:  # 5% tolerance
                    result["match"] = False
                result["deg_diff"][key] = {
                    "stored": stored_val,
                    "predicted": pred_val,
                    "diff": round(diff, 3)
                }

        # Compare reason (just check if both exist)
        if "reason" in stored and "reason" in predicted:
            result["reason_match"] = True
            result["stored_reason_preview"] = stored["reason"][:100] + "..." if len(stored["reason"]) > 100 else stored["reason"]
            result["predicted_reason_preview"] = predicted["reason"][:100] + "..." if len(predicted["reason"]) > 100 else predicted["reason"]

        return result

    def validate(self, sample_size: int = 10, output_file: str | None = None):
        """
        Validate stored bias against fresh predictions.

        Args:
            sample_size: Number of articles to sample
            output_file: Optional file to save validation results
        """
        total_with_bias = self.count_articles_with_bias()
        print(f"Total articles with bias: {total_with_bias}")
        print(f"Sampling {sample_size} articles for validation...\n")

        articles = self.get_articles_with_bias(sample_size)

        if not articles:
            print("No articles found with bias field.")
            return

        results = []
        matches = 0
        mismatches = 0
        errors = 0

        for article in tqdm(articles, desc="Validating", unit="article"):
            article_id = article["_id"]
            title = article.get("title", "Unknown")
            article_text = article.get("article", "")
            stored_bias_str = article.get("bias", "")

            # Parse stored bias
            stored_bias = self.parse_stored_bias(stored_bias_str)
            if not stored_bias:
                print(f"\n  Error: Could not parse stored bias for {article_id}")
                errors += 1
                continue

            # Get fresh prediction
            predicted_bias = self.detect_bias(article_text)
            if not predicted_bias:
                print(f"\n  Error: Could not get prediction for {article_id}")
                errors += 1
                continue

            # Compare
            comparison = self.compare_bias(stored_bias, predicted_bias)

            result = {
                "id": str(article_id),
                "title": title[:80],
                "match": comparison["match"],
                "stored": stored_bias,
                "predicted": predicted_bias,
                "comparison": comparison
            }
            results.append(result)

            if comparison["match"]:
                matches += 1
            else:
                mismatches += 1
                tqdm.write(f"\nMismatch: {article_id}")
                tqdm.write(f"  Title: {title[:60]}...")
                tqdm.write(f"  Dir diff: {comparison['dir_diff']}")
                tqdm.write(f"  Deg diff: {comparison['deg_diff']}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Validation Summary")
        print(f"{'='*60}")
        print(f"Total sampled: {len(articles)}")
        print(f"Matches: {matches}")
        print(f"Mismatches: {mismatches}")
        print(f"Errors: {errors}")
        print(f"Match rate: {matches/(matches+mismatches)*100:.1f}%" if (matches+mismatches) > 0 else "N/A")

        # Save results if output file specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total": len(articles),
                        "matches": matches,
                        "mismatches": mismatches,
                        "errors": errors
                    },
                    "results": results
                }, f, indent=2)
            print(f"\nResults saved to: {output_file}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate stored bias results against fresh predictions"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of articles to sample (default: 10)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL of T5 bias detection server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save validation results to JSON file",
    )
    args = parser.parse_args()

    try:
        validator = BiasValidator(api_url=args.api_url)
        validator.validate(sample_size=args.sample, output_file=args.output)
        validator.close()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
