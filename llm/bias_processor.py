#!/usr/bin/env python3
"""
Process articles from MongoDB and detect political bias using T5 server.

Finds articles with empty/missing 'bias' field, processes them in reverse
chronological order (most recent first), and writes bias detection results.

Usage:
    python bias_processor.py [--batch-size N] [--api-url URL]
"""

import os
import sys
import json
import re
import signal
import argparse
import requests
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId


class BiasProcessor:
    """Process articles for bias detection using T5 server."""

    def __init__(self, api_url: str = "http://localhost:8000", output_file: str | None = None):
        """
        Initialize MongoDB connection and API endpoint.

        Args:
            api_url: URL of the T5 bias detection server
            output_file: Path to save processed article IDs
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
        
        # Track processed articles
        self.processed_ids: list[dict] = []
        self.output_file = output_file or f"processed_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\nShutdown signal received. Finishing current article and saving...")
        self.shutdown_requested = True

    def save_processed_ids(self):
        """Save processed article IDs to JSON file."""
        if not self.processed_ids:
            print("No articles were processed.")
            return
        
        try:
            with open(self.output_file, "w") as f:
                json.dump(self.processed_ids, f, indent=2)
            print(f"Saved {len(self.processed_ids)} processed article IDs to: {self.output_file}")
        except Exception as e:
            print(f"Error saving processed IDs: {e}")

    def get_articles_without_bias(self, batch_size: int | None = None):
        """
        Get articles missing the bias field, sorted by published date descending.

        Args:
            batch_size: Number of articles to fetch per batch

        Returns:
            MongoDB cursor for articles without bias
        """
        query = {
            "$or": [
                {"bias": {"$exists": False}},
                {"bias": None},
                {"bias": ""},
            ],
            "article": {"$exists": True, "$ne": None, "$ne": ""},
        }

        cursor = self.collection.find(query).sort("published", -1)
        if batch_size:
            cursor = cursor.limit(batch_size)
        return cursor

    def count_articles_without_bias(self) -> int:
        """
        Count total articles missing the bias field.

        Returns:
            Count of articles without bias
        """
        query = {
            "$or": [
                {"bias": {"$exists": False}},
                {"bias": None},
                {"bias": ""},
            ],
            "article": {"$exists": True, "$ne": None, "$ne": ""},
        }
        return self.collection.count_documents(query)

    def repair_json_string(self, raw: str, pbar=None) -> dict | None:
        """
        Attempt to repair malformed JSON from LLM output.
        
        Handles common issues:
        - Missing outer braces: "dir":... → {"dir":...}
        - Missing nested braces: "dir":"L":0.2 → "dir":{"L":0.2}
        - Missing quotes on keys: dir": → "dir":
        - Unterminated strings: "reason":"text → "reason":"text"}
        - Extra quotes around JSON
        
        Args:
            raw: Raw string from model
            pbar: Progress bar for output
            
        Returns:
            Parsed dict or None if unrepairable
        """
        def output(msg: str):
            if pbar:
                pbar.write(msg)
            else:
                print(msg)
        
        if not raw:
            return None
        
        s = raw.strip()
        
        # Remove outer quotes if present
        if s.startswith('"') and s.endswith('"'):
            try:
                s = json.loads(s)
            except json.JSONDecodeError:
                s = s[1:-1]
        
        # Fix missing opening quote on "dir" at start
        if s.startswith('dir":'):
            s = '"' + s
        
        # Fix missing quotes on known keys (dir, deg, reason)
        s = re.sub(r'(?<!")\b(dir|deg|reason)\b(?!"):', r'"\1":', s)
        
        # Add outer braces if missing
        if not s.startswith('{'):
            s = '{' + s
        if not s.endswith('}'):
            s = s + '}'
        
        # Fix missing braces after "dir": and "deg":
        # Pattern: "dir":"L":0.2 → "dir":{"L":0.2}
        
        # Fix "dir":"L":value patterns
        s = re.sub(r'"dir"\s*:\s*"([LRC])"\s*:', r'"dir":{"\1":', s)
        
        # Fix "deg":"L":value patterns
        s = re.sub(r'"deg"\s*:\s*"([LMH])"\s*:', r'"deg":{"\1":', s)
        
        # Find positions of dir and deg objects
        dir_match = re.search(r'"dir"\s*:\s*\{', s)
        deg_match = re.search(r'"deg"\s*:\s*\{', s)
        reason_match = re.search(r'"reason"\s*:', s)
        
        if dir_match and deg_match:
            # Find where "deg" starts to close "dir" before it
            deg_start = deg_match.start()
            dir_section = s[:deg_start]
            
            # Check if dir section needs closing brace
            open_braces = dir_section.count('{') - dir_section.count('}')
            if open_braces > 0:
                # Insert closing brace before "deg"
                s = s[:deg_start] + '}' * (open_braces - 1) + ',' + s[deg_start:]
        
        # Re-find positions after potential modification
        deg_match = re.search(r'"deg"\s*:\s*\{', s)
        reason_match = re.search(r'"reason"\s*:', s)
        
        if deg_match and reason_match:
            # Close deg object before reason
            reason_start = reason_match.start()
            deg_section = s[:reason_start]
            
            open_braces = deg_section.count('{') - deg_section.count('}')
            if open_braces > 0:
                s = s[:reason_start] + '}' * (open_braces - 1) + ',' + s[reason_start:]
        
        # Fix unterminated string at end (reason value missing closing quote)
        # Count quotes - if odd number, add closing quote before final }
        if s.endswith('}'):
            # Find the last "reason":" value
            reason_val_match = re.search(r'"reason"\s*:\s*"([^"]*)$', s)
            if reason_val_match:
                # Add closing quote before the }
                s = s[:-1] + '"}'
        
        # Ensure proper closing at the end
        open_braces = s.count('{') - s.count('}')
        if open_braces > 0:
            s = s.rstrip('}') + '}' * (open_braces + 1)
        
        # Fix trailing commas before } (invalid JSON) - MUST BE LAST
        s = re.sub(r',\s*}', '}', s)
        
        # Try to parse
        try:
            result = json.loads(s)
            output(f"  JSON repair successful")
            return result
        except json.JSONDecodeError as e:
            output(f"  JSON repair failed: {e}")
            output(f"  Attempted to parse: {s[:200]}...")
            return None

    def normalize_bias_result(self, result: dict, pbar=None) -> dict | None:
        """
        Normalize and correct common LLM output issues.
        
        Handles:
        - Key name variations (left→L, center→C, right→R, low→L, medium→M, high→H)
        - Case variations (LEFT→L, Center→C, etc.)
        - Missing fields if we can infer them
        - Raw output that needs JSON repair
        
        Args:
            result: Raw bias result from API
            pbar: Progress bar for output (optional)
            
        Returns:
            Normalized result, or None if unfixable
        """
        def output(msg: str):
            if pbar:
                pbar.write(msg)
            else:
                print(msg)
        
        if not result:
            return None
        
        # Handle raw_output - attempt to repair the JSON
        if "raw_output" in result:
            raw = result["raw_output"]
            output(f"  Attempting JSON repair on raw output...")
            repaired = self.repair_json_string(raw, pbar)
            if repaired:
                result = repaired
            else:
                return None
        
        normalized = dict(result)
        corrections = []
        
        # Key mappings for normalization
        dir_key_map = {
            "left": "L", "l": "L", "LEFT": "L", "Left": "L",
            "center": "C", "c": "C", "CENTER": "C", "Center": "C", "centre": "C",
            "right": "R", "r": "R", "RIGHT": "R", "Right": "R",
        }
        
        deg_key_map = {
            "low": "L", "l": "L", "LOW": "L", "Low": "L",
            "medium": "M", "m": "M", "MEDIUM": "M", "Medium": "M", "moderate": "M",
            "high": "H", "h": "H", "HIGH": "H", "High": "H", "heavy": "H",
        }
        
        # Normalize "dir" field
        if "dir" in normalized and isinstance(normalized["dir"], dict):
            original_dir = normalized["dir"]
            new_dir = {}
            for key, value in original_dir.items():
                new_key = dir_key_map.get(key, key)
                if new_key != key:
                    corrections.append(f"dir: '{key}' → '{new_key}'")
                new_dir[new_key] = value
            normalized["dir"] = new_dir
        
        # Normalize "deg" field
        if "deg" in normalized and isinstance(normalized["deg"], dict):
            original_deg = normalized["deg"]
            new_deg = {}
            for key, value in original_deg.items():
                new_key = deg_key_map.get(key, key)
                if new_key != key:
                    corrections.append(f"deg: '{key}' → '{new_key}'")
                new_deg[new_key] = value
            normalized["deg"] = new_deg
        
        # Handle alternative field names for "dir"
        if "dir" not in normalized:
            for alt in ["direction", "bias_dir", "political_dir", "orientation"]:
                if alt in normalized:
                    normalized["dir"] = normalized.pop(alt)
                    corrections.append(f"field: '{alt}' → 'dir'")
                    break
        
        # Handle alternative field names for "deg"
        if "deg" not in normalized:
            for alt in ["degree", "bias_deg", "intensity", "strength"]:
                if alt in normalized:
                    normalized["deg"] = normalized.pop(alt)
                    corrections.append(f"field: '{alt}' → 'deg'")
                    break
        
        # Handle alternative field names for "reason"
        if "reason" not in normalized:
            for alt in ["explanation", "rationale", "analysis", "why", "justification"]:
                if alt in normalized:
                    normalized["reason"] = normalized.pop(alt)
                    corrections.append(f"field: '{alt}' → 'reason'")
                    break
        
        # Print corrections if any were made
        if corrections:
            output(f"  Corrections applied:")
            for correction in corrections:
                output(f"    {correction}")
        
        return normalized

    def validate_bias_result(self, result: dict, pbar=None) -> dict | None:
        """
        Validate and clean bias result to ensure proper format.
        
        Expected format: {"dir": {...}, "deg": {...}, "reason": "..."}
        
        Args:
            result: Raw bias result from API
            pbar: Progress bar for output (optional)
            
        Returns:
            Validated/cleaned result, or None if invalid
        """
        def output(msg: str):
            if pbar:
                pbar.write(msg)
            else:
                print(msg)
        
        if not result:
            output("  Warning: Empty result from model")
            return None
        
        # First normalize the result
        normalized = self.normalize_bias_result(result, pbar)
        if not normalized:
            return None
        
        # Check for required fields
        required_fields = ["dir", "deg", "reason"]
        missing = [f for f in required_fields if f not in normalized]
        if missing:
            output(f"  Warning: Missing required fields: {missing}")
            output(f"  Model output: {json.dumps(normalized, indent=2)}")
            return None
        
        # Validate dir structure
        if not isinstance(normalized["dir"], dict):
            output(f"  Warning: 'dir' is not a dict")
            output(f"  Model output: {json.dumps(normalized, indent=2)}")
            return None
        
        dir_keys = set(normalized["dir"].keys())
        if not dir_keys.issuperset({"L", "C", "R"}):
            output(f"  Warning: 'dir' missing L/C/R keys, got: {list(dir_keys)}")
            output(f"  Model output: {json.dumps(normalized, indent=2)}")
            return None
        
        # Validate deg structure
        if not isinstance(normalized["deg"], dict):
            output(f"  Warning: 'deg' is not a dict")
            output(f"  Model output: {json.dumps(normalized, indent=2)}")
            return None
        
        deg_keys = set(normalized["deg"].keys())
        if not deg_keys.issuperset({"L", "M", "H"}):
            output(f"  Warning: 'deg' missing L/M/H keys, got: {list(deg_keys)}")
            output(f"  Model output: {json.dumps(normalized, indent=2)}")
            return None
        
        # Build clean result with proper key order
        clean_result = {
            "dir": {
                "L": normalized["dir"]["L"],
                "C": normalized["dir"]["C"],
                "R": normalized["dir"]["R"]
            },
            "deg": {
                "L": normalized["deg"]["L"],
                "M": normalized["deg"]["M"],
                "H": normalized["deg"]["H"]
            },
            "reason": normalized["reason"]
        }
        
        return clean_result

    def detect_bias(self, text: str, pbar=None) -> dict | None:
        """
        Call the T5 bias detection API.

        Args:
            text: Article text to analyze
            pbar: Progress bar for output (optional)

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
            raw_result = data.get("result", {})
            
            # Validate and clean the result
            return self.validate_bias_result(raw_result, pbar)
            
        except requests.exceptions.RequestException as e:
            if pbar:
                pbar.write(f"  API Error: {e}")
            else:
                print(f"API Error: {e}")
            return None

    def update_bias_field(self, article_id: ObjectId, bias_result: dict) -> bool:
        """
        Update the bias field in MongoDB.

        Args:
            article_id: MongoDB document ID
            bias_result: Bias detection result to store

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.collection.update_one(
                {"_id": article_id},
                {"$set": {"bias": json.dumps(bias_result)}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"\nMongoDB Error: {e}")
            return False

    def process_articles(self, batch_size: int | None = None, dry_run: bool = False, max_failures: int = 3):
        """
        Process all articles missing bias field.

        Args:
            batch_size: Optional limit on number of articles to process
            dry_run: If True, don't write to database
            max_failures: Stop after this many consecutive failures (default: 3)
        """
        total_count = self.count_articles_without_bias()
        
        if total_count == 0:
            print("No articles found without bias field.")
            return

        if batch_size:
            total_count = min(total_count, batch_size)
            print(f"Processing up to {total_count} articles (batch limit)...")
        else:
            print(f"Found {total_count} articles to process.")
        
        print(f"Max failures before shutdown: {max_failures}")

        cursor = self.get_articles_without_bias(batch_size)

        processed = 0
        failed = 0
        consecutive_failures = 0

        with tqdm(total=total_count, desc="Processing articles", unit="article") as pbar:
            for article in cursor:
                # Check for shutdown request
                if self.shutdown_requested:
                    break

                article_id = article["_id"]
                article_text = article.get("article", "")
                title = article.get("title", "Unknown title")

                if not article_text or not article_text.strip():
                    pbar.update(1)
                    continue

                # Print article ID being processed
                pbar.write(f"Processing: {article_id}")

                # Detect bias
                bias_result = self.detect_bias(article_text, pbar)

                if bias_result is None:
                    failed += 1
                    consecutive_failures += 1
                    pbar.write(f"  FAILED - API/validation error (consecutive: {consecutive_failures}/{max_failures})")
                    pbar.set_postfix(failed=failed)
                    pbar.update(1)
                    
                    # Check if we've hit max failures
                    if consecutive_failures >= max_failures:
                        pbar.write(f"\nMax failures ({max_failures}) reached. Initiating graceful shutdown...")
                        self.shutdown_requested = True
                    continue

                # Reset consecutive failures on success
                consecutive_failures = 0

                # Print the bias result
                pbar.write(f"  Result: {json.dumps(bias_result, indent=2)}")

                # Update MongoDB
                success = False
                if not dry_run:
                    success = self.update_bias_field(article_id, bias_result)
                    if not success:
                        failed += 1
                        consecutive_failures += 1
                        pbar.write(f"  FAILED - MongoDB write error (consecutive: {consecutive_failures}/{max_failures})")
                        pbar.set_postfix(failed=failed)
                        
                        # Check if we've hit max failures
                        if consecutive_failures >= max_failures:
                            pbar.write(f"\nMax failures ({max_failures}) reached. Initiating graceful shutdown...")
                            self.shutdown_requested = True
                    else:
                        pbar.write(f"  SUCCESS - Updated bias field")
                else:
                    pbar.write(f"  [DRY RUN] Would update bias field")
                    success = True

                # Track processed article
                self.processed_ids.append({
                    "id": str(article_id),
                    "title": title[:100],
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "bias_result": bias_result
                })

                processed += 1
                pbar.update(1)

        print(f"\nCompleted: {processed} processed, {failed} failed")
        
        # Save processed IDs on completion
        self.save_processed_ids()

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process articles for bias detection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Limit number of articles to process",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL of T5 bias detection server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save processed article IDs (default: processed_articles_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to database, just show what would be done",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Stop after this many consecutive failures (default: 3)",
    )
    args = parser.parse_args()

    processor = None
    try:
        processor = BiasProcessor(api_url=args.api_url, output_file=args.output_file)
        processor.process_articles(
            batch_size=args.batch_size, 
            dry_run=args.dry_run,
            max_failures=args.max_failures
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Already handled by signal handler, but catch just in case
        print("\nInterrupted by user")
        if processor:
            processor.save_processed_ids()
        sys.exit(0)
    finally:
        if processor:
            processor.close()


if __name__ == "__main__":
    main()
