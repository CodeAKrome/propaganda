#!/usr/bin/env python3

"""
Large Language Model integration and processing module.
"""
"""
Compare local bias inference (stored in MongoDB) against commercial LLM services.

This program:
1. Fetches articles from MongoDB with stored local bias inference
2. Uses Google Gemini (with failover) as the commercial LLM to compute bias on the same articles
3. Falls back to Groq if Gemini fails
4. Compares the commercial result with the stored local result
5. Generates an accuracy report

Usage:
    python commercial_bias_comparison.py --sample 20
    python commercial_bias_comparison.py --sample 20 --output report.json
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from typing import Optional, Tuple
from tqdm import tqdm

try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
except ImportError:
    print("Error: pymongo is required. Install with: pip install pymongo")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Gemini service will not be available.")
    genai = None

try:
    from groq import Groq
except ImportError:
    print("Warning: groq not installed. Groq service will not be available.")
    Groq = None


# Bias detection prompt template
BIAS_PROMPT_TEMPLATE = """You are an expert political analyst. Analyze the following news article for political bias.

Provide your response in JSON format with the following structure:
{{
    "dir": {{"L": <float>, "C": <float>, "R": <float>}},
    "deg": {{"L": <float>, "M": <float>, "H": <float>}},
    "reason": "<brief explanation>"
}}

Where:
- dir: Direction scores (L=Left, C=Center, R=Right) that sum to 1.0
- deg: Degree of bias (L=Low, M=Medium, H=High) that sum to 1.0
- reason: A Brief explanation (1-3 sentences) of your classification

Article:
{article_text}

Respond ONLY with valid JSON, no additional text:"""


# Model failover configurations (similar to db/report.py)
GEMINI_MODEL_PAIRS = [
    ("gemini", "models/gemini-3.1-pro-preview"),
    ("gemini", "models/gemini-3-flash-preview"),
    ("gemini", "models/gemini-2.5-pro"),
    ("gemini", "models/gemini-2.5-flash"),
    ("gemini", "gemini-2.0-flash"),
]

GROQ_MODEL_PAIRS = [
    ("groq", "llama-3.1-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "mixtral-8x7b-32768"),
]

ALL_MODEL_PAIRS = GEMINI_MODEL_PAIRS + GROQ_MODEL_PAIRS


class CommercialBiasComparator:
    """Compare local MongoDB bias inference against commercial LLM services."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_url: str = "http://localhost:8000"
    ):
        """
        Initialize the comparator with automatic failover.

        Args:
            model: Optional specific model to use (bypass failover)
            api_url: URL of local T5 server (for reference)
        """
        self.api_url = api_url
        self.used_models = []  # Track which models were attempted
        
        # Initialize MongoDB
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
        
        # Initialize commercial LLM clients
        self.gemini_model = None
        self.groq_client = None
        self.current_model = None
        self._init_clients(model)

    def _init_clients(self, specific_model: Optional[str] = None):
        """Initialize LLM clients based on available services."""
        # Initialize Gemini if available
        if genai:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                except Exception as e:
                    print(f"Warning: Could not configure Gemini: {e}")
        
        # Initialize Groq if available
        if Groq:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    self.groq_client = Groq(api_key=groq_key)
                except Exception as e:
                    print(f"Warning: Could not initialize Groq: {e}")
        
        # Set specific model if provided
        if specific_model:
            self.current_model = specific_model
            self._setup_model(specific_model)

    def _setup_model(self, model_name: str) -> bool:
        """Setup a specific model. Returns True if successful."""
        self.current_model = model_name
        
        # Check if it's a Gemini model
        if model_name.startswith("gemini-") or model_name.startswith("models/"):
            if not genai:
                return False
            try:
                self.gemini_model = genai.GenerativeModel(model_name)
                return True
            except Exception as e:
                print(f"Warning: Could not setup Gemini model {model_name}: {e}")
                return False
        
        # Check if it's a Groq model
        if not self.groq_client:
            return False
        
        # Groq uses the client directly with model name in the call
        return True

    def _try_model_with_fallback(self, model_pair: Tuple[str, str]) -> Optional[dict]:
        """
        Try a specific model pair, return bias dict or None.
        Model pair is (service, model_name)
        """
        service, model_name = model_pair
        self.used_models.append(model_name)
        
        if service == "gemini" and self.gemini_model:
            try:
                return {"service": service, "model": model_name}
            except Exception as e:
                print(f"Gemini {model_name} failed: {e}")
                return None
        
        elif service == "groq" and self.groq_client:
            try:
                return {"service": service, "model": model_name}
            except Exception as e:
                print(f"Groq {model_name} failed: {e}")
                return None
        
        return None

    def get_articles_with_bias(self, sample_size: int = 10) -> list:
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

        pipeline = [
            {"$match": query},
            {"$sample": {"size": sample_size}}
        ]

        return list(self.collection.aggregate(pipeline))

    def count_articles_with_bias(self) -> int:
        """Count total articles with bias field populated."""
        query = {
            "bias": {"$exists": True, "$ne": None, "$ne": ""},
        }
        return self.collection.count_documents(query)

    def parse_stored_bias(self, bias_str) -> Optional[dict]:
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

    def detect_bias_with_fallback(self, text: str, model_pairs: list) -> Optional[dict]:
        """
        Detect bias with automatic failover through model pairs.

        Args:
            text: Article text to analyze
            model_pairs: List of (service, model) tuples to try in order

        Returns:
            Dictionary with bias detection results, or None on all failures
        """
        # Truncate text if too long
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        prompt = BIAS_PROMPT_TEMPLATE.format(article_text=text)

        for service, model_name in model_pairs:
            try:
                if service == "gemini" and genai:
                    # Try to setup Gemini model
                    if not self._setup_model(model_name):
                        continue
                    
                    response = self.gemini_model.generate_content(prompt)
                    response_text = str(response.text) if response.text else ""
                    
                elif service == "groq" and self.groq_client:
                    response = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    response_text = str(response.choices[0].message.content) if response.choices[0].message.content else ""
                else:
                    continue

                # Track what we used
                self.current_model = f"{service}:{model_name}"
                
                # Parse JSON from response
                result = self._parse_bias_response(response_text)
                if result:
                    result["_service"] = service
                    result["_model"] = model_name
                    return result
                    
            except Exception as e:
                print(f"  Error with {service}:{model_name}: {e}")
                continue

        print(f"  All model pairs failed for this article")
        return None

    def detect_bias_commercial(self, text: str) -> Optional[dict]:
        """
        Detect bias using commercial LLM with failover.
        Tries Gemini models first, then Groq models.
        """
        return self.detect_bias_with_fallback(text, ALL_MODEL_PAIRS)

    def _parse_bias_response(self, response_text: str) -> Optional[dict]:
        """
        Parse bias response from commercial LLM.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed bias dict or None
        """
        # Extract JSON from response (handle potential markdown code blocks)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if not json_match:
            # Try to find any JSON-like structure
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{'):
                    json_match = re.search(r'\{.*\}', line, re.DOTALL)
                    if json_match:
                        break
        
        if not json_match:
            print(f"Could not extract JSON from response: {response_text[:200]}")
            return None

        try:
            bias_data = json.loads(json_match.group())
            
            # Validate and normalize the structure
            result = {
                "dir": {
                    "L": float(bias_data.get("dir", {}).get("L", 0.33)),
                    "C": float(bias_data.get("dir", {}).get("C", 0.33)),
                    "R": float(bias_data.get("dir", {}).get("R", 0.34)),
                },
                "deg": {
                    "L": float(bias_data.get("deg", {}).get("L", 0.33)),
                    "M": float(bias_data.get("deg", {}).get("M", 0.33)),
                    "H": float(bias_data.get("deg", {}).get("H", 0.34)),
                },
                "reason": str(bias_data.get("reason", ""))
            }
            
            # Normalize to sum to 1.0
            total_dir = sum(result["dir"].values())
            if total_dir > 0:
                result["dir"] = {k: v/total_dir for k, v in result["dir"].items()}
            
            total_deg = sum(result["deg"].values())
            if total_deg > 0:
                result["deg"] = {k: v/total_deg for k, v in result["deg"].items()}
            
            return result
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing bias response: {e}")
            return None

    def compare_bias(self, local: dict, commercial: dict) -> dict:
        """
        Compare local and commercial bias values.

        Args:
            local: Local bias dict from MongoDB
            commercial: Commercial LLM bias dict

        Returns:
            Comparison result dict
        """
        result = {
            "match": True,
            "direction_match": True,
            "degree_match": True,
            "dir_diff": {},
            "deg_diff": {},
            "dir_mae": 0.0,
            "deg_mae": 0.0,
        }

        # Compare direction values
        if "dir" in local and "dir" in commercial:
            dir_diffs = []
            for key in ["L", "C", "R"]:
                local_val = local["dir"].get(key, 0)
                comm_val = commercial["dir"].get(key, 0)
                diff = abs(local_val - comm_val)
                dir_diffs.append(diff)
                if diff > 0.15:  # 15% tolerance for direction
                    result["direction_match"] = False
                result["dir_diff"][key] = {
                    "local": round(local_val, 3),
                    "commercial": round(comm_val, 3),
                    "diff": round(diff, 3)
                }
            result["dir_mae"] = sum(dir_diffs) / len(dir_diffs)

        # Compare degree values
        if "deg" in local and "deg" in commercial:
            deg_diffs = []
            for key in ["L", "M", "H"]:
                local_val = local["deg"].get(key, 0)
                comm_val = commercial["deg"].get(key, 0)
                diff = abs(local_val - comm_val)
                deg_diffs.append(diff)
                if diff > 0.15:  # 15% tolerance for degree
                    result["degree_match"] = False
                result["deg_diff"][key] = {
                    "local": round(local_val, 3),
                    "commercial": round(comm_val, 3),
                    "diff": round(diff, 3)
                }
            result["deg_mae"] = sum(deg_diffs) / len(deg_diffs)

        # Overall match requires both direction and degree to match
        result["match"] = result["direction_match"] and result["degree_match"]

        return result

    def get_dominant_bias(self, bias: dict) -> str:
        """Get the dominant bias direction."""
        if "dir" in bias:
            dir_vals = bias["dir"]
            return max(dir_vals, key=dir_vals.get)
        return "Unknown"

    def run_comparison(self, sample_size: int = 10, output_file: Optional[str] = None):
        """
        Run the bias comparison.

        Args:
            sample_size: Number of articles to sample
            output_file: Optional file to save results
        """
        total_with_bias = self.count_articles_with_bias()
        print(f"Total articles with bias in MongoDB: {total_with_bias}")
        print(f"Using automatic model failover (Gemini -> Groq)")
        print(f"Sampling {sample_size} articles for comparison...\n")

        articles = self.get_articles_with_bias(sample_size)

        if not articles:
            print("No articles found with bias field.")
            return

        results = []
        matches = 0
        direction_matches = 0
        degree_matches = 0
        errors = 0

        total_dir_mae = 0.0
        total_deg_mae = 0.0

        # Track model usage
        model_success_count = {}

        for article in tqdm(articles, desc="Comparing with commercial LLMs", unit="article"):
            article_id = article["_id"]
            title = article.get("title", "Unknown")[:80]
            article_text = article.get("article", "")
            local_bias_str = article.get("bias", "")

            # Parse local bias from MongoDB
            local_bias = self.parse_stored_bias(local_bias_str)
            if not local_bias:
                print(f"\nError: Could not parse local bias for {article_id}")
                errors += 1
                continue

            # Reset used models for this article
            self.used_models = []
            
            # Get commercial LLM bias with failover
            commercial_bias = self.detect_bias_commercial(article_text)
            if not commercial_bias:
                print(f"\nError: Could not get commercial bias for {article_id}")
                errors += 1
                continue

            # Track which model succeeded
            model_key = f"{commercial_bias.get('_service', 'unknown')}:{commercial_bias.get('_model', 'unknown')}"
            model_success_count[model_key] = model_success_count.get(model_key, 0) + 1
            
            # Remove internal tracking keys before storing
            service_used = commercial_bias.pop("_service", None)
            model_used = commercial_bias.pop("_model", None)

            # Compare
            comparison = self.compare_bias(local_bias, commercial_bias)

            result = {
                "id": str(article_id),
                "title": title,
                "local_dominant": self.get_dominant_bias(local_bias),
                "commercial_dominant": self.get_dominant_bias(commercial_bias),
                "model_used": f"{service_used}:{model_used}" if service_used and model_used else "unknown",
                "match": comparison["match"],
                "direction_match": comparison["direction_match"],
                "degree_match": comparison["degree_match"],
                "local": local_bias,
                "commercial": commercial_bias,
                "comparison": comparison
            }
            results.append(result)

            if comparison["match"]:
                matches += 1
            if comparison["direction_match"]:
                direction_matches += 1
            if comparison["degree_match"]:
                degree_matches += 1

            total_dir_mae += comparison["dir_mae"]
            total_deg_mae += comparison["deg_mae"]

        # Calculate statistics
        valid_count = len(results)
        if valid_count > 0:
            avg_dir_mae = total_dir_mae / valid_count
            avg_deg_mae = total_deg_mae / valid_count
        else:
            avg_dir_mae = 0.0
            avg_deg_mae = 0.0

        # Print summary
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY: Local (MongoDB) vs Commercial LLM with Failover")
        print(f"{'='*70}")
        print(f"Total sampled: {len(articles)}")
        print(f"Valid comparisons: {valid_count}")
        print(f"Errors: {errors}")
        print(f"\nModel Usage:")
        for model, count in sorted(model_success_count.items(), key=lambda x: -x[1]):
            print(f"  - {model}: {count} articles")
        print(f"\nAccuracy Metrics:")
        print(f"  - Overall Match Rate: {matches}/{valid_count} ({matches/valid_count*100:.1f}%)" if valid_count > 0 else "  - Overall Match Rate: N/A")
        print(f"  - Direction Match Rate: {direction_matches}/{valid_count} ({direction_matches/valid_count*100:.1f}%)" if valid_count > 0 else "  - Direction Match Rate: N/A")
        print(f"  - Degree Match Rate: {degree_matches}/{valid_count} ({degree_matches/valid_count*100:.1f}%)" if valid_count > 0 else "  - Degree Match Rate: N/A")
        print(f"\nMean Absolute Error:")
        print(f"  - Direction MAE: {avg_dir_mae:.3f}")
        print(f"  - Degree MAE: {avg_deg_mae:.3f}")
        print(f"  - Combined MAE: {(avg_dir_mae + avg_deg_mae)/2:.3f}")

        # Direction distribution comparison
        local_directions = {}
        commercial_directions = {}
        for r in results:
            ld = r["local_dominant"]
            cd = r["commercial_dominant"]
            local_directions[ld] = local_directions.get(ld, 0) + 1
            commercial_directions[cd] = commercial_directions.get(cd, 0) + 1

        print(f"\nBias Direction Distribution:")
        print(f"  Local:       L={local_directions.get('L', 0)}, C={local_directions.get('C', 0)}, R={local_directions.get('R', 0)}")
        print(f"  Commercial:  L={commercial_directions.get('L', 0)}, C={commercial_directions.get('C', 0)}, R={commercial_directions.get('R', 0)}")

        # Save results if output file specified
        if output_file:
            report = {
                "timestamp": datetime.now().isoformat(),
                "service": "automatic_failover",
                "model_pairs_tried": [f"{s}:{m}" for s, m in ALL_MODEL_PAIRS],
                "summary": {
                    "total_sampled": len(articles),
                    "valid_comparisons": valid_count,
                    "errors": errors,
                    "matches": matches,
                    "match_rate": matches/valid_count if valid_count > 0 else 0,
                    "direction_matches": direction_matches,
                    "direction_match_rate": direction_matches/valid_count if valid_count > 0 else 0,
                    "degree_matches": degree_matches,
                    "degree_match_rate": degree_matches/valid_count if valid_count > 0 else 0,
                    "avg_dir_mae": round(avg_dir_mae, 3),
                    "avg_deg_mae": round(avg_deg_mae, 3),
                    "combined_mae": round((avg_dir_mae + avg_deg_mae)/2, 3),
                    "local_direction_distribution": local_directions,
                    "commercial_direction_distribution": commercial_directions,
                    "model_usage": model_success_count,
                },
                "results": results
            }
            
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {output_file}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare local MongoDB bias inference against commercial LLM services with failover"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of articles to sample (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use (bypasses failover, e.g., 'gemini-2.0-flash')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison results to JSON file",
    )
    args = parser.parse_args()

    try:
        comparator = CommercialBiasComparator(model=args.model)
        comparator.run_comparison(
            sample_size=args.sample,
            output_file=args.output
        )
        comparator.close()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
