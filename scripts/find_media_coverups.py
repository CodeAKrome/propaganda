#!/usr/bin/env python3
"""
Media Coverup Detection - Analyze MongoDB bias data to identify subjects with
extreme coverage bias indicating potential media coverups or suppression.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

try:
    import pymongo
except ImportError:
    print("Error: pymongo not installed. Run: pip install pymongo")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"

# Sensitive topics to check for coverage gaps
SENSITIVE_TOPICS = [
    ("Xinjiang/Uyighur", ["Uighur", "Xinjiang", "Uyghur", "XUAR"]),
    ("Tibet/Dalai Lama", ["Tibet", "Dalai Lama", "Lhasa"]),
    ("Falun Gong", ["Falun Gong", "Shen Yun"]),
    ("Tiananmen Square", ["Tiananmen", "1989 massacre"]),
    ("Rohingya", ["Rohingya", "Myanmar ethnic cleansing"]),
    ("Nigerian Christians", ["Nigeria Christian", "Christian persecution Nigeria"]),
    ("North Korea Prison Camps", ["Gulag", "labor camp", "political prison"]),
    ("Hong Kong Democracy", ["Hong Kong protests", "Hong Kong democracy"]),
    ("Venezuela Opposition", ["Venezuela opposition", "Venezuelan dissidents"]),
    ("Iran Protests", ["Iran protest", "Iran uprising"]),
    ("Myanmar Genocide", ["Myanmar genocide", "Rohingya"]),
]


class MediaCoverupDetector:
    def __init__(self, min_articles=50, imbalance_threshold=0.40, 
                 coverage_gap_threshold=100, entity_type="ALL"):
        self.min_articles = min_articles
        self.imbalance_threshold = imbalance_threshold
        self.coverage_gap_threshold = coverage_gap_threshold
        self.entity_type = entity_type.upper() if entity_type.upper() != "ALL" else None
        
        self.client = pymongo.MongoClient(MONGO_URI)
        self.coll = self.client[MONGO_DB][MONGO_COLL]
        
    def get_source_bias(self):
        """Get bias characteristics of each news source."""
        pipeline = [
            {"$match": {"bias": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": "$source",
                "count": {"$sum": 1},
                "avgL": {"$avg": "$bias.dir.L"},
                "avgC": {"$avg": "$bias.dir.C"},
                "avgR": {"$avg": "$bias.dir.R"}
            }}
        ]
        source_stats = {}
        for doc in self.coll.aggregate(pipeline):
            avg_bias = (doc.get("avgL", 0) or 0, doc.get("avgC", 0) or 0, doc.get("avgR", 0) or 0)
            # Determine direction: requires clear dominance (not just > R, but dominant relative to all)
            # Use max margin: how much larger is the dominant vs the second-largest
            max_val = max(avg_bias)
            max_idx = avg_bias.index(max_val)
            # Check if dominant is meaningfully larger than others (margin > 0.1)
            if max_val - sorted(avg_bias)[-2] > 0.1:
                if max_idx == 0:
                    direction = "LEFT"
                elif max_idx == 2:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"
            else:
                direction = "CENTER"  # No clear dominance
            source_stats[doc["_id"]] = {
                "count": doc["count"],
                "avgL": avg_bias[0],
                "avgC": avg_bias[1],
                "avgR": avg_bias[2],
                "direction": direction
            }
        return source_stats

    def analyze_bias_imbalance(self):
        """Identify subjects with extreme bias coverage imbalance."""
        print("\n=== EXTREME BIAS IMBALANCE (Potential Coverup) ===")
        print("Subject                   | Type   | Count | L    | C    | R    | Degree | Imbalance | Direction")
        print("-" * 100)
        
        entity_types = ["GPE", "PERSON", "ORG"] if not self.entity_type else [self.entity_type]
        
        results = []
        
        for entity_type in entity_types:
            pipeline = [
                {"$match": {
                    "bias": {"$exists": True, "$ne": None},
                    "ner.entities": {"$exists": True, "$ne": []}
                }},
                {"$unwind": "$ner.entities"},
                {"$match": {"ner.entities.label": entity_type}},
                {"$project": {
                    "entity": "$ner.entities.text",
                    "dir_L": "$bias.dir.L",
                    "dir_C": "$bias.dir.C",
                    "dir_R": "$bias.dir.R",
                    "deg_H": "$bias.deg.H"
                }},
            ]
            
            entity_stats = defaultdict(lambda: {"count": 0, "L": 0, "C": 0, "R": 0, "H": 0})
            
            for doc in self.coll.aggregate(pipeline):
                entity = doc.get("entity", "")
                if not entity or len(entity) < 2:
                    continue
                entity_stats[entity]["count"] += 1
                entity_stats[entity]["L"] += doc.get("dir_L", 0) or 0
                entity_stats[entity]["C"] += doc.get("dir_C", 0) or 0
                entity_stats[entity]["R"] += doc.get("dir_R", 0) or 0
                entity_stats[entity]["H"] += doc.get("deg_H", 0) or 0
            
            for entity, stats in entity_stats.items():
                if stats["count"] < self.min_articles:
                    continue
                avg_L = stats["L"] / stats["count"]
                avg_C = stats["C"] / stats["count"]
                avg_R = stats["R"] / stats["count"]
                avg_H = stats["H"] / stats["count"]
                imbalance = abs(avg_R - avg_L)
                
                if imbalance >= self.imbalance_threshold:
                    direction = "LEFT" if avg_L > avg_R else "RIGHT"
                    results.append({
                        "entity": entity[:28],
                        "type": entity_type[:8],
                        "count": stats["count"],
                        "avgL": avg_L,
                        "avgC": avg_C,
                        "avgR": avg_R,
                        "avgH": avg_H,
                        "imbalance": imbalance,
                        "direction": direction
                    })
        
        results.sort(key=lambda x: -x["imbalance"])
        
        for r in results[:20]:
            print(f"{r['entity']:<30} | {r['type']:<7} | {r['count']:>5} | {r['avgL']:.2f} | {r['avgC']:.2f} | {r['avgR']:.2f} | {r['avgH']:.2f}   | {r['imbalance']:.2f}      | {r['direction']}")
        
        return results

    def detect_coverage_gaps(self):
        """Identify topics with minimal or zero coverage."""
        print("\n=== COVERAGE GAPS (Potential Suppression) ===")
        print("Topic                          | Articles | Sources | Signal")
        print("-" * 70)
        
        source_stats = self.get_source_bias()
        
        for topic_name, keywords in SENSITIVE_TOPICS:
            query = {"$regex": "|".join(keywords), "$options": "i"}
            pipeline = [
                {"$match": {"article": query, "bias": {"$exists": True}}},
                {"$group": {"_id": "$source", "count": {"$sum": 1}}}
            ]
            
            sources = list(self.coll.aggregate(pipeline))
            total_articles = sum(s["count"] for s in sources)
            
            if total_articles == 0:
                signal = "🚨 ZERO COVERAGE"
            elif total_articles < 10:
                signal = "🚨 VERY LOW"
            elif total_articles < self.coverage_gap_threshold:
                signal = "⚠️ LOW COVERAGE"
            else:
                signal = f"✓ {total_articles} articles"
            
            print(f"{topic_name:<32} | {total_articles:>8} | {len(sources):>7} | {signal}")

    def analyze_one_sided_coverage(self):
        """Analyze topics that are only covered by one political side."""
        print("\n=== ONE-SIDED COVERAGE (Articles Only by One Side) ===")
        print("Topic                   | Left Sources | Right Sources | Dominant Side")
        print("-" * 70)
        
        source_stats = self.get_source_bias()
        
        # Classify sources as left or right
        left_sources = {s for s, info in source_stats.items() if info["direction"] == "LEFT"}
        right_sources = {s for s, info in source_stats.items() if info["direction"] == "RIGHT"}
        
        for topic_name, keywords in SENSITIVE_TOPICS[:10]:
            query = {"$regex": "|".join(keywords), "$options": "i"}
            pipeline = [
                {"$match": {"article": query, "bias": {"$exists": True}}},
                {"$group": {"_id": "$source", "count": {"$sum": 1}}}
            ]
            
            sources = list(self.coll.aggregate(pipeline))
            
            left_count = sum(s["count"] for s in sources if s["_id"] in left_sources)
            right_count = sum(s["count"] for s in sources if s["_id"] in right_sources)
            total = left_count + right_count
            
            if total == 0:
                continue
            elif left_count > 0 and right_count == 0:
                side = "LEFT ONLY ⚠️"
            elif right_count > 0 and left_count == 0:
                side = "RIGHT ONLY ⚠️"
            elif left_count / total > 0.9:
                side = "LEFT HEAVY ⚠️"
            elif right_count / total > 0.9:
                side = "RIGHT HEAVY ⚠️"
            else:
                side = "BALANCED"
            
            print(f"{topic_name:<24} | {left_count:>12} | {right_count:>14} | {side}")

    def analyze_source_concentration(self):
        """Identify topics covered by very few sources."""
        print("\n=== SOURCE CONCENTRATION ===")
        print("Topic                   | Total | Top3 %  | Signal")
        print("-" * 55)
        
        for topic_name, keywords in SENSITIVE_TOPICS[:8]:
            query = {"$regex": "|".join(keywords), "$options": "i"}
            pipeline = [
                {"$match": {"article": query, "bias": {"$exists": True}}},
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 3}
            ]
            
            sources = list(self.coll.aggregate(pipeline))
            
            # Get total count properly
            total_result = list(self.coll.aggregate([
                {"$match": {"article": query, "bias": {"$exists": True}}},
                {"$count": "total"}
            ]))
            total = total_result[0]["total"] if total_result else 0
            
            if total == 0:
                continue
                
            top3 = sum(s["count"] for s in sources)
            concentration = (top3 / total * 100) if total > 0 else 0
            
            if concentration > 70:
                signal = "⚠️ HIGH CONCENTRATION"
            elif concentration > 50:
                signal = "⚠️ MODERATE"
            else:
                signal = "✓ DIVERSE"
            
            print(f"{topic_name:<24} | {total:>5} | {concentration:>6.0f}%  | {signal}")

    def run_all(self):
        """Run all analyses."""
        total = self.coll.count_documents({})
        bias_count = self.coll.count_documents({"bias": {"$exists": True, "$ne": None}})
        
        print(f"MongoDB: {total:,} total articles, {bias_count:,} with bias data")
        
        self.analyze_bias_imbalance()
        self.detect_coverage_gaps()
        self.analyze_one_sided_coverage()
        self.analyze_source_concentration()
        
        print("\n" + "=" * 70)
        print("Analysis complete. Use --help to adjust thresholds.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MongoDB bias data to identify potential media coverups"
    )
    parser.add_argument("-n", "--min-articles", type=int, default=50,
                        help="Minimum articles for subject (default: 50)")
    parser.add_argument("-i", "--imbalance-threshold", type=float, default=0.40,
                        help="Bias imbalance threshold (default: 0.40)")
    parser.add_argument("-c", "--coverage-gap-threshold", type=int, default=100,
                        help="Coverage gap threshold (default: 100)")
    parser.add_argument("-e", "--entity-type", default="ALL",
                        help="Entity type: GPE, PERSON, ORG, ALL (default: ALL)")
    parser.add_argument("-o", "--output", default="console",
                        help="Output format: console, csv, json (default: console)")
    parser.add_argument("--output-csv", default="output/media_coverups.csv",
                        help="CSV output path")
    parser.add_argument("--output-json", default="output/media_coverups.json",
                        help="JSON output path")
    
    args = parser.parse_args()
    
    detector = MediaCoverupDetector(
        min_articles=args.min_articles,
        imbalance_threshold=args.imbalance_threshold,
        coverage_gap_threshold=args.coverage_gap_threshold,
        entity_type=args.entity_type
    )
    
    detector.run_all()


if __name__ == "__main__":
    main()