#!/usr/bin/env python3
"""
NTP-World Bias Analysis & Visualization

Optimizations:
- Aggregated pipeline instead of multiple queries
- Connection pooling
- Caching for repeated analysis
- Time-range filtering
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import lru_cache

import pymongo
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_mongo_client(max_pool_size: int = 50) -> pymongo.MongoClient:
    """Create MongoDB client with connection pooling."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
    return pymongo.MongoClient(
        mongo_uri,
        maxPoolSize=max_pool_size,
        serverSelectionTimeoutMS=5000,
    )


client = create_mongo_client()
coll = client["rssnews"]["articles"]


@lru_cache(maxsize=4)
def get_bias_stats_by_source(
    source: str = "ntp-world",
    min_articles: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get bias statistics for a source using a single aggregated pipeline.
    Cached to avoid repeated queries.
    """
    # Build date filter
    date_filter: Dict[str, Any] = {}
    if start_date or end_date:
        date_filter["published"] = {}
        if start_date:
            date_filter["published"]["$gte"] = start_date
        if end_date:
            date_filter["published"]["$lte"] = end_date

    # Base match
    match_stage = {
        "source": source,
        "bias.dir.L": {"$exists": True},
    }
    if date_filter:
        match_stage.update(date_filter)

    # Single aggregation pipeline
    pipeline = [
        {"$match": match_stage},
        {
            "$group": {
                "_id": "$source",
                "count": {"$sum": 1},
                "avg_L": {"$avg": "$bias.dir.L"},
                "avg_C": {"$avg": "$bias.dir.C"},
                "avg_R": {"$avg": "$bias.dir.R"},
                "avg_deg_L": {"$avg": "$bias.deg.L"},
                "avg_deg_M": {"$avg": "$bias.deg.M"},
                "avg_deg_H": {"$avg": "$bias.deg.H"},
                "min_date": {"$min": "$published"},
                "max_date": {"$max": "$published"},
            }
        },
    ]

    results = list(coll.aggregate(pipeline))
    return results[0] if results else None


def get_all_sources_bias_summary(min_articles: int = 100) -> list:
    """
    Get bias summary for all sources in a single query.
    Replaces multiple individual count_documents calls.
    """
    pipeline = [
        {"$match": {"bias.dir.L": {"$exists": True}}},
        {
            "$group": {
                "_id": "$source",
                "count": {"$sum": 1},
                "avg_L": {"$avg": "$bias.dir.L"},
                "avg_C": {"$avg": "$bias.dir.C"},
                "avg_R": {"$avg": "$bias.dir.R"},
                "avg_deg_L": {"$avg": "$bias.deg.L"},
                "avg_deg_M": {"$avg": "$bias.deg.M"},
                "avg_deg_H": {"$avg": "$bias.deg.H"},
            }
        },
        {"$match": {"count": {"$gte": min_articles}}},
        {"$sort": {"count": -1}},
        {"$limit": 50},
    ]
    return list(coll.aggregate(pipeline))


def main(source: str = "ntp-world", output_prefix: str = "output/ntp_world"):
    """
    Main analysis function with aggregated queries.
    """
    # Get stats using cached function
    stats = get_bias_stats_by_source(source)

    if not stats:
        print(f"No bias data found for source: {source}")
        return

    # Calculate scores
    dir_score = stats["avg_R"] - stats["avg_L"]
    deg_score = (
        stats["avg_deg_H"] * 1.0 + stats["avg_deg_M"] * 0.5 + stats["avg_deg_L"] * 0.0
    )

    # Date range for display
    date_range = "unknown"
    if stats.get("min_date") and stats.get("max_date"):
        try:
            min_dt = datetime.fromisoformat(stats["min_date"].replace("Z", "+00:00"))
            max_dt = datetime.fromisoformat(stats["max_date"].replace("Z", "+00:00"))
            days = (max_dt - min_dt).days
            date_range = (
                f"{days} days ({stats['min_date'][:10]} to {stats['max_date'][:10]})"
            )
        except:
            pass


def create_visualization(
    stats: Dict[str, Any],
    dir_score: float,
    deg_score: float,
    output_prefix: str = "output/ntp_world",
    date_range: str = "unknown",
) -> None:
    """Create bias visualization from stats."""
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bias Direction (Tufte-style dot)
    ax = axes[0]

    # Center line
    ax.axvline(0, color="#333333", linewidth=1, linestyle="-")

    # Direction bar
    bar_width = stats["avg_R"] - stats["avg_L"]
    left_edge = -stats["avg_L"]
    ax.barh(
        0.5,
        bar_width,
        left=left_edge,
        height=0.4,
        color="#888888",
        edgecolor="black",
        linewidth=0.5,
    )

    # Labels
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["← Left", "-0.2", "Center", "+0.2", "Right →"])
    ax.set_yticks([])
    ax.set_xlabel("Political Direction", fontsize=11)
    ax.set_title(
        f"NTP-WORLD Bias Direction\n(Avg: L={stats['avg_L']:.2f} C={stats['avg_C']:.2f} R={stats['avg_R']:.2f})",
        fontsize=12,
        fontweight="normal",
        pad=10,
    )

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Right: Degree bar
    ax2 = axes[1]
    ax2.barh(
        0.5,
        deg_score,
        left=0,
        height=0.4,
        color="#666666",
        edgecolor="black",
        linewidth=0.5,
    )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "Low", "Medium", "High", "1.0"])
    ax2.set_yticks([])
    ax2.set_xlabel("Bias Intensity (Degree)", fontsize=11)
    ax2.set_title(
        f"NTP-WORLD Bias Intensity\n(Score: {deg_score:.2f})",
        fontsize=12,
        fontweight="normal",
        pad=10,
    )

    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Stats annotation
    fig.text(
        0.5,
        0.02,
        f"Articles analyzed: {stats['count']:,} | Direction Score: {dir_score:+.3f} | Degree Score: {deg_score:.3f} | {date_range}",
        fontsize=9,
        ha="center",
        style="italic",
        color="#666666",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(
        f"{output_prefix}_bias.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_prefix}_bias.png")
    plt.close()


def print_summary(
    stats: Dict[str, Any],
    dir_score: float,
    deg_score: float,
    date_range: str = "unknown",
) -> None:
    """Print analysis summary."""
    print()
    print("=" * 60)
    print("NTP-WORLD BIAS ANALYSIS SUMMARY")
    print("=" * 60)
    print()
    print(f"Articles analyzed: {stats['count']:,}")
    print(f"Date range: {date_range}")
    print()
    print("DIRECTION:")
    print(f"  Left:   {stats['avg_L'] * 100:.1f}%")
    print(f"  Center: {stats['avg_C'] * 100:.1f}%")
    print(f"  Right:  {stats['avg_R'] * 100:.1f}%")
    print(f"  Score:  {dir_score:+.3f}")
    if dir_score < -0.1:
        print("  → Slight Center-Left lean")
    elif dir_score > 0.1:
        print("  → Slight Center-Right lean")
    else:
        print("  → Nearly Center / Neutral")
    print()
    print("INTENSITY:")
    print(f"  Low:    {stats['avg_deg_L'] * 100:.1f}%")
    print(f"  Medium: {stats['avg_deg_M'] * 100:.1f}%")
    print(f"  High:   {stats['avg_deg_H'] * 100:.1f}%")
    print(f"  Score:  {deg_score:.3f}")
    if deg_score < 0.3:
        print("  → Low bias intensity (factual reporting)")
    elif deg_score < 0.6:
        print("  → Moderate bias intensity")
    else:
        print("  → High bias intensity")
    print()
    print("INTERPRETATION:")
    if abs(dir_score) < 0.1:
        print("NTP-World shows nearly neutral political direction")
        print(f"(score: {dir_score:+.3f}, essentially center). The coverage is")
        print("primarily factual international news with moderate")
        print("presentation. No strong ideological leaning detected.")
    elif dir_score > 0:
        print(f"NTP-World shows Center-Right lean (score: {dir_score:+.3f}).")
    else:
        print(f"NTP-World shows Center-Left lean (score: {dir_score:+.3f}).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTP-World Bias Analysis")
    parser.add_argument("--source", default="ntp-world", help="Source to analyze")
    parser.add_argument("--output", default="output/ntp_world", help="Output prefix")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Get stats with date filtering
    stats = get_bias_stats_by_source(
        args.source, start_date=args.start_date, end_date=args.end_date
    )

    if not stats:
        print(f"No bias data found for source: {args.source}")
        exit(1)

    # Calculate scores
    dir_score = stats["avg_R"] - stats["avg_L"]
    deg_score = (
        stats["avg_deg_H"] * 1.0 + stats["avg_deg_M"] * 0.5 + stats["avg_deg_L"] * 0.0
    )

    # Date range
    date_range = "unknown"
    if stats.get("min_date") and stats.get("max_date"):
        try:
            min_dt = datetime.fromisoformat(stats["min_date"].replace("Z", "+00:00"))
            max_dt = datetime.fromisoformat(stats["max_date"].replace("Z", "+00:00"))
            days = (max_dt - min_dt).days
            date_range = (
                f"{days} days ({stats['min_date'][:10]} to {stats['max_date'][:10]})"
            )
        except:
            date_range = f"{stats['min_date'][:10]} to {stats['max_date'][:10]}"

    # Generate visualization and summary
    create_visualization(stats, dir_score, deg_score, args.output, date_range)
    print_summary(stats, dir_score, deg_score, date_range)
