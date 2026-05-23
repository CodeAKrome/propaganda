#!/usr/bin/env python3
"""
Media Bias & Manipulation Visualization
Tufte Principles: Maximize data-ink ratio,Remove chartjunk
"""

import pymongo
import math
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
client = pymongo.MongoClient(MONGO_URI, maxPoolSize=50)
coll = client["rssnews"]["articles"]

WEIGHTS = {"propaganda": 3, "disinfo": 3, "distort": 2, "misleading": 2}


def main():
    pipeline = [
        {"$match": {"bias": {"$exists": True}, "ner": {"$exists": True}}},
        {
            "$project": {
                "source": 1,
                "dL": "$bias.dir.L",
                "dC": "$bias.dir.C",
                "dR": "$bias.dir.R",
            }
        },
        {"$group": {"_id": "$source", "count": {"$sum": 1}, "avgL": {"$avg": "$dL"}, "avgR": {"$avg": "$dR"}}},
        {"$match": {"count": {"$gte": 100}}},
        {"$sort": {"count": -1}},
        {"$limit": 50},
    ]
    results = list(coll.aggregate(pipeline))
    for r in results:
        r["dir_score"] = r.get("avgR", 0) - r.get("avgL", 0)

    top_30 = sorted(results, key=lambda x: x["count"], reverse=True)[:30]

    fig, ax = plt.subplots(figsize=(14, 11))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color="#ddd", linewidth=0.75)

    names = [r["_id"] for r in top_30]
    dirs = [r["dir_score"] for r in top_30]
    counts = [r["count"] for r in top_30]
    y_pos = list(range(len(names)))
    max_cnt = max(counts)
    areas = [600 * (c / max_cnt) for c in counts]

    for i, (x, y, area) in enumerate(zip(dirs, y_pos, areas)):
        radius = math.sqrt(area) / 35
        circle = plt.Circle(
            (x, y), radius, facecolor="#666", edgecolor="#333", linewidth=0.5
        )
        ax.add_patch(circle)
        ax.text(1.08, i, names[i], fontsize=7, fontfamily="monospace")

    ax.set_xlim(-1.15, 1.5)
    ax.set_ylim(-1, len(names))
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["Left", "", "Center", "", "Right"])
    ax.set_xlabel("Political Direction")
    ax.set_title("Media Bias Dot Plot")
    plt.tight_layout()
    plt.savefig("output/bias_plotA_dotplot.png", dpi=300, bbox_inches="tight")
    print("Saved: output/bias_plotA_dotplot.png")
    plt.close()


if __name__ == "__main__":
    main()
