#!/usr/bin/env python3
"""
Create comprehensive bias visualizations for all sources.
Uses existing bias data - no new analysis.
"""

import pymongo
import math
import os
import json
from collections import defaultdict

matplotlib = None
plt = None


def init_matplotlib():
    global matplotlib, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def get_all_sources_bias():
    """Get bias stats for all sources."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
    client = pymongo.MongoClient(
        mongo_uri, maxPoolSize=50
    )
    coll = client["rssnews"]["articles"]

    pipeline = [
        {"$match": {"bias.dir.L": {"$exists": True}}},
        {
            "$project": {
                "source": 1,
                "dir_L": "$bias.dir.L",
                "dir_C": "$bias.dir.C",
                "dir_R": "$bias.dir.R",
                "deg_L": "$bias.deg.L",
                "deg_M": "$bias.deg.M",
                "deg_H": "$bias.deg.H",
            }
        },
        {
            "$group": {
                "_id": "$source",
                "count": {"$sum": 1},
                "avg_L": {"$avg": "$dir_L"},
                "avg_C": {"$avg": "$dir_C"},
                "avg_R": {"$avg": "$dir_R"},
                "avg_deg_L": {"$avg": "$deg_L"},
                "avg_deg_M": {"$avg": "$deg_M"},
                "avg_deg_H": {"$avg": "$deg_H"},
            }
        },
    ]
    return list(coll.aggregate(pipeline))


def create_comparison_chart(sources_data, output_dir):
    """Create a comprehensive comparison chart of all sources."""
    matplotlib, plt = init_matplotlib()

    # Sort by direction score
    for s in sources_data:
        s["dir_score"] = s["avg_R"] - s["avg_L"]
        s["deg_score"] = s["avg_deg_H"] * 1.0 + s["avg_deg_M"] * 0.5

    sources_data.sort(key=lambda x: x["dir_score"])

    # Take top 40 by count for readability
    top_sources = sorted(sources_data, key=lambda x: x["count"], reverse=True)[:40]
    top_sources.sort(key=lambda x: x["dir_score"])

    fig, ax = plt.subplots(figsize=(16, 14))

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Vertical guides
    for x in [-0.5, 0, 0.5]:
        ax.axvline(x, color="#dddddd", linewidth=0.75, zorder=0)

    names = [s["_id"] for s in top_sources]
    dirs = [s["dir_score"] for s in top_sources]
    counts = [s["count"] for s in top_sources]
    degs = [s["deg_score"] for s in top_sources]
    y_pos = list(range(len(names)))

    max_cnt = max(counts)
    areas = [800 * (c / max_cnt) for c in counts]

    # Color by direction: blue (left) -> gray (center) -> red (right)
    for i, (x, y, area, deg) in enumerate(zip(dirs, y_pos, areas, degs)):
        radius = math.sqrt(area) / 35
        # Color based on direction
        if x < -0.2:
            color = "#4169e1"  # Royal blue for left
        elif x > 0.2:
            color = "#dc143c"  # Crimson for right
        else:
            color = "#808080"  # Gray for center

        circle = plt.Circle(
            (x, y), radius, facecolor=color, edgecolor="#222", linewidth=0.5, alpha=0.7
        )
        ax.add_patch(circle)

    # Labels
    for i, (name, cnt) in enumerate(zip(names, counts)):
        ax.text(
            1.08, i, name, fontsize=6, va="center", ha="left", fontfamily="monospace"
        )
        ax.text(1.35, i, f"({cnt:,})", fontsize=5, va="center", ha="left", color="#666")

    ax.set_xlim(-1.15, 1.5)
    ax.set_ylim(-1, len(names))
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["← Left", "", "Center", "", "Right →"], fontsize=10)
    ax.set_xlabel("Political Direction", fontsize=11, labelpad=12)
    ax.set_yticks([])
    ax.set_title(
        "Media Bias Comparison - All Sources\n(Dot size = article count, Color = direction)",
        fontsize=14,
        pad=20,
    )

    # Legend
    ax.text(-1.0, len(names) + 0.5, "Blue = Left", fontsize=8, color="#4169e1")
    ax.text(-0.6, len(names) + 0.5, "Gray = Center", fontsize=8, color="#808080")
    ax.text(-0.1, len(names) + 0.5, "Red = Right", fontsize=8, color="#dc143c")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/all_sources_comparison.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_dir}/all_sources_comparison.png")
    plt.close()


def create_source_detail(source_name, source_data, output_dir):
    """Create detailed visualization for a single source."""
    matplotlib, plt = init_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Direction
    ax = axes[0]
    avg_L = source_data["avg_L"]
    avg_C = source_data["avg_C"]
    avg_R = source_data["avg_R"]
    dir_score = source_data["dir_score"]

    ax.axvline(0, color="#333", linewidth=1)
    bar_width = avg_R - avg_L
    left_edge = -avg_L
    ax.barh(
        0.5,
        bar_width,
        left=left_edge,
        height=0.4,
        color="#888",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["← Left", "-0.2", "Center", "+0.2", "Right →"])
    ax.set_yticks([])
    ax.set_xlabel("Political Direction")
    ax.set_title(
        f"{source_name}\nL={avg_L:.2f} C={avg_C:.2f} R={avg_R:.2f}\nScore: {dir_score:+.3f}"
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Degree
    ax2 = axes[1]
    deg_score = source_data["deg_score"]
    ax2.barh(
        0.5,
        deg_score,
        left=0,
        height=0.4,
        color="#666",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "Low", "Medium", "High", "1.0"])
    ax2.set_yticks([])
    ax2.set_xlabel("Bias Intensity")
    ax2.set_title(f"Intensity: {deg_score:.3f}")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5,
        0.02,
        f"Articles: {source_data['count']:,}",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    safe_name = source_name.replace("/", "_")
    plt.savefig(
        f"{output_dir}/source_{safe_name}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def create_category_summary(sources_data, output_dir):
    """Create summary by category (left/center/right)."""
    matplotlib, plt = init_matplotlib()

    left_sources = []
    center_sources = []
    right_sources = []

    for s in sources_data:
        if s["dir_score"] < -0.1:
            left_sources.append(s)
        elif s["dir_score"] > 0.1:
            right_sources.append(s)
        else:
            center_sources.append(s)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for ax, sources, title, color in [
        (axes[0], left_sources, "LEFT-LEANING SOURCES", "#4169e1"),
        (axes[1], center_sources, "CENTER SOURCES", "#808080"),
        (axes[2], right_sources, "RIGHT-LEANING SOURCES", "#dc143c"),
    ]:
        sources.sort(key=lambda x: x["count"], reverse=True)
        names = [s["_id"][:20] for s in sources[:15]]
        counts = [s["count"] for s in sources[:15]]

        y_pos = list(range(len(names)))
        ax.barh(y_pos, counts, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Article Count")
        ax.set_title(f"{title}\n({len(sources)} sources)", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/sources_by_category.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_dir}/sources_by_category.png")
    plt.close()

    return left_sources, center_sources, right_sources


def create_direction_histogram(sources_data, output_dir):
    """Create histogram of direction scores."""
    matplotlib, plt = init_matplotlib()

    dir_scores = [s["dir_score"] for s in sources_data]
    counts = [s["count"] for s in sources_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Weighted histogram
    bins = [-1, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 1]
    hist = defaultdict(lambda: {"weighted_count": 0, "sources": 0})

    for score, cnt in zip(dir_scores, counts):
        bin_idx = min(range(len(bins)), key=lambda i: abs(bins[i] - score))
        hist[bin_idx]["weighted_count"] += cnt
        hist[bin_idx]["sources"] += 1

    bin_labels = [
        "<-0.6\nHard Left",
        "-0.6 to -0.4\nLeft",
        "-0.4 to -0.2\nLean Left",
        "-0.2 to 0\nSlight Left",
        "0 to 0.2\nCenter",
        "0.2 to 0.4\nSlight Right",
        "0.4 to 0.6\nLean Right",
        ">0.6\nHard Right",
    ]

    x_vals = list(range(len(bins) - 1))
    heights = [hist[i]["weighted_count"] for i in x_vals]
    colors = [
        "#4169e1",
        "#6a8efc",
        "#9bb5fe",
        "#c9d6ff",
        "#808080",
        "#ffcccc",
        "#fe6b6b",
        "#dc143c",
    ]

    ax.bar(x_vals, heights, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Total Articles")
    ax.set_title("Distribution of Political Bias Across All News Sources", fontsize=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/direction_histogram.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_dir}/direction_histogram.png")
    plt.close()


def create_intensity_analysis(sources_data, output_dir):
    """Create intensity vs direction scatter."""
    matplotlib, plt = init_matplotlib()

    dirs = [s["dir_score"] for s in sources_data]
    degs = [s["deg_score"] for s in sources_data]
    counts = [s["count"] for s in sources_data]
    names = [s["_id"] for s in sources_data]

    fig, ax = plt.subplots(figsize=(14, 10))

    sizes = [math.sqrt(c) * 2 for c in counts]

    # Color by direction
    colors = []
    for d in dirs:
        if d < -0.2:
            colors.append("#4169e1")
        elif d > 0.2:
            colors.append("#dc143c")
        else:
            colors.append("#808080")

    ax.scatter(
        dirs, degs, s=sizes, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5
    )

    # Labels for notable sources
    for i, name in enumerate(names):
        if counts[i] > 1500 or abs(dirs[i]) > 0.3:
            ax.annotate(name[:12], (dirs[i], degs[i]), fontsize=6, alpha=0.8)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Political Direction (Left ← → Right)")
    ax.set_ylabel("Bias Intensity (Low → High)")
    ax.set_title("Direction vs Intensity\n(Larger dots = more articles)", fontsize=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/direction_vs_intensity.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {output_dir}/direction_vs_intensity.png")
    plt.close()


def create_summary_json(sources_data, output_dir):
    """Create JSON summary of all sources."""
    summary = {
        "total_sources": len(sources_data),
        "total_articles": sum(s["count"] for s in sources_data),
        "sources": sorted(sources_data, key=lambda x: x["count"], reverse=True),
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir}/summary.json")


def main():
    output_dir = "visualization"
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching bias data for all sources...")
    sources_data = get_all_sources_bias()
    print(f"Found {len(sources_data)} sources with bias data")

    # Calculate scores
    for s in sources_data:
        s["dir_score"] = s["avg_R"] - s["avg_L"]
        s["deg_score"] = s["avg_deg_H"] * 1.0 + s["avg_deg_M"] * 0.5

    print("\nCreating visualizations...")

    create_comparison_chart(sources_data, output_dir)
    left, center, right = create_category_summary(sources_data, output_dir)
    create_direction_histogram(sources_data, output_dir)
    create_intensity_analysis(sources_data, output_dir)
    create_summary_json(sources_data, output_dir)

    # Create individual source charts for top 20 by count
    top_20 = sorted(sources_data, key=lambda x: x["count"], reverse=True)[:20]
    print(f"\nCreating individual charts for top 20 sources...")
    for s in top_20:
        create_source_detail(s["_id"], s, output_dir)

    print(f"\n{'=' * 60}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Total sources: {len(sources_data)}")
    print(f"Left-leaning: {len(left)}")
    print(f"Center: {len(center)}")
    print(f"Right-leaning: {len(right)}")
    print(f"\nFiles created:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
