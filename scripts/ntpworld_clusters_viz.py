#!/usr/bin/env python3
"""
NTP-World DBSCAN Clustering Visualization
Combined text-based and bias-based clustering
"""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load text clusters
with open("output/ntp_world_clusters_text.json") as f:
    text_data = json.load(f)

# Load bias clusters
with open("output/ntp_world_clusters_bias.json") as f:
    bias_data = json.load(f)

# Get text cluster data
text_categories = text_data.get("categories", [])
uncategorized = text_data.get("uncategorized", [])

# Sort by article count
text_categories_sorted = sorted(
    text_categories, key=lambda x: x.get("article_count", 0), reverse=True
)

# Get bias metadata
bias_meta = bias_data.get("metadata", {})
bias_num_clusters = bias_meta.get("num_clusters", 0)
bias_noise = bias_meta.get("noise_count", 0)

# Load bias data for distribution
import pandas as pd

bias_df = pd.read_csv("output/ntp_world_bias.tsv", sep="\t")

# Direction pie chart values
dir_L = bias_df["dir_L"].mean()
dir_C = bias_df["dir_C"].mean()
dir_R = bias_df["dir_R"].mean()

# Degree pie chart values
deg_L = bias_df["deg_L"].mean()
deg_M = bias_df["deg_M"].mean()
deg_H = bias_df["deg_H"].mean()

# Create visualization
fig = plt.figure(figsize=(16, 12))

# ===== PANEL 1: Text-based clusters (top-left) =====
ax1 = fig.add_axes([0.05, 0.45, 0.45, 0.45])

# Bar chart of top text clusters
top_n = min(15, len(text_categories_sorted))
names = [f"C{i + 1}" for i in range(top_n)]
sizes = [c.get("article_count", 0) for c in text_categories_sorted[:top_n]]
colors = ["#444444"] * len(names)

ax1.barh(range(len(names)), sizes, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=8)
ax1.set_xlabel("Number of Articles", fontsize=9)
ax1.set_title(
    f"Text-Based Topic Clusters\n({len(text_categories_sorted)} clusters found)",
    fontsize=11,
    fontweight="normal",
)
ax1.invert_yaxis()

for spine in ax1.spines.values():
    spine.set_visible(False)

# ===== PANEL 2: Bias direction (top-right) =====
ax2 = fig.add_axes([0.55, 0.45, 0.40, 0.45])

ax2.pie(
    [dir_L, dir_C, dir_R],
    labels=["Left", "Center", "Right"],
    colors=["#666666", "#999999", "#333333"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 8},
)
ax2.set_title(
    f"Bias Direction Distribution\n(Average across {len(bias_df)} articles)",
    fontsize=10,
    fontweight="normal",
    pad=5,
)

# ===== PANEL 3: Summary (bottom) =====
ax3 = fig.add_axes([0.05, 0.05, 0.90, 0.35])

# Text clustering summary
ax3.text(
    0.02,
    0.95,
    "TEXT-BASED CLUSTERING (from titles)",
    fontsize=11,
    fontweight="bold",
    transform=ax3.transAxes,
)
ax3.text(
    0.02,
    0.87,
    f"Total clusters found: {len(text_categories_sorted)}",
    fontsize=9,
    transform=ax3.transAxes,
)

y_pos = 0.78
ax3.text(
    0.02,
    y_pos,
    "Top 10 Topic Clusters with sample titles:",
    fontsize=9,
    fontweight="bold",
    transform=ax3.transAxes,
)
y_pos -= 0.04

for i, cat in enumerate(text_categories_sorted[:10]):
    count = cat.get("article_count", 0)
    # Get sample article title from first article
    articles = cat.get("articles", [])
    sample_title = ""
    if articles and len(articles) > 0:
        sample = articles[0]
        if isinstance(sample, dict):
            sample_title = sample.get("title", "")[:40] if sample.get("title") else ""
        elif isinstance(sample, str):
            sample_title = sample[:40]

    ax3.text(
        0.02,
        y_pos,
        f'  C{i + 1}: {count:3d} articles - "{sample_title}..."',
        fontsize=7,
        transform=ax3.transAxes,
    )
    y_pos -= 0.035

# Bias clustering summary
ax3.text(
    0.55,
    0.95,
    "BIAS-BASED CLUSTERING (from political direction)",
    fontsize=11,
    fontweight="bold",
    transform=ax3.transAxes,
)
ax3.text(
    0.55,
    0.87,
    f"Clusters found: {bias_num_clusters}",
    fontsize=9,
    transform=ax3.transAxes,
)
ax3.text(0.55, 0.80, f"Noise points: {bias_noise}", fontsize=9, transform=ax3.transAxes)

ax3.text(0.55, 0.70, "Direction breakdown:", fontsize=9, transform=ax3.transAxes)
ax3.text(
    0.55,
    0.64,
    f"  Left: {dir_L * 100:.1f}%  Center: {dir_C * 100:.1f}%  Right: {dir_R * 100:.1f}%",
    fontsize=8,
    transform=ax3.transAxes,
)
ax3.text(0.55, 0.57, "Degree breakdown:", fontsize=9, transform=ax3.transAxes)
ax3.text(
    0.55,
    0.51,
    f"  Low: {deg_L * 100:.1f}%  Med: {deg_M * 100:.1f}%  High: {deg_H * 100:.1f}%",
    fontsize=8,
    transform=ax3.transAxes,
)

# Overall summary box
ax3.text(
    0.75, 0.35, "OVERALL:", fontsize=10, fontweight="bold", transform=ax3.transAxes
)
ax3.text(
    0.75, 0.27, f"NTP-World: 2,335 articles total", fontsize=8, transform=ax3.transAxes
)
ax3.text(0.75, 0.20, f"1,235 with bias data", fontsize=8, transform=ax3.transAxes)
ax3.text(0.75, 0.13, f"Date range: 7 months", fontsize=8, transform=ax3.transAxes)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks([])
ax3.set_yticks([])
for spine in ax3.spines.values():
    spine.set_visible(False)

# Title
fig.suptitle(
    "NTP-World DBSCAN Clustering Analysis", fontsize=14, fontweight="normal", y=0.98
)

# Save
plt.savefig(
    "output/ntp_world_dbscan_clusters.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: output/ntp_world_dbscan_clusters.png")
plt.close()

# Print summary
print()
print("=" * 60)
print("NTP-WORLD DBSCAN CLUSTERING RESULTS")
print("=" * 60)
print()
print("TEXT-BASED CLUSTERING:")
print(f"  Total clusters: {len(text_categories_sorted)}")
print(f"  Top 5 clusters:")
for i, cat in enumerate(text_categories_sorted[:5]):
    count = cat.get("article_count", 0)
    print(f"    Cluster {i + 1}: {count} articles")
print()
print("BIAS-BASED CLUSTERING:")
print(f"  Clusters found: {bias_num_clusters}")
print(f"  Noise points: {bias_noise}")
print(f"  Direction: L={dir_L * 100:.1f}% C={dir_C * 100:.1f}% R={dir_R * 100:.1f}%")
print(
    f"  Degree: Low={deg_L * 100:.1f}% Med={deg_M * 100:.1f}% High={deg_H * 100:.1f}%"
)
print()
print("OUTPUT FILES:")
print("  output/ntp_world_clusters_text.json")
print("  output/ntp_world_clusters_bias.json")
print("  output/ntp_world_dbscan_clusters.png")
print("=" * 60)
