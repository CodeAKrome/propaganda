#!/usr/bin/env python3
"""Process Iran/Hormuz articles - improved LLM extraction."""

import os
import re
import subprocess
from collections import Counter

# Read articles
with open("output/iran_hormuz_7day.txt") as f:
    content = f.read()

# Split by article separator
# NOTE: Limiting to 10 articles for batch processing. Adjust as needed.
articles = content.split("\n---\n")[:10]

# Build prompt with actual article content
prompt = """You are an expert at extracting Subject-Verb-Object relationships from news articles.

Extract ALL relationships from these Iran/Hormuz news headlines. 

Format each relationship as:
SUBJ, VERB, OBJ

For example:
Iran, threatened, US
Trump, sent, delegation
US, blocked, Hormuz

Extract every relationship you can find:

"""

for i, art in enumerate(articles):
    for line in art.split("\n"):
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()
            prompt += f"{title}\n"
            break

# Call LLM
print("Calling gpt-oss:120b for SVO extraction...")
result = subprocess.run(
    ["ollama", "run", "--hidethinking", "gpt-oss:120b"],
    input=prompt,
    capture_output=True,
    text=True,
    timeout=300,
)
llm_output = result.stdout
print(f"Response: {llm_output[:500]}...")

# Parse tuples - look for lines with commas
svo_tuples = []
for line in llm_output.split("\n"):
    line = line.strip()
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            subj = parts[0]
            verb = parts[1]
            obj = ",".join(parts[2:]).strip()  # Handle commas in object

            # Filter noise
            if len(subj) > 2 and len(obj) > 2 and len(subj) < 30:
                svo_tuples.append((subj, verb, obj))

print(f"\nExtracted {len(svo_tuples)} tuples")

# Count
edge_cnt = Counter(svo_tuples)
print("\nTop relationships:")
for (s, v, o), c in edge_cnt.most_common(25):
    print(f"  {s} -> {o} ({v}): {c}")

# Nodes
nodes = set()
for s, v, o in svo_tuples:
    nodes.add(s)
    nodes.add(o)

print(f"\nNodes: {nodes}")

# DOT
dot = "output/iran_hormuz_7day.dot"
with open(dot, "w") as f:
    f.write("digraph IranHormuz7day {\n")
    f.write("  rankdir=LR;\n")
    f.write("  node [shape=box, style=filled, fontsize=11];\n")
    f.write("  edge [color=gray, fontsize=9];\n\n")

    countries = {
        "US",
        "Iran",
        "China",
        "Russia",
        "UK",
        "France",
        "Pakistan",
        "Israel",
        "Europe",
        "Trump",
    }
    for n in sorted(nodes):
        fill = "lightcoral" if n in countries else "lightblue"
        safe = n.replace('"', "")[:25]
        f.write(f'  "{safe}" [label="{n}", fillcolor="{fill}"];\n')

    f.write("\n")

    for (s, v, o), c in sorted(edge_cnt.items(), key=lambda x: -x[1])[:40]:
        safe_s = s.replace('"', "")[:25]
        safe_o = o.replace('"', "")[:25]
        safe_v = v.replace('"', "")[:15]
        w = min(c * 0.5, 3)
        f.write(f'  "{safe_s}" -> "{safe_o}" [label="{safe_v}", penwidth={w}];\n')

    f.write("}\n")

print(f"\nCreated: {dot}")

# PNG
png = dot.replace(".dot", ".png")
subprocess.run(["dot", "-Tpng", dot, "-o", png], check=False, capture_output=True)
if os.path.exists(png):
    print(f"PNG: {png} ({os.path.getsize(png)} bytes)")
