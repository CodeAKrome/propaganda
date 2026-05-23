#!/usr/bin/env python3
"""Iran/Hormuz news extraction - fixed."""

import pymongo
from datetime import datetime, timedelta
import os
import subprocess
from collections import defaultdict, Counter

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")


def extract_articles(limit=500, output_file="output/iran_hormuz.txt"):
    """Extract Iran/Hormuz articles."""
    client = pymongo.MongoClient(MONGO_URI)
    coll = client["rssnews"]["articles"]

    print(f"Extracting {limit} Iran/Hormuz articles...")

    query = {
        "$or": [
            {"title": {"$regex": "Iran", "$options": "i"}},
            {"title": {"$regex": "Hormuz", "$options": "i"}},
        ]
    }

    count = 0
    with open(output_file, "w") as f:
        for doc in coll.find(query).limit(limit):
            title = doc.get("title", "")[:200]
            source = doc.get("source", "")
            _id = str(doc["_id"])

            f.write(f"ID:{_id}\n")
            f.write(f"SOURCE:{source}\n")
            f.write(f"TITLE:{title}\n")
            f.write("\n")
            count += 1

    print(f"Extracted {count} articles")
    return count


def parse_articles(filename):
    """Parse to list of titles."""
    articles = []
    with open(filename) as f:
        for line in f:
            if line.startswith("TITLE:"):
                articles.append({"title": line[6:].strip()})
    return articles


def extract_svo(articles):
    """Extract SVO from titles."""

    pairs = [
        ("Iran", "US"),
        ("Iran", "Trump"),
        ("Iran", "Pakistan"),
        ("Iran", "Israel"),
        ("Iran", "China"),
        ("Iran", "Russia"),
        ("Iran", "UK"),
        ("Iran", "France"),
        ("Iran", "Europe"),
        ("Strait of Hormuz", "Iran"),
        ("Strait of Hormuz", "US"),
        ("Strait of Hormuz", "China"),
        ("Trump", "Iran"),
        ("Trump", "Pakistan"),
    ]

    # Verbs ordered by specificity (most specific first)
    # Modal auxiliaries ("will", "may", "can") are weaker signals, placed last
    verbs = [
        "threatened",
        "declared",
        "announced",
        "rejected",
        "accepted",
        "agreed",
        "seized",
        "blocked",
        "closed",
        "opened",
        "fired",
        "sent",
        "warned",
        "said",
        "will",
        "may",
        "can",
    ]

    svo, nodes = [], set()

    for art in articles:
        title = art.get("title", "").lower()

        for subj, obj in pairs:
            if subj.lower() in title and obj.lower() in title:
                verb = "mentioned"
                for v in verbs:
                    if v in title:
                        verb = v
                        break

                svo.append((subj, verb, obj))
                nodes.add(subj)
                nodes.add(obj)

    return svo, nodes


def create_graph(svo, nodes, output):
    """Make DOT graph."""
    edge_cnt = Counter(svo)
    dot = output.replace(".txt", ".dot")

    with open(dot, "w") as f:
        f.write("digraph IranHormuz {\n")
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
        }
        for n in sorted(nodes):
            fill = "lightcoral" if n in countries else "lightblue"
            safe = n.replace('"', "")[:20]
            f.write(f'  "{safe}" [label="{n}", fillcolor="{fill}"];\n')

        f.write("\n")

        for (s, v, o), c in sorted(edge_cnt.items(), key=lambda x: -x[1])[:35]:
            safe_s = s.replace('"', "")[:20]
            safe_o = o.replace('"', "")[:20]
            w = min(c * 0.5, 2.5)
            f.write(f'  "{safe_s}" -> "{safe_o}" [label="{v}", penwidth={w}];\n')

        f.write("}\n")

    print(f"Graph: {dot}")

    png = dot.replace(".dot", ".png")
    subprocess.run(["dot", "-Tpng", dot, "-o", png], check=False, capture_output=True)
    if os.path.exists(png):
        print(f"PNG: {png}")


def main():
    print("=" * 60)
    print("IRAN/HORMUZ EXTRACTION")
    print("=" * 60)

    extract_articles(300)

    articles = parse_articles("output/iran_hormuz.txt")
    print(f"Articles: {len(articles)}")

    svo, nodes = extract_svo(articles)
    print(f"SVO: {len(svo)}, Nodes: {len(nodes)}")

    if svo:
        for (s, v, o), c in Counter(svo).most_common(15):
            print(f"  {s} -> {o} ({v}): {c}")

    create_graph(svo, nodes, "output/iran_hormuz.txt")
    print("\nDONE")


if __name__ == "__main__":
    main()
