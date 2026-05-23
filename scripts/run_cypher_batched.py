#!/usr/bin/env python3
"""
Batched SVO extraction pipeline.
Run: python3 scripts/run_cypher_batched.py
"""

import os
import re
import subprocess
from collections import Counter

MODEL = "gpt-oss:120b"
SVO_PROMPT = "db/prompt/kgsvo.txt"
BATCH_SIZE = 3
TIMEOUT = 90  # seconds per batch


def format_batch(articles, batch_num):
    """Format articles into .vec format."""
    lines = [
        "=" * 80,
        f"CATEGORY: Iran/Hormuz - Batch {batch_num}",
        f"Articles: {len(articles)}",
        "=" * 80,
        "",
    ]

    for art in articles:
        article_lines = art.strip().split("\n")
        aid = src = title = text = ""
        for l in article_lines:
            if l.startswith("ID:"):
                aid = l.replace("ID:", "").strip()
            elif l.startswith("SOURCE:"):
                src = l.replace("SOURCE:", "").strip()
            elif l.startswith("TITLE:"):
                title = l.replace("TITLE:", "").strip()
            elif l.startswith("ARTICLE:"):
                text = l.replace("ARTICLE:", "").strip()
            elif text:
                text += " " + l.strip()

        # NOTE: Published date is hardcoded placeholder. Extract from article metadata for production use.
        lines.extend(
            [
                "---",
                f"ID: {aid}",
                f"Title:  {title}",
                "Published: 2026-04-20T00:00:00",  # TODO: Extract actual publish date
                f"Source: {src}",
                "Bias: {}",
                "<entities>",
                "</entities>",
                f"Text: {title}",
                text[:600],  # Truncated to 600 chars for LLM context window
                "",
            ]
        )

    return "\n".join(lines)


def parse_tuples(output):
    """Extract tuples from LLM output."""
    tuples = []
    for line in output.split("\n"):
        line = line.strip()
        if not line.startswith("(") or '", "' not in line:
            continue

        parts = re.findall(r'"([^"]+)"', line)
        if len(parts) >= 3:
            s = parts[0][:40].strip()
            t = parts[1][:40].strip()
            r = parts[2][:40].strip()
            if s and t and r and len(s) > 1 and len(t) > 1:
                tuples.append((s, t, r))

    return tuples


def main():
    # Read articles
    with open("output/iran_hormuz_7day.txt") as f:
        articles = f.read().split("\n---\n")

    # Process in batches
    all_tuples = []
    total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(articles), BATCH_SIZE):
        batch = articles[batch_idx : batch_idx + BATCH_SIZE]
        if not batch or not batch[0].strip():
            continue

        batch_num = batch_idx // BATCH_SIZE + 1

        vec = format_batch(batch, batch_num)
        with open(SVO_PROMPT) as f:
            prompt = f.read()

        try:
            print(f"Batch {batch_num}/{total_batches}...", end=" ", flush=True)
            result = subprocess.run(
                ["ollama", "run", MODEL],
                input=prompt + "\n" + vec,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )

            # Clean ANSI
            output = re.sub(r"\x1b\[[0-9;]*[JK]", "", result.stdout)
            tuples = parse_tuples(output)
            print(f"{len(tuples)}")
            all_tuples.extend(tuples)

        except subprocess.TimeoutExpired:
            print("timeout")
            continue

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tuples:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"\nTotal unique: {len(unique)}")

    # Write cypher
    cypher_file = "output/iran_hormuz_7day.cypher"
    with open(cypher_file, "w") as f:
        f.write("\n")
        for s, t, r in unique:
            f.write(f'("{s}", "{t}", "{r}", "")\n')
    print(f"Wrote: {cypher_file}")

    # Write DOT
    nodes = set(s for s, t, r in unique) | set(t for s, t, r in unique)
    countries = {
        "Iran",
        "US",
        "United States",
        "Iraq",
        "Pakistan",
        "Trump",
        "Israel",
        "China",
        "Russia",
        "UK",
        "France",
    }

    dot_file = "output/iran_hormuz_7day.dot"
    with open(dot_file, "w") as f:
        f.write("digraph IranHormuz7day {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=box, style=filled, fontsize=11];\n")
        f.write("  edge [color=gray, fontsize=9];\n\n")

        for n in sorted(nodes):
            fill = "lightcoral" if n in countries else "lightblue"
            safe = n.replace('"', "")[:30]
            f.write(f'  "{safe}" [label="{n}", fillcolor="{fill}"];\n')

        f.write("\n")
        for s, t, r in unique:
            safe_s = s.replace('"', "")[:25]
            safe_t = t.replace('"', "")[:25]
            safe_r = r.replace('"', "")[:15]
            f.write(f'  "{safe_s}" -> "{safe_t}" [label="{safe_r}"];\n')
        f.write("}\n")
    print(f"Wrote: {dot_file}")

    # Generate PNG - use list form to avoid shell injection
    png_file = dot_file.replace(".dot", ".png")
    subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=False)
    if os.path.exists(png_file):
        print(f"PNG: {png_file} ({os.path.getsize(png_file)} bytes)")


if __name__ == "__main__":
    main()
