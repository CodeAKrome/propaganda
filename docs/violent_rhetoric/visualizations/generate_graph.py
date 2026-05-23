#!/usr/bin/env python3
"""
Generate HTML and DOT files from violent rhetoric analysis.
"""

import json
import os
import sys
sys.path.insert(0, '/Users/kyle/hub/propaganda/db')
from cypher_to_graph import generate_html, generate_dot

OUTPUT_DIR = '/Users/kyle/hub/propaganda/docs/violent_rhetoric'


def main():
    # Load MongoDB results
    with open(f'{OUTPUT_DIR}/data/mongo_results.json') as f:
        data = json.load(f)
    
    stats = data['stats']
    
    # Build nodes and edges from stats
    nodes = set()
    edges = []
    
    # Add orientation nodes
    for orient in ['L', 'C', 'R']:
        label = {'L': 'LEFT', 'C': 'CENTER', 'R': 'RIGHT'}[orient]
        nodes.add(label)
        
        # Add top targets
        for target, count in stats[orient]['top_targets'][:5]:
            nodes.add(target)
            edges.append({
                'source': label,
                'target': target,
                'relationship': 'targets',
                'description': f'{count} mentions'
            })
        
        # Add top terms
        for term, count in stats[orient]['top_terms'][:3]:
            nodes.add(term)
            edges.append({
                'source': label,
                'target': term,
                'relationship': 'uses',
                'description': f'{count} times'
            })
    
    nodes = list(nodes)
    
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    # Generate HTML
    html = generate_html(nodes, edges, "Violent Rhetoric by Political Orientation")
    with open(f'{OUTPUT_DIR}/output/violent_rhetoric.html', 'w') as f:
        f.write(html)
    print(f"[✓] HTML: {OUTPUT_DIR}/output/violent_rhetoric.html")
    
    # Generate DOT
    dot = generate_dot(nodes, edges, "Violent Rhetoric")
    with open(f'{OUTPUT_DIR}/output/violent_rhetoric.dot', 'w') as f:
        f.write(dot)
    print(f"[✓] DOT: {OUTPUT_DIR}/output/violent_rhetoric.dot")


if __name__ == '__main__':
    main()