#!/usr/bin/env python3
"""
Cypher to GraphViz DOT and Interactive HTML Converter

Reads .cypher files from db/output/ and converts them to:
1. Graphviz DOT files for static rendering
2. Interactive HTML with menus using vis.js for graph manipulation

Entity types are detected and assigned colors/shapes:
- PERSON: blue, ellipse
- ORGANIZATION: red, box  
- LOCATION: green, diamond
- DATE/TIME: purple, triangle
- NUMBER: orange, hexagon
- OTHER: gray, box
"""

import re
import os
import glob
import argparse
from collections import defaultdict


# Entity type definitions
ENTITY_TYPES = {
    'PERSON': {
        'keywords': ['president', 'minister', 'governor', 'king', 'queen', 'prime minister',
                    'chief', 'director', 'ceo', 'chairman', 'bishop', 'archbishop', 'barrister',
                    'leader', 'founder', 'administrator', 'spokesperson', 'senator', 'member',
                    'officer', 'judge', 'attorney', 'lawyer', 'chancellor', 'secretary',
                    'vicol', 'farage', 'starmer', 'trump', 'maduro', 'abbott', 'maltz', 'marchenko'],
        'color': '#3498db',
        'shape': 'ellipse'
    },
    'ORGANIZATION': {
        'keywords': ['party', 'government', 'department', 'agency', 'ngo', 'cartel', 
                    'parliament', 'assembly', 'council', 'church', 'military', 'court',
                    'chambers', 'centre', 'committee', 'union', 'group', 'command',
                    'reform uk', 'labour', 'conservative', 'democratic', 'republican',
                    'dea', 'fbi', 'cia', 'ice', 'un', 'eu', 'sanctions'],
        'color': '#e74c3c',
        'shape': 'box'
    },
    'LOCATION': {
        'keywords': ['city', 'state', 'country', 'border', 'island', 'ocean', 'sea', 
                    'river', 'mountain', 'pass', 'texas', 'mexico', 'uk', 'us', 'venezuela',
                    'london', 'caracas', 'dover', 'minneapolis', 'port', 'zone', 'region',
                    'area', 'capital', 'province', 'district', 'congress', 'capitol',
                    'channel', 'europe', 'america', 'asia', 'africa'],
        'color': '#2ecc71',
        'shape': 'diamond'
    },
    'DATE': {
        'keywords': ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                    'august', 'september', 'october', 'november', 'december', 'year',
                    'day', 'week', 'month', 'century', 'decade', '2024', '2025', '2026'],
        'color': '#9b59b6',
        'shape': 'triangle'
    },
    'NUMBER': {
        'keywords': ['000', 'million', 'billion', 'percent', '%', '$', 'amount', 'number',
                    'count', 'total', 'population', 'rate', 'score', 'level'],
        'color': '#f39c12',
        'shape': 'hexagon'
    }
}

DEFAULT_ENTITY = {
    'color': '#95a5a6',
    'shape': 'box'
}


def detect_entity_type(node: str) -> dict:
    """Detect entity type based on node name."""
    node_lower = node.lower()
    
    # Check each entity type
    for entity_type, config in ENTITY_TYPES.items():
        for keyword in config['keywords']:
            if keyword in node_lower:
                return {
                    'type': entity_type,
                    'color': config['color'],
                    'shape': config['shape']
                }
    
    return {
        'type': 'OTHER',
        'color': DEFAULT_ENTITY['color'],
        'shape': DEFAULT_ENTITY['shape']
    }


def parse_cypher_file(filepath: str) -> tuple:
    """Parse a cypher file and extract nodes and edges."""
    nodes = set()
    edges = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: ("Source", "Target", "relationship", "description")
    pattern = r'\("([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\)'
    
    matches = re.findall(pattern, content)
    
    for source, target, relationship, description in matches:
        nodes.add(source)
        nodes.add(target)
        edges.append({
            'source': source,
            'target': target,
            'relationship': relationship,
            'description': description
        })
    
    return nodes, edges


def sanitize_id(name: str) -> str:
    """Convert name to a safe graph ID."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace('&', '&').replace('<', '<').replace('>', '>').replace('"', '"').replace("'", "'")


def escape_js(text: str) -> str:
    """Escape JavaScript string special characters."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def generate_dot(nodes: set, edges: list, title: str) -> str:
    """Generate Graphviz DOT format with entity-based colors and shapes."""
    lines = []
    lines.append(f'digraph "{title}" {{')
    lines.append('  graph [')
    lines.append('    rankdir=LR')
    lines.append('    splines=curved')
    lines.append('    overlap=false')
    lines.append('    fontname="Helvetica"')
    lines.append('    fontsize=14')
    lines.append('    pad=0.5')
    lines.append('  ];')
    lines.append('')
    lines.append('  node [')
    lines.append('    fontname="Helvetica"')
    lines.append('    fontsize=12')
    lines.append('    style=filled')
    lines.append('    margin=0.2')
    lines.append('  ];')
    lines.append('')
    lines.append('  edge [')
    lines.append('    fontname="Helvetica"')
    lines.append('    fontsize=10')
    lines.append('    arrowsize=0.8')
    lines.append('    penwidth=1.5')
    lines.append('    color=gray50')
    lines.append('  ];')
    lines.append('')
    
    # Add node definitions with entity-based colors/shapes
    for node in sorted(nodes):
        entity_info = detect_entity_type(node)
        node_id = sanitize_id(node)
        escaped_node = node.replace('"', '\\"')
        
        # DOT shape: ellipse, box, diamond, triangle, hexagon, circle
        dot_shape = entity_info['shape']
        dot_color = entity_info['color']
        
        lines.append(f'  node_{node_id} [')
        lines.append(f'    label="{escaped_node}"')
        lines.append(f'    shape={dot_shape}')
        lines.append(f'    fillcolor={dot_color}')
        lines.append(f'    fontcolor=white')
        lines.append('  ];')
    
    lines.append('')
    
    # Add edges
    for edge in edges:
        source_id = sanitize_id(edge['source'])
        target_id = sanitize_id(edge['target'])
        rel = edge['relationship'].replace('"', '\\"')
        
        lines.append(f'  node_{source_id} -> node_{target_id} [')
        lines.append(f'    label="{rel}"')
        lines.append('  ];')
    
    lines.append('}')
    
    return '\n'.join(lines)


def generate_html(nodes: set, edges: list, title: str) -> str:
    """Generate interactive HTML using vis.js with entity-based colors and shapes."""
    
    # Pre-process nodes and edges
    sorted_nodes = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    
    # Build nodes data with entity-based styling
    nodes_json = []
    for i, node in enumerate(sorted_nodes):
        entity_info = detect_entity_type(node)
        escaped_label = escape_js(node)
        
        # Build nodes data with entity-based styling
    nodes_json = []
    for i, node in enumerate(sorted_nodes):
        entity_info = detect_entity_type(node)
        escaped_label = escape_js(node)
        color = entity_info['color']
        shape = entity_info['shape']
        # Use smaller size for box shapes (OTHER category)
        size = 2 if shape == "box" else 25
        nodes_json.append('{id: %d, label: "%s", color: { background: "%s", border: "#2c3e50" }, shape: "%s", size: %d}' % (i, escaped_label, color, shape, size))

    
    # Build edges data
    edges_json = []
    for edge in edges:
        source_idx = node_to_idx[edge['source']]
        target_idx = node_to_idx[edge['target']]
        rel = escape_js(edge['relationship'])
        edges_json.append(f'{{from: {source_idx}, to: {target_idx}, label: "{rel}"}}')
    
    nodes_str = '[' + ', '.join(nodes_json) + ']'
    edges_str = '[' + ', '.join(edges_json) + ']'
    
    safe_title = escape_html(title)
    safe_filename = sanitize_id(title)
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Count entity types for legend
    entity_counts = defaultdict(int)
    for node in nodes:
        entity_type = detect_entity_type(node)['type']
        entity_counts[entity_type] += 1
    
    # Build legend HTML
    legend_items = []
    for entity_type, config in ENTITY_TYPES.items():
        if entity_counts[entity_type] > 0:
            legend_items.append(
                f'<div class="legend-item"><div class="legend-color" style="background:{config["color"]}"></div>'
                f'<span>{entity_type}: {entity_counts[entity_type]}</span></div>'
            )
    if entity_counts['OTHER'] > 0:
        legend_items.append(
            f'<div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div>'
            f'<span>OTHER: {entity_counts["OTHER"]}</span></div>'
        )
    legend_html = '\n      '.join(legend_items)
    
    html = '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Knowledge Graph: ''' + safe_title + '''</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    body { font-family: sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
    #mynetwork { width: 100%; height: calc(100vh - 220px); min-height: 500px; border: 1px solid #ccc; background: white; }
    .controls { margin: 10px 0; display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
    .controls label { margin-right: 5px; }
    .stats { color: #666; font-size: 14px; margin-left: auto; }
    .legend { display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; padding: 10px; background: white; border: 1px solid #ddd; border-radius: 4px; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
    .legend-color { width: 16px; height: 16px; border-radius: 3px; }
    .shape-icon { width: 16px; height: 16px; display: inline-block; }
  </style>
</head>
<body>
  <h2>Knowledge Graph: ''' + safe_title + '''</h2>
  <div class="legend">
    ''' + legend_html + '''
  </div>
  <div class="controls">
    <label>Layout: 
      <select id="layout">
        <option value="distribute">Distribute</option>
        <option value="stabilize">Stabilize</option>
      </select>
    </label>
    <button onclick="fit()">Fit</button>
    <button onclick="exportPNG()">Export PNG</button>
    <span class="stats">''' + str(num_nodes) + ''' nodes, ''' + str(num_edges) + ''' edges</span>
  </div>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    var nodes = new vis.DataSet(''' + nodes_str + ''');
    var edges = new vis.DataSet(''' + edges_str + ''');
    var container = document.getElementById('mynetwork');
    var data = { nodes: nodes, edges: edges };
    var options = {
      nodes: {
        font: { size: 14, face: 'Helvetica' },
        borderWidth: 2,
        shadow: true
      },
      edges: {
        arrows: 'to',
        font: { size: 10, align: 'middle' },
        smooth: { type: 'continuous' }
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2'
      }
    };
    var network = new vis.Network(container, data, options);
    
    function fit() { network.fit(); }
    
    function exportPNG() {
      var canvas = container.querySelector('canvas');
      var link = document.createElement('a');
      link.download = "''' + safe_filename + '''.png";
      link.href = canvas.toDataURL();
      link.click();
    }
    
    document.getElementById('layout').addEventListener('change', function() {
      if (this.value === 'stabilize') {
        network.setOptions({ physics: { enabled: false } });
      } else {
        network.setOptions({ physics: { enabled: true } });
      }
    });
  </script>
</body>
</html>'''
    
    return html


def process_cypher_file(input_path: str, output_dir: str) -> None:
    """Process a single cypher file and generate DOT and HTML."""
    filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    print(f"Processing: {filename}")
    
    nodes, edges = parse_cypher_file(input_path)
    
    if not nodes:
        print(f"  Warning: No nodes found in {filename}")
        return
    
    print(f"  Found {len(nodes)} nodes, {len(edges)} edges")
    
    # Generate DOT file
    dot_content = generate_dot(nodes, edges, name_without_ext)
    dot_path = os.path.join(output_dir, f"{name_without_ext}.dot")
    with open(dot_path, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    print(f"  Generated: {dot_path}")
    
    # Generate HTML file
    html_content = generate_html(nodes, edges, name_without_ext)
    html_path = os.path.join(output_dir, f"{name_without_ext}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  Generated: {html_path}")


def create_combined_graph(input_dir: str, output_dir: str) -> None:
    """Create a combined graph from all cypher files."""
    print("\n" + "="*50)
    print("Creating combined graph...")
    
    all_nodes = set()
    all_edges = []
    
    cypher_files = glob.glob(os.path.join(input_dir, "*.cypher"))
    
    for filepath in cypher_files:
        nodes, edges = parse_cypher_file(filepath)
        all_nodes.update(nodes)
        all_edges.extend(edges)
    
    if not all_nodes:
        print("No nodes found!")
        return
    
    print(f"Combined: {len(all_nodes)} nodes, {len(all_edges)} edges")
    
    # Generate combined DOT
    dot_content = generate_dot(all_nodes, all_edges, "Combined Knowledge Graph")
    dot_path = os.path.join(output_dir, "combined.dot")
    with open(dot_path, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    print(f"Generated: {dot_path}")
    
    # Generate combined HTML
    html_content = generate_html(all_nodes, all_edges, "Combined Knowledge Graph")
    html_path = os.path.join(output_dir, "combined.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Cypher files to Graphviz DOT and interactive HTML'
    )
    parser.add_argument(
        '--input-dir', '-i',
        default='db/output',
        help='Directory containing .cypher files (default: db/output)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='db/output',
        help='Output directory for .dot and .html files (default: db/output)'
    )
    parser.add_argument(
        '--combined', '-c',
        action='store_true',
        help='Also create a combined graph from all files'
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all cypher files
    cypher_files = glob.glob(os.path.join(input_dir, "*.cypher"))
    
    if not cypher_files:
        print(f"No .cypher files found in {input_dir}")
        return
    
    print(f"Found {len(cypher_files)} cypher files")
    print("="*50)
    print("Entity Types:")
    for etype, config in ENTITY_TYPES.items():
        print(f"  - {etype}: {config['shape']} ({config['color']})")
    print("="*50)
    
    # Process each file
    for filepath in cypher_files:
        process_cypher_file(filepath, output_dir)
    
    # Create combined graph if requested
    if args.combined:
        create_combined_graph(input_dir, output_dir)
    
    print("="*50)
    print("Done!")


if __name__ == '__main__':
    main()
