#!/usr/bin/env python3
"""
Cypher to GraphViz DOT and Interactive HTML Converter

Reads .cypher files from db/output/ and converts them to:
1. Graphviz DOT files for static rendering
2. Interactive HTML with menus using vis.js for graph manipulation
"""

import re
import os
import glob
import argparse
from collections import defaultdict


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
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", "&#39;")


def escape_js(text: str) -> str:
    """Escape JavaScript string special characters."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def get_node_color(node: str) -> str:
    """Determine node color based on name patterns."""
    node_lower = node.lower()
    
    # Person indicators
    person_keywords = ['king', 'queen', 'president', 'minister', 'governor', 'prime minister', 
                       'chief', 'director', 'ceo', 'chairman', 'bishop', 'archbishop',
                       'barrister', 'spokesperson', 'leader', 'founder', 'administrator']
    if any(kw in node_lower for kw in person_keywords):
        return '#3498db'
    
    # Organization indicators
    org_keywords = ['party', 'government', 'department', 'agency', 'ngo', 'cartel', 
                    'parliament', 'assembly', 'council', 'un', 'eu', 'church', 'military']
    if any(kw in node_lower for kw in org_keywords):
        return '#e74c3c'
    
    # Location indicators
    loc_keywords = ['city', 'state', 'country', 'border', 'island', 'ocean', 'sea', 
                   'river', 'mountain', 'pass', 'texas', 'mexico', 'uk', 'us', 'venezuela']
    if any(kw in node_lower for kw in loc_keywords):
        return '#2ecc71'
    
    return '#f39c12'


def generate_dot(nodes: set, edges: list, title: str) -> str:
    """Generate Graphviz DOT format."""
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
    lines.append('    shape=box')
    lines.append('    style=filled')
    lines.append('    fillcolor=lightblue')
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
    
    # Group nodes by first letter for clustering
    node_groups = defaultdict(list)
    for node in sorted(nodes):
        if node:
            first_char = node[0].upper()
            node_groups[first_char].append(node)
    
    # Add nodes with clusters
    for letter, node_list in sorted(node_groups.items()):
        lines.append(f'  subgraph cluster_{letter} {{')
        lines.append(f'    label="{letter}"')
        lines.append('    style=filled')
        lines.append('    color=lightgray')
        for node in node_list:
            node_id = sanitize_id(node)
            escaped_node = node.replace('"', '\\"')
            lines.append(f'    node_{node_id} [label="{escaped_node}"];')
        lines.append('  }')
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
    """Generate interactive HTML using vis.js with menus."""
    
    # Pre-process nodes and edges with escaped values
    sorted_nodes = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    
    nodes_data = []
    for i, node in enumerate(sorted_nodes):
        color = get_node_color(node)
        escaped_label = escape_js(node)
        nodes_data.append(f'    {{ id: {i}, label: "{escaped_label}", color: "{color}", font: {{ face: "Helvetica", size: 14 }} }}')
    
    edges_data = []
    for edge in edges:
        source_idx = node_to_idx[edge['source']]
        target_idx = node_to_idx[edge['target']]
        
        # Create edge title with full description
        src = escape_js(edge['source'])
        tgt = escape_js(edge['target'])
        rel = escape_js(edge['relationship'])
        desc = escape_js(edge['description'])
        title_str = f"{src} --[{rel}]--> {tgt}\\n\\n{desc}"
        
        edges_data.append(f'    {{ from: {source_idx}, to: {target_idx}, label: "{rel}", title: "{title_str}", arrows: "to", font: {{ face: "Helvetica", size: 10, color: "gray" }} }}')
    
    nodes_str = '[\n' + ',\n'.join(nodes_data) + '\n  ]'
    edges_str = '[\n' + ',\n'.join(edges_data) + '\n  ]'
    
    safe_title = escape_html(title)
    safe_filename = sanitize_id(title)
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '  <title>Knowledge Graph: ' + safe_title + '</title>\n'
        '  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
        '  <style type="text/css">\n'
        '    * { margin: 0; padding: 0; box-sizing: border-box; }\n'
        '    body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }\n'
        '    #menu { position: fixed; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #16213e 0%, #0f3460 100%); padding: 15px 20px; display: flex; align-items: center; gap: 20px; z-index: 1000; box-shadow: 0 2px 20px rgba(0,0,0,0.5); }\n'
        '    #menu h1 { font-size: 1.4rem; font-weight: 600; color: #e94560; }\n'
        '    #menu select, #menu button { padding: 8px 16px; border-radius: 6px; border: 1px solid #e94560; background: #0f3460; color: #eee; font-size: 0.9rem; cursor: pointer; transition: all 0.3s ease; }\n'
        '    #menu select:hover, #menu button:hover { background: #e94560; color: #fff; }\n'
        '    #menu label { font-size: 0.85rem; color: #aaa; }\n'
        '    #menu input[type="checkbox"] { margin-right: 5px; }\n'
        '    #stats { position: fixed; bottom: 20px; left: 20px; background: rgba(15, 52, 96, 0.9); padding: 15px 20px; border-radius: 8px; font-size: 0.85rem; color: #aaa; z-index: 1000; border: 1px solid #e94560; }\n'
        '    #stats strong { color: #e94560; }\n'
        '    #graph { width: 100vw; height: 100vh; padding-top: 60px; }\n'
        '    #node-info { position: fixed; top: 80px; right: 20px; width: 350px; max-height: calc(100vh - 120px); overflow-y: auto; background: rgba(15, 52, 96, 0.95); padding: 20px; border-radius: 10px; border: 1px solid #e94560; display: none; z-index: 1000; box-shadow: 0 4px 30px rgba(0,0,0,0.5); }\n'
        '    #node-info h3 { color: #e94560; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; }\n'
        '    #node-info .edge-item { padding: 10px; margin-bottom: 8px; background: rgba(0,0,0,0.2); border-radius: 5px; font-size: 0.85rem; }\n'
        '    #node-info .edge-item .relation { color: #4ecca3; font-weight: 600; }\n'
        '    #node-info .edge-item .target { color: #e94560; }\n'
        '    #node-info .edge-item .desc { color: #aaa; margin-top: 5px; font-size: 0.8rem; }\n'
        '    .legend { position: fixed; bottom: 20px; right: 20px; background: rgba(15, 52, 96, 0.9); padding: 15px; border-radius: 8px; border: 1px solid #333; z-index: 1000; }\n'
        '    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 0.75rem; }\n'
        '    .legend-color { width: 15px; height: 15px; border-radius: 3px; }\n'
        '  </style>\n'
        '</head>\n'
        '<body>\n'
        '  <div id="menu">\n'
        '    <h1>🕸️ ' + safe_title + '</h1>\n'
        '    <label>Layout:</label>\n'
        '    <select id="layout-select">\n'
        '      <option value="hierarchical">Hierarchical</option>\n'
        '      <option value="horizontal">Horizontal</option>\n'
        '      <option value="vertical">Vertical</option>\n'
        '      <option value="circle">Circle</option>\n'
        '    </select>\n'
        '    <label>Physics:</label>\n'
        '    <input type="checkbox" id="physics-toggle" checked>\n'
        '    <label>Labels:</label>\n'
        '    <input type="checkbox" id="labels-toggle" checked>\n'
        '    <button id="zoom-in">Zoom In</button>\n'
        '    <button id="zoom-out">Zoom Out</button>\n'
        '    <button id="fit-graph">Fit</button>\n'
        '    <button id="export-png">Export PNG</button>\n'
        '  </div>\n'
        '\n'
        '  <div id="stats">\n'
        '    <strong>' + str(num_nodes) + '</strong> nodes | <strong>' + str(num_edges) + '</strong> edges\n'
        '  </div>\n'
        '\n'
        '  <div id="node-info">\n'
        '    <h3 id="node-title"></h3>\n'
        '    <div id="node-edges"></div>\n'
        '  </div>\n'
        '\n'
        '  <div class="legend">\n'
        '    <div class="legend-item"><div class="legend-color" style="background: #3498db;"></div><span>Person</span></div>\n'
        '    <div class="legend-item"><div class="legend-color" style="background: #e74c3c;"></div><span>Organization</span></div>\n'
        '    <div class="legend-item"><div class="legend-color" style="background: #2ecc71;"></div><span>Location</span></div>\n'
        '    <div class="legend-item"><div class="legend-color" style="background: #f39c12;"></div><span>Other</span></div>\n'
        '  </div>\n'
        '\n'
        '  <div id="graph"></div>\n'
        '\n'
        '  <script type="text/javascript">\n'
        '    // Node and edge data\n'
        '    const nodes = new vis.DataSet(' + nodes_str + ');\n'
        '    const edges = new vis.DataSet(' + edges_str + ');\n'
        '\n'
        '    // Create network\n'
        '    const container = document.getElementById("graph");\n'
        '    const data = { nodes: nodes, edges: edges };\n'
        '\n'
        '    const options = {\n'
        '      nodes: {\n'
        '        shape: "box",\n'
        '        margin: 10,\n'
        '        borderWidth: 2,\n'
        '        shadow: true,\n'
        '        font: { face: "Helvetica", size: 14, color: "#eee" }\n'
        '      },\n'
        '      edges: {\n'
        '        width: 1.5,\n'
        '        color: { color: "#555", highlight: "#e94560" },\n'
        '        smooth: { type: "continuous", roundness: 0.5 },\n'
        '        arrows: { to: { enabled: true, scaleFactor: 0.8 } }\n'
        '      },\n'
        '      physics: {\n'
        '        enabled: true,\n'
        '        solver: "forceAtlas2Based",\n'
        '        forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01, springLength: 150, springConstant: 0.08 },\n'
        '        maxVelocity: 50,\n'
        '        stabilization: { enabled: true, iterations: 200 }\n'
        '      },\n'
        '      layout: {\n'
        '        hierarchical: { enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 150, nodeSpacing: 100 }\n'
        '      },\n'
        '      interaction: {\n'
        '        hover: true,\n'
        '        tooltipDelay: 200,\n'
        '        hideEdgesOnDrag: true,\n'
        '        multiselect: true,\n'
        '        navigationButtons: false,\n'
        '        keyboard: true\n'
        '      }\n'
        '    };\n'
        '\n'
        '    let network = new vis.Network(container, data, options);\n'
        '\n'
        '    // Layout switcher\n'
        '    document.getElementById("layout-select").addEventListener("change", function() {\n'
        '      const layout = this.value;\n'
        '      const newOptions = { layout: { hierarchical: { enabled: false } } };\n'
        '\n'
        '      if (layout === "hierarchical") {\n'
        '        newOptions.layout.hierarchical = { enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 150, nodeSpacing: 100 };\n'
        '      } else if (layout === "horizontal") {\n'
        '        newOptions.layout = { improvedLayout: true, hierarchical: { enabled: false } };\n'
        '      } else if (layout === "vertical") {\n'
        '        newOptions.layout = { improvedLayout: true, hierarchical: { enabled: true, direction: "UD", sortMethod: "directed" } };\n'
        '      } else if (layout === "circle") {\n'
        '        newOptions.layout = { improvedLayout: true, hierarchical: { enabled: false } };\n'
        '        setTimeout(() => { network.layoutEngine.options = { randomSeed: 2 }; network.moveTo({ position: { x: 0, y: 0 }, scale: 0.5 }); }, 100);\n'
        '      }\n'
        '\n'
        '      network.setOptions(newOptions);\n'
        '    });\n'
        '\n'
        '    // Physics toggle\n'
        '    document.getElementById("physics-toggle").addEventListener("change", function() {\n'
        '      network.setOptions({ physics: { enabled: this.checked } });\n'
        '    });\n'
        '\n'
        '    // Labels toggle\n'
        '    document.getElementById("labels-toggle").addEventListener("change", function() {\n'
        '      const nodeIds = nodes.getIds();\n'
        '      nodes.update(nodeIds.map(id => ({ id: id, label: this.checked ? nodes.get(id).label : "" })));\n'
        '    });\n'
        '\n'
        '    // Zoom controls\n'
        '    document.getElementById("zoom-in").addEventListener("click", () => { network.moveTo({ scale: network.getScale() * 1.2, animation: true }); });\n'
        '    document.getElementById("zoom-out").addEventListener("click", () => { network.moveTo({ scale: network.getScale() / 1.2, animation: true }); });\n'
        '    document.getElementById("fit-graph").addEventListener("click", () => { network.fit({ animation: true }); });\n'
        '\n'
        '    // Node click handler - show info panel\n'
        '    network.on("click", function(params) {\n'
        '      const nodeInfo = document.getElementById("node-info");\n'
        '      const nodeTitle = document.getElementById("node-title");\n'
        '      const nodeEdges = document.getElementById("node-edges");\n'
        '\n'
        '      if (params.nodes.length > 0) {\n'
        '        const nodeId = params.nodes[0];\n'
        '        const nodeLabel = nodes.get(nodeId).label;\n'
        '\n'
        '        const connectedEdges = edges.get().filter(e => e.from === nodeId || e.to === nodeId);\n'
        '\n'
        '        nodeTitle.textContent = nodeLabel;\n'
        '        nodeEdges.innerHTML = "";\n'
        '\n'
        '        connectedEdges.forEach(function(edge) {\n'
        '          const isSource = edge.from === nodeId;\n'
        '          const otherNode = isSource ? edge.to : edge.from;\n'
        '          const otherLabel = nodes.get(otherNode).label;\n'
        '\n'
        '          const edgeDiv = document.createElement("div");\n'
        '          edgeDiv.className = "edge-item";\n'
        '          edgeDiv.innerHTML = "<span class="relation">" + (edge.label || "related_to") + "</span> <span class="target">" + otherLabel + "</span><div class="desc"></div>";\n'
        '          nodeEdges.appendChild(edgeDiv);\n'
        '        });\n'
        '\n'
        '        nodeInfo.style.display = "block";\n'
        '      } else {\n'
        '        nodeInfo.style.display = "none";\n'
        '      }\n'
        '    });\n'
        '\n'
        '    // Export PNG\n'
        '    document.getElementById("export-png").addEventListener("click", () => {\n'
        '      const canvas = container.querySelector("canvas");\n'
        '      const link = document.createElement("a");\n'
        '      link.download = "' + safe_filename + '.png";\n'
        '      link.href = canvas.toDataURL("image/png");\n'
        '      link.click();\n'
        '    });\n'
        '\n'
        '    // Double-click to focus node\n'
        '    network.on("doubleClick", function(params) {\n'
        '      if (params.nodes.length > 0) { network.focus(params.nodes[0], { scale: 1.5, animation: true }); }\n'
        '    });\n'
        '  </script>\n'
        '</body>\n'
        '</html>'
    )
    
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
        
        # Add source file as a tag to edges
        filename = os.path.splitext(os.path.basename(filepath))[0]
        for edge in edges:
            edge['source_file'] = filename
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
