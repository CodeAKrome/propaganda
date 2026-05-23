#!/usr/bin/env python3
"""
Generate visualizations for violent rhetoric analysis using Tufte principles.
"""

import json
import os

OUTPUT_DIR = '/Users/kyle/hub/propaganda/docs/violent_rhetoric'


def generate_visualizations():
    """Generate SVG visualizations"""
    
    with open(f'{OUTPUT_DIR}/data/mongo_results.json') as f:
        data = json.load(f)
    
    stats = data['stats']
    
    # Data extraction
    orientations = ['LEFT', 'CENTER', 'RIGHT']
    percentages = [stats['L']['percentage'], stats['C']['percentage'], stats['R']['percentage']]
    counts = [stats['L']['violent'], stats['C']['violent'], stats['R']['violent']]
    totals = [stats['L']['total'], stats['C']['total'], stats['R']['total']]
    
    # Visualization 1: Bar chart - Violent % by orientation
    bar_svg = generate_bar_chart(orientations, percentages, totals)
    with open(f'{OUTPUT_DIR}/visualizations/coverage.svg', 'w') as f:
        f.write(bar_svg)
    
    # Visualization 2: Top violent terms comparison
    terms_svg = generate_terms_chart(stats)
    with open(f'{OUTPUT_DIR}/visualizations/terms.svg', 'w') as f:
        f.write(terms_svg)
    
    # Visualization 3: Comparison table
    table_svg = generate_comparison_table(orientations, counts, totals, percentages)
    with open(f'{OUTPUT_DIR}/visualizations/comparison.svg', 'w') as f:
        f.write(table_svg)
    
    print("[✓] Generated visualizations:")
    print(f"  - coverage.svg")
    print(f"  - terms.svg")
    print(f"  - comparison.svg")


def generate_bar_chart(orientations, percentages, totals):
    """Generate bar chart for violent % by orientation"""
    
    max_val = max(percentages) * 1.2
    bar_width = 80
    bar_gap = 40
    chart_height = 200
    chart_width = len(orientations) * (bar_width + bar_gap) + 100
    
    colors = {'LEFT': '#3498db', 'CENTER': '#95a5a6', 'RIGHT': '#e74c3c'}
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {chart_width} 300" 
       style="font-family: 'SF Pro', -apple-system, sans-serif;">
  <style>
    .label {{ font-size: 12px; fill: #333; }}
    .title {{ font-size: 14px; font-weight: 600; }}
    .value {{ font-size: 11px; fill: #666; }}
    .axis {{ stroke: #ddd; stroke-width: 1; }}
  </style>
  
  <text x="{chart_width/2}" y="25" text-anchor="middle" class="title">Violent Rhetoric by Political Orientation</text>
  <text x="{chart_width/2}" y="45" text-anchor="middle" class="value">Percentage of articles containing violent language</text>
'''
    
    # Y-axis
    svg += f'<line x1="60" y1="70" x2="60" y2="{70 + chart_height}" class="axis"/>\n'
    
    # Y-axis labels
    for i in range(5):
        y = 70 + (chart_height / 4) * i
        val = max_val - (max_val / 4) * i
        svg += f'<text x="50" y="{y+4}" text-anchor="end" class="value">{val:.1f}%</text>\n'
        svg += f'<line x1="58" y1="{y}" x2="{chart_width-20}" y2="{y}" stroke="#eee" stroke-width="1"/>\n'
    
    # Bars
    for i, (orient, pct) in enumerate(zip(orientations, percentages)):
        x = 80 + i * (bar_width + bar_gap)
        bar_height = (pct / max_val) * chart_height
        y = 70 + chart_height - bar_height
        
        svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{colors[orient]}" rx="2"/>\n'
        svg += f'<text x="{x + bar_width/2}" y="{y-8}" text-anchor="middle" class="value">{pct:.1f}%</text>\n'
        svg += f'<text x="{x + bar_width/2}" y="{70 + chart_height + 20}" text-anchor="middle" class="label">{orient}</text>\n'
        
        # Count annotation
        svg += f'<text x="{x + bar_width/2}" y="{y + bar_height/2}" text-anchor="middle" fill="white" font-size="10">{int(pct*totals[i]/100)}</text>\n'
    
    svg += '</svg>'
    return svg


def generate_terms_chart(stats):
    """Generate chart comparing top violent terms"""
    
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 350"
       style="font-family: 'SF Pro', -apple-system, sans-serif;">
  <style>
    .label {{ font-size: 11px; }}
    .title {{ font-size: 14px; font-weight: 600; }}
    .subtitle {{ font-size: 11px; fill: #666; }}
  </style>
  
  <text x="300" y="25" text-anchor="middle" class="title">Top Violent Terms by Political Orientation</text>
  <text x="300" y="45" text-anchor="middle" class="subtitle">Most frequently used violent language</text>
'''
    
    cols = [
        ('L', 'LEFT', '#3498db'),
        ('C', 'CENTER', '#95a5a6'),
        ('R', 'RIGHT', '#e74c3c')
    ]
    
    col_width = 180
    start_x = 30
    
    for col_idx, (key, label, color) in enumerate(cols):
        x = start_x + col_idx * col_width
        
        # Column header
        svg += f'<text x="{x + 80}" y="70" text-anchor="middle" class="label" font-weight="600">{label}</text>\n'
        
        # Top 5 terms
        terms = stats[key]['top_terms'][:5]
        for i, (term, count) in enumerate(terms):
            y = 95 + i * 25
            
            # Normalize width
            max_count = stats[key]['top_terms'][0][1] if stats[key]['top_terms'] else 1
            bar_width = (count / max_count) * 120
            
            svg += f'<text x="{x}" y="{y}" class="label">{term}</text>\n'
            svg += f'<rect x="{x + 60}" y="{y-10}" width="{bar_width}" height="14" fill="{color}" rx="2" opacity="0.7"/>\n'
            svg += f'<text x="{x + 65 + bar_width}" y="{y}" class="label" fill="#666">{count}</text>\n'
    
    svg += '</svg>'
    return svg


def generate_comparison_table(orientations, counts, totals, percentages):
    """Generate comparison table"""
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200"
       style="font-family: 'SF Pro', -apple-system, sans-serif;">
  <style>
    .header {{ font-size: 12px; font-weight: 600; }}
    .cell {{ font-size: 11px; fill: #333; }}
    .title {{ font-size: 14px; font-weight: 600; }}
  </style>
  
  <text x="250" y="25" text-anchor="middle" class="title">Violent Rhetoric Summary</text>
  
  <!-- Header row -->
  <rect x="20" y="50" width="460" height="25" fill="#f5f5f5"/>
  <text x="40" y="67" class="header">Orientation</text>
  <text x="160" y="67" class="header">Total Articles</text>
  <text x="260" y="67" class="header">Violent Articles</text>
  <text x="380" y="67" class="header">Violent %</text>
  
  <!-- Data rows -->
'''
    
    colors = {'LEFT': '#e8f4fc', 'CENTER': '#f8f8f8', 'RIGHT': '#fde8e8'}
    text_colors = {'LEFT': '#3498db', 'CENTER': '#95a5a6', 'RIGHT': '#e74c3c'}
    
    for i, (orient, count, total, pct) in enumerate(zip(orientations, counts, totals, percentages)):
        y = 80 + i * 30
        svg += f'<rect x="20" y="{y}" width="460" height="25" fill="{colors[orient]}"/>\n'
        svg += f'<text x="40" y="{y+17}" class="cell" font-weight="600" fill="{text_colors[orient]}">{orient}</text>\n'
        svg += f'<text x="160" y="{y+17}" class="cell">{total:,}</text>\n'
        svg += f'<text x="260" y="{y+17}" class="cell">{count:,}</text>\n'
        svg += f'<text x="380" y="{y+17}" class="cell" font-weight="600">{pct:.2f}%</text>\n'
    
    svg += '</svg>'
    return svg


if __name__ == '__main__':
    generate_visualizations()