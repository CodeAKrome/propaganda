#!/usr/bin/env python3

"""
Module for generating news videos using AI-driven backgrounds and TTS.
"""
"""
Synthetic Bias Training Data Generator
========================================
Generates balanced training data with all combinations of:
- dir: L, C, R (political direction)
- deg: L, M, H (bias degree)

Each combination has equal samples to prevent data skew.
"""

import json
import random
import argparse
from typing import Dict, List, Any
from pathlib import Path


# ==============================================================================
# TEMPLATES FOR SYNTHETIC ARTICLES
# ==============================================================================

# Templates for LEFT bias
LEFT_TEMPLATES = [
    "Progressive advocates called for significant reforms today, emphasizing the need for systemic change to address inequality. Community organizers gathered to discuss solutions that prioritize working families and social justice initiatives.",
    "Labor unions announced support for new legislation aimed at protecting workers' rights and expanding healthcare access. The bill includes provisions for minimum wage increases and stronger workplace safety regulations.",
    "Environmental groups praised the new climate policy, calling it a crucial step toward sustainability. Scientists emphasized the urgent need for renewable energy investment and environmental protection measures.",
    "Civil rights organizations celebrated the court's decision as a victory for equality. Activists argued that this ruling advances the ongoing struggle for social justice and human rights.",
    "Progressives in Congress unveiled a comprehensive plan to address income inequality. The proposal includes tax reforms targeting wealth redistribution and expanded social programs.",
    "Community leaders called for criminal justice reform, citing disproportionate impacts on marginalized communities. Research shows systemic biases that advocates say must be addressed immediately.",
    "Healthcare advocates pushed for universal coverage, arguing that healthcare is a fundamental human right. Pollsters note growing public support for single-payer systems.",
    "Education reformers called for increased funding for public schools in underserved areas. Teachers' unions backed the initiative, saying it would help close achievement gaps.",
    "Voting rights activists rallied support for election reform measures. They argue that accessible voting is essential for democracy and civic participation.",
    "Social justice advocates organized a rally to protest discriminatory practices. Participants demanded accountability from institutions they say perpetuate systemic oppression."
]

LEFT_REASONS = [
    "The article presents progressive viewpoints and frames issues in terms of social justice, workers' rights, and equality. Sources are primarily from progressive advocacy groups and liberal perspectives.",
    "The tone strongly favors progressive policies and frames conservative positions as harmful to working families. Language emphasizes the need for systemic change and criticizes existing power structures.",
    "The piece features left-leaning sources almost exclusively and uses loaded language like 'social justice', 'inequality', and 'workers' rights'. It frames reform as morally necessary.",
    "The article uses framing that emphasizes systemic oppression and advocates for structural changes. Sources are uniformly from progressive organizations and activists.",
    "The language and source selection strongly favor left-wing perspectives on economic and social issues. The article frames inequality as a central problem requiring government intervention."
]

# Templates for CENTER bias
CENTER_TEMPLATES = [
    "Analysts offered mixed views on the latest policy proposal, noting both potential benefits and drawbacks. The legislation faces scrutiny from multiple stakeholder groups with competing interests.",
    "A new poll shows divided public opinion on the controversial measure. Supporters and critics presented competing arguments about its potential impact on the economy and society.",
    "Both political parties offered competing proposals on the issue. Observers note that finding common ground may prove challenging given the current political climate.",
    "Economists presented conflicting assessments of the policy's potential effects. Some warn of unintended consequences while others see potential benefits for certain sectors.",
    "The proposed legislation generated debate among experts with varying perspectives. Policy analysts suggest the final outcome may depend on political negotiations.",
    "Stakeholders from across the political spectrum weighed in on the regulatory changes. Business groups and consumer advocates offered different assessments of potential impacts.",
    "The governor's proposal drew reactions from multiple ideological corners. Political observers note that building consensus will require significant compromise.",
    "Experts disagreed on the effectiveness of the new approach. Some research supports the methodology while other studies question its implementation.",
    "The bipartisan commission released its findings after months of study. Members acknowledged areas of agreement and ongoing disagreement on key recommendations.",
    "Policy watchers described the debate as nuanced, noting that the issue defies simple ideological categorization. Both sides presented data to support their positions."
]

CENTER_REASONS = [
    "The article presents multiple perspectives from different political viewpoints. Sources include both liberal and conservative voices, with balanced language and factual reporting.",
    "The piece uses neutral framing and presents competing arguments without clear editorial bias. The tone is measured and avoids loaded language.",
    "The article cites experts across the political spectrum and presents data from multiple sources. No single ideology dominates the narrative.",
    "The coverage balances positive and negative aspects of the issue. Language is objective and avoids emotional appeals or loaded terminology.",
    "The piece presents the debate as complex with legitimate concerns from multiple perspectives. Sources include bipartisan or nonpartisan experts."
]

# Templates for RIGHT bias
RIGHT_TEMPLATES = [
    "Conservatives criticized the proposed regulations as government overreach that threatens individual liberty and free enterprise. Business leaders warned of economic consequences from excessive bureaucracy.",
    "Freedom advocates warned that the new policy undermines personal responsibility and traditional values. They argued that citizens should make their own choices without government interference.",
    "Patriotic Americans expressed concern about the erosion of national sovereignty. Critics said the proposal weakens borders and fails to protect citizens from foreign threats.",
    "Pro-family groups denounced the initiative as harmful to traditional family structures. Religious leaders called for a return to foundational values that have sustained communities.",
    "Fiscal conservatives warned that the spending measure adds to the national debt. They argued for smaller government and responsible fiscal management.",
    "Second Amendment supporters rallied against restrictions on gun rights. They argued that the Constitution guarantees the right to bear arms for self-defense.",
    "Local officials pushed back against federal overreach, asserting state sovereignty. They argued that decisions should be made by elected representatives closer to the people.",
    "Capitalism advocates warned that socialist policies threaten economic prosperity. They argued that free market principles have driven American success.",
    "Traditional values supporters organized to defend cultural institutions they say are under attack. They called for preservation of heritage and family-centered communities.",
    "Law enforcement advocates pushed for stronger public safety measures. They argued that protecting communities requires firm enforcement of existing laws."
]

RIGHT_REASONS = [
    "The article presents conservative viewpoints emphasizing individual liberty, free enterprise, and traditional values. Sources are primarily from conservative think tanks and right-leaning commentators.",
    "The tone strongly favors conservative positions and frames progressive policies as threats to freedom. Language emphasizes personal responsibility and limited government.",
    "The piece uses framing that appeals to patriotism, traditional values, and economic freedom. Sources are uniformly from conservative organizations and spokespeople.",
    "The article uses loaded language like 'government overreach', 'individual liberty', and 'traditional values'. It frames conservative positions as patriotic and progressive ones as dangerous.",
    "The language and source selection strongly favor right-wing perspectives. The article frames government intervention negatively and advocates for free market solutions."
]


# ==============================================================================
# BIAS VALUE GENERATION
# ==============================================================================

def generate_bias_values(dir_type: str, deg_type: str) -> Dict[str, Dict[str, float]]:
    """
    Generate bias probability values.
    
    For the dominant direction (dir_type) and degree (deg_type),
    the value will be >= 0.5. Other values will be < 0.5.
    """
    # Base values with dominant category >= 0.5
    if dir_type == 'L':
        dir_values = {
            'L': round(random.uniform(0.50, 0.85), 2),
            'C': round(random.uniform(0.05, 0.25), 2),
            'R': round(random.uniform(0.05, 0.25), 2)
        }
    elif dir_type == 'C':
        dir_values = {
            'L': round(random.uniform(0.05, 0.25), 2),
            'C': round(random.uniform(0.50, 0.85), 2),
            'R': round(random.uniform(0.05, 0.25), 2)
        }
    else:  # 'R'
        dir_values = {
            'L': round(random.uniform(0.05, 0.25), 2),
            'C': round(random.uniform(0.05, 0.25), 2),
            'R': round(random.uniform(0.50, 0.85), 2)
        }
    
    # Normalize to sum to 1.0
    dir_sum = sum(dir_values.values())
    dir_values = {k: round(v/dir_sum, 2) for k, v in dir_values.items()}
    
    # Degree values
    if deg_type == 'L':
        deg_values = {
            'L': round(random.uniform(0.50, 0.85), 2),
            'M': round(random.uniform(0.05, 0.25), 2),
            'H': round(random.uniform(0.05, 0.25), 2)
        }
    elif deg_type == 'M':
        deg_values = {
            'L': round(random.uniform(0.05, 0.25), 2),
            'M': round(random.uniform(0.50, 0.85), 2),
            'H': round(random.uniform(0.05, 0.25), 2)
        }
    else:  # 'H'
        deg_values = {
            'L': round(random.uniform(0.05, 0.25), 2),
            'M': round(random.uniform(0.05, 0.25), 2),
            'H': round(random.uniform(0.50, 0.85), 2)
        }
    
    # Normalize to sum to 1.0
    deg_sum = sum(deg_values.values())
    deg_values = {k: round(v/deg_sum, 2) for k, v in deg_values.items()}
    
    return {
        'dir': dir_values,
        'deg': deg_values
    }


def get_article_and_reason(dir_type: str, deg_type: str) -> tuple:
    """Get a template article and reason based on bias type."""
    if dir_type == 'L':
        templates = LEFT_TEMPLATES
        reasons = LEFT_REASONS
    elif dir_type == 'C':
        templates = CENTER_TEMPLATES
        reasons = CENTER_REASONS
    else:
        templates = RIGHT_TEMPLATES
        reasons = RIGHT_REASONS
    
    # Select random template
    template = random.choice(templates)
    
    # Expand the template with some variation based on degree
    if deg_type == 'L':
        # Low degree - make it more mild
        modifier = "Some observers noted the moderate tone of the discussion."
    elif deg_type == 'M':
        # Medium degree
        modifier = "The debate highlighted clear differences in perspective among participants."
    else:
        # High degree - make it more intense
        modifier = "The controversy has sparked intense debate among stakeholders."
    
    article = f"{template} {modifier}"
    reason = random.choice(reasons)
    
    return article, reason


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_training_data(
    samples_per_combination: int = 500
) -> List[Dict[str, Any]]:
    """
    Generate balanced training data with all combinations.
    
    Args:
        samples_per_combination: Number of samples for each dir/deg combination
        
    Returns:
        List of training samples
    """
    # All combinations
    directions = ['L', 'C', 'R']
    degrees = ['L', 'M', 'H']
    
    total_combinations = len(directions) * len(degrees)
    total_samples = samples_per_combination * total_combinations
    
    print(f"Generating {total_samples} training samples...")
    print(f"  - {len(directions)} directions x {len(degrees)} degrees = {total_combinations} combinations")
    print(f"  - {samples_per_combination} samples per combination")
    
    data = []
    
    for direction in directions:
        for degree in degrees:
            print(f"Generating {samples_per_combination} samples for dir={direction}, deg={degree}...")
            
            for i in range(samples_per_combination):
                # Generate bias values
                bias = generate_bias_values(direction, degree)
                
                # Get article and reason
                article, reason = get_article_and_reason(direction, degree)
                
                # Create the label
                label = {
                    'dir': bias['dir'],
                    'deg': bias['deg'],
                    'reason': reason
                }
                
                # Create the training sample
                sample = {
                    'article': article,
                    'label': label
                }
                
                data.append(sample)
    
    # Shuffle the data
    random.shuffle(data)
    
    return data


def validate_bias_values(data: List[Dict]) -> Dict[str, Any]:
    """Validate that generated data has proper bias values >= 0.5 for dominant category."""
    stats = {
        'total': len(data),
        'combinations': {}
    }
    
    for sample in data:
        # Find dominant dir
        dir_vals = sample['label']['dir']
        dominant_dir = max(dir_vals, key=dir_vals.get)
        
        # Find dominant deg
        deg_vals = sample['label']['deg']
        dominant_deg = max(deg_vals, key=deg_vals.get)
        
        key = f"{dominant_dir}-{dominant_deg}"
        if key not in stats['combinations']:
            stats['combinations'][key] = 0
        stats['combinations'][key] += 1
        
        # Verify dominant value >= 0.5
        if dir_vals[dominant_dir] < 0.5:
            print(f"Warning: dir {dominant_dir} has value {dir_vals[dominant_dir]} < 0.5")
        if deg_vals[dominant_deg] < 0.5:
            print(f"Warning: deg {dominant_deg} has value {deg_vals[dominant_deg]} < 0.5")
    
    return stats


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic bias training data'
    )
    parser.add_argument(
        '-o', '--output',
        default='train_synthetic.json',
        help='Output file path (default: train_synthetic.json)'
    )
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=500,
        help='Number of samples per combination (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("Synthetic Bias Training Data Generator")
    print("=" * 60)
    
    # Generate data
    data = generate_training_data(samples_per_combination=args.samples)
    
    # Validate
    print("\nValidating generated data...")
    stats = validate_bias_values(data)
    
    print("\nStatistics:")
    print(f"  Total samples: {stats['total']}")
    print("  Samples per combination:")
    for combo, count in sorted(stats['combinations'].items()):
        print(f"    {combo}: {count}")
    
    # Save to file
    output_path = Path(args.output)
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(data)} training samples")
    print(f"✓ Saved to {output_path}")
    print("\nTo train the model, run:")
    print(f"  ./run_training.sh --data {args.output}")


if __name__ == '__main__':
    main()
