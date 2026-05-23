#!/usr/bin/env python3
"""
MongoDB violent rhetoric extraction and analysis by political orientation.
Compares prevalence of violent language in left vs right news sources.
"""

import os
import json
import re
from pymongo import MongoClient
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import sys

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://root:example@localhost:27017')

# Explicit violent keywords
EXPLICIT_VIOLENCE = [
    'kill', 'killed', 'killing', 'murder', 'murdered', 'murdering',
    'massacre', 'slaughter', 'slaughtered', 'genocide', 'genocidal',
    'attack', 'attacked', 'assault', 'assaulted',
    'shoot', 'shot', 'shooting', 'gun', 'bomb', 'explosion',
    'assassinate', 'assassinated', 'executed', 'execution',
    'brutal', 'brutality', 'torture', 'tortured'
]

# Implicit rhetorical violence (from academic research)
IMPLICIT_VIOLENCE = [
    'enemy', 'enemies', 'vermin', 'invaders', 'invasion',
    'existential threat', 'annihilate', 'eliminate', 'eradicate',
    'destroy', 'obliterate', 'wipe out', 'get rid of',
    'thugs', 'criminals', 'radicals', 'extremists',
    'blood', 'sacrifice', 'fight', 'war', 'battle'
]

VIOLENT_WORDS = EXPLICIT_VIOLENCE + IMPLICIT_VIOLENCE

# Source bias classification based on MongoDB analysis
LEFT_SOURCES = ['et-chinahuman', 'voa-intnl', 'et-la', 'et-me', 'voa-africa']
CENTER_SOURCES = ['abc-int', 'abc-pol', 'abc-top', 'abc-us', 'bbc-me', 'bbc-pol', 
                  'bbc-top', 'bbc-world', 'cnbc-asia', 'cnbc-europe', 'cnbc-politics',
                  'cnbc-top', 'cnbc-us', 'cnbc-world', 'dn-egypt', 'dnaindia', 
                  'dw', 'dw-eu', 'dw-world', 'egyptian-independant', 'egyptian-streets',
                  'et-africa', 'et-americas', 'et-asiapac', 'et-aus', 'et-canada',
                  'et-europe', 'et-intl', 'et-uk', 'et-usf', 'et-uspol',
                  'f24-africa', 'f24-americas', 'f24-asiapac', 'f24-euro', 
                  'f24-france', 'f24-me', 'japantimes', 'japantoday', 'mainichi',
                  'mehr-news', 'ndtv', 'nyt-africa', 'nyt-americas', 'nyt-asia',
                  'nyt-europe', 'nyt-middleeast', 'nyt-world', 'saudi-gaz',
                  'straits-times', 'syriadirect', 'syrian-obs', 'tasnim-news',
                  'tass', 'thehindu', 'voa-china', 'voa-easia', 'voa-europe',
                  'voa-im', 'voa-issues', 'voa-meast', 'voa-sncentasia',
                  'voa-theissue', 'voa-ukraine', 'voa-usa', 'waustralian-int']
RIGHT_SOURCES = ['breitbart', 'fox-latest', 'fox-politics', 'fox-us', 'fox-world',
                 'gbnews', 'gbnews-op', 'gbnews-pol', 'arutz-sheva', 'nyp-pol', 'nyp-us']

LEFT_SOURCES = set(LEFT_SOURCES)
CENTER_SOURCES = set(CENTER_SOURCES)
RIGHT_SOURCES = set(RIGHT_SOURCES)


def classify_source(source):
    if source in LEFT_SOURCES:
        return 'L'
    elif source in RIGHT_SOURCES:
        return 'R'
    else:
        return 'C'


def contains_violence(text):
    """Check if text contains violent keywords"""
    text_lower = text.lower()
    for word in VIOLENT_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False


def extract_violent_terms(text):
    """Extract which violent terms appear in text"""
    text_lower = text.lower()
    found = []
    for word in VIOLENT_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            found.append(word)
    return found


def extract_entities(doc):
    """Extract PERSON and NORP entities from NER field"""
    ner = doc.get('ner', {})
    if not ner:
        return [], []
    
    entities = ner.get('entities', [])
    persons = []
    norps = []
    
    for e in entities:
        label = e.get('label', '')
        text = e.get('text', '')
        if label == 'PERSON':
            persons.append(text)
        elif label == 'NORP':
            norps.append(text)
    
    return persons, norps


def analyze():
    print("=" * 70)
    print("VIOLENT RHETORIC ANALYSIS BY POLITICAL ORIENTATION")
    print("=" * 70)
    
    client = MongoClient(MONGO_URI)
    coll = client['rssnews']['articles']
    
    # Get total counts by source classification
    print("\n[1] Analyzing source distribution...")
    
    pipeline = [
        {'$match': {'bias': {'$exists': True, '$ne': {}}}},
        {'$project': {'source': 1, 'bias': 1}},
        {'$limit': 100000}  # Sample for performance
    ]
    
    docs = list(coll.aggregate(pipeline))
    print(f"  Processed {len(docs)} articles with bias")
    
    # Classify by orientation
    orient_stats = {'L': 0, 'C': 0, 'R': 0}
    orient_sources = {'L': set(), 'C': set(), 'R': set()}
    
    for doc in docs:
        source = doc.get('source', '')
        orient = classify_source(source)
        orient_stats[orient] += 1
        orient_sources[orient].add(source)
    
    print(f"  Left sources: {len(orient_sources['L'])} ({orient_stats['L']} articles)")
    print(f"  Center sources: {len(orient_sources['C'])} ({orient_stats['C']} articles)")
    print(f"  Right sources: {len(orient_sources['R'])} ({orient_stats['R']} articles)")
    
    # Now analyze violent content by orientation
    print("\n[2] Searching for violent rhetoric...")
    
    # Query for violent articles
    violent_query = {
        'bias': {'$exists': True, '$ne': {}},
        '$or': [
            {'article': {'$regex': '|'.join(VIOLENT_WORDS), '$options': 'i'}}
        ]
    }
    
    violent_docs = list(coll.find(violent_query).limit(20000))
    print(f"  Found {len(violent_docs)} articles with violent keywords")
    
    # Group by orientation
    violent_by_orient = {'L': [], 'C': [], 'R': []}
    violent_term_counts = {'L': Counter(), 'C': Counter(), 'R': Counter()}
    targets_by_orient = {'L': Counter(), 'C': Counter(), 'R': Counter()}
    
    for doc in violent_docs:
        source = doc.get('source', '')
        orient = classify_source(source)
        
        article_text = doc.get('article', '') + ' ' + doc.get('title', '')
        
        violent_by_orient[orient].append(doc)
        
        # Count violent terms
        terms = extract_violent_terms(article_text)
        for t in terms:
            violent_term_counts[orient][t] += 1
        
        # Extract targets
        persons, norps = extract_entities(doc)
        for p in persons[:5]:
            targets_by_orient[orient][p] += 1
        for n in norps[:3]:
            targets_by_orient[orient][n] += 1
    
    print("\n[3] Statistics by Political Orientation:")
    print("-" * 50)
    
    stats = {}
    for orient in ['L', 'C', 'R']:
        orient_label = {'L': 'LEFT', 'C': 'CENTER', 'R': 'RIGHT'}[orient]
        total_articles = orient_stats[orient]
        violent_count = len(violent_by_orient[orient])
        
        if total_articles > 0:
            pct = (violent_count / total_articles) * 100
        else:
            pct = 0
        
        stats[orient] = {
            'label': orient_label,
            'total': total_articles,
            'violent': violent_count,
            'percentage': pct,
            'top_terms': violent_term_counts[orient].most_common(10),
            'top_targets': targets_by_orient[orient].most_common(10)
        }
        
        print(f"\n  {orient_label}:")
        print(f"    Total articles: {total_articles}")
        print(f"    Violent articles: {violent_count}")
        print(f"    Violent %: {pct:.2f}%")
        print(f"    Top violent terms: {stats[orient]['top_terms'][:5]}")
        print(f"    Top targets: {stats[orient]['top_targets'][:5]}")
    
    # Save results
    output = {
        'generated': datetime.now().isoformat(),
        'total_articles': len(docs),
        'total_violent': len(violent_docs),
        'stats': stats,
        'orient_sources': {k: list(v) for k, v in orient_sources.items()}
    }
    
    with open('/Users/kyle/hub/propaganda/docs/violent_rhetoric/data/mongo_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[✓] Saved to data/mongo_results.json")
    
    # Generate .vec file
    generate_vec_file(violent_by_orient)
    
    return output


def generate_vec_file(violent_by_orient):
    """Generate .vec file with violent articles"""
    
    vec_lines = [
        "=" * 80,
        "CATEGORY: Violent Rhetoric Analysis by Political Orientation",
        f"Total violent articles: {sum(len(v) for v in violent_by_orient.values())}",
        "=" * 80,
        ""
    ]
    
    for orient in ['L', 'C', 'R']:
        orient_label = {'L': 'LEFT', 'C': 'CENTER', 'R': 'RIGHT'}[orient]
        docs = violent_by_orient[orient]
        
        vec_lines.extend([
            f"{'=' * 80}",
            f"POLITICAL ORIENTATION: {orient_label}",
            f"Articles: {len(docs)}",
            f"{'=' * 80}",
            ""
        ])
        
        for doc in docs[:50]:  # Limit to 50 per orientation
            title = doc.get('title', '')[:80]
            source = doc.get('source', 'unknown')
            published = doc.get('published', 'unknown')
            
            article_text = doc.get('article', '')[:200]
            
            vec_lines.extend([
                "-" * 40,
                f"Title: {title}",
                f"Source: {source} ({orient_label})",
                f"Published: {published}",
                f"Excerpt: {article_text}...",
                ""
            ])
    
    vec_path = '/Users/kyle/hub/propaganda/docs/violent_rhetoric/output/violent_rhetoric.vec'
    with open(vec_path, 'w') as f:
        f.write('\n'.join(vec_lines))
    
    print(f"[✓] .vec: {vec_path}")


if __name__ == '__main__':
    analyze()