#!/usr/bin/env python
import os
import re
import argparse
from collections import defaultdict
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"


def normalize_line(line):
    """Remove all non-alphanumeric characters for comparison"""
    normalized = re.sub(r'[^a-zA-Z0-9]', '', line.strip().lower())
    # Keep the normalized version even if empty - we'll filter by original line length
    return normalized


def find_repeated_lines(articles):
    """
    Find lines that appear in multiple articles from the same source
    Returns a set of normalized lines that should be removed
    """
    # Track which articles contain each normalized line
    line_to_articles = defaultdict(set)
    # Track original line text for reporting
    line_to_original = {}
    
    articles_with_text = 0
    
    for article in articles:
        article_id = str(article['_id'])
        # Only process the 'article' field
        text = article.get('article', '')
        
        if not text:
            continue
        
        articles_with_text += 1
        lines = text.split('\n')
        # Track unique normalized lines seen in THIS article
        seen_normalized = set()
        
        for line in lines:
            stripped = line.strip()
            # Skip truly empty lines
            if not stripped:
                continue
            
            # Skip very short lines (less than 10 characters) to avoid false positives
            if len(stripped) < 10:
                continue
            
            normalized = normalize_line(stripped)
            
            # Skip if normalization resulted in empty or very short string (less than 5 chars)
            if len(normalized) < 5:
                continue
            
            # Only count each unique normalized line once per article
            if normalized not in seen_normalized:
                line_to_articles[normalized].add(article_id)
                seen_normalized.add(normalized)
                # Store an example of the original line
                if normalized not in line_to_original:
                    line_to_original[normalized] = stripped
    
    print(f"Processed {articles_with_text} articles with 'article' field content")
    print(f"Found {len(line_to_articles)} unique line patterns")
    
    # Find lines that appear in MORE than one article
    repeated_lines = {
        norm_line for norm_line, article_ids in line_to_articles.items()
        if len(article_ids) > 1
    }
    
    return repeated_lines, line_to_articles, line_to_original


def remove_repeated_lines(text, repeated_lines):
    """Remove lines that match the repeated lines set"""
    if not text:
        return text, 0
    
    lines = text.split('\n')
    filtered_lines = []
    removed_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Keep empty lines as-is
        if not stripped:
            filtered_lines.append(line)
            continue
        
        # Skip very short lines from removal check
        if len(stripped) < 10:
            filtered_lines.append(line)
            continue
        
        normalized = normalize_line(stripped)
        
        # Skip very short normalized lines from removal check
        if len(normalized) < 5:
            filtered_lines.append(line)
            continue
        
        if normalized in repeated_lines:
            removed_count += 1
            # Don't add this line to filtered_lines
        else:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines), removed_count


def process_source(collection, source, safe_mode=True):
    """Process all articles from a specific source"""
    print(f"\n{'='*80}")
    print(f"Processing source: {source}")
    print(f"{'='*80}")
    
    # Find all articles from this source
    query = {'source': source}
    articles = list(collection.find(query))
    
    if not articles:
        print(f"No articles found for source: {source}")
        return {
            'source': source,
            'total_articles': 0,
            'articles_modified': 0,
            'total_lines_removed': 0,
            'repeated_lines_count': 0
        }
    
    print(f"Found {len(articles)} articles from {source}")
    
    # Find repeated lines
    repeated_lines, line_to_articles, line_to_original = find_repeated_lines(articles)
    
    print(f"Found {len(repeated_lines)} repeated line patterns")
    
    # Show sample of repeated lines
    if repeated_lines:
        print("\nSample of repeated lines (first 10):")
        sorted_lines = sorted(repeated_lines, key=lambda x: len(line_to_articles[x]), reverse=True)
        for i, norm_line in enumerate(sorted_lines[:10]):
            article_count = len(line_to_articles[norm_line])
            original = line_to_original[norm_line]
            # Truncate long lines for display
            display_text = original[:100] + '...' if len(original) > 100 else original
            print(f"  {i+1}. '{display_text}' (in {article_count} articles)")
    
    # Process each article
    articles_modified = 0
    total_lines_removed = 0
    
    print(f"\n{'Processing articles...' if not safe_mode else 'SAFE MODE: Simulating changes...'}")
    
    articles_with_field = 0
    articles_without_field = 0
    
    for article in articles:
        article_id = article['_id']
        
        # Only process the 'article' field
        text = article.get('article', '')
        
        if not text:
            articles_without_field += 1
            continue
        
        articles_with_field += 1
        cleaned_text, removed_count = remove_repeated_lines(text, repeated_lines)
        
        if removed_count > 0:
            articles_modified += 1
            total_lines_removed += removed_count
            
            if not safe_mode:
                collection.update_one(
                    {'_id': article_id},
                    {'$set': {'article': cleaned_text}}
                )
    
    print(f"Articles with 'article' field: {articles_with_field}")
    print(f"Articles without 'article' field: {articles_without_field}")
    print(f"Articles modified: {articles_modified}/{articles_with_field}")
    print(f"Total lines removed: {total_lines_removed}")
    
    return {
        'source': source,
        'total_articles': len(articles),
        'articles_modified': articles_modified,
        'total_lines_removed': total_lines_removed,
        'repeated_lines_count': len(repeated_lines)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Remove repeated lines (headers/footers) from RSS articles'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Specific source to process (defaults to all sources)'
    )
    parser.add_argument(
        '--safe',
        action='store_true',
        help='Safe mode: only report changes without making them'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("RSS Article Header/Footer Removal Tool")
    print("="*80)
    print(f"Mode: {'SAFE MODE (no changes will be made)' if args.safe else 'LIVE MODE (will modify database)'}")
    print(f"Target source: {args.source if args.source else 'ALL SOURCES'}")
    
    # Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[MONGO_DB]
        collection = db[MONGO_COLL]
        print(f"✓ Connected to MongoDB: {MONGO_DB}.{MONGO_COLL}")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        return
    
    # Get list of sources to process
    if args.source:
        sources = [args.source]
    else:
        sources = collection.distinct('source')
        print(f"\nFound {len(sources)} unique sources in database")
    
    # Process each source
    results = []
    for source in sources:
        result = process_source(collection, source, safe_mode=args.safe)
        results.append(result)
    
    # Generate final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    
    total_articles = sum(r['total_articles'] for r in results)
    total_modified = sum(r['articles_modified'] for r in results)
    total_lines = sum(r['total_lines_removed'] for r in results)
    
    print(f"\nSummary:")
    print(f"  Sources processed: {len(results)}")
    print(f"  Total articles: {total_articles}")
    print(f"  Articles modified: {total_modified}")
    print(f"  Total lines removed: {total_lines}")
    
    print(f"\nBreakdown by source:")
    print(f"{'Source':<40} {'Articles':<12} {'Modified':<12} {'Lines Removed':<15} {'Repeated Patterns'}")
    print("-"*100)
    
    for result in sorted(results, key=lambda x: x['total_lines_removed'], reverse=True):
        print(f"{result['source']:<40} {result['total_articles']:<12} "
              f"{result['articles_modified']:<12} {result['total_lines_removed']:<15} "
              f"{result['repeated_lines_count']}")
    
    if args.safe:
        print(f"\n⚠ SAFE MODE: No changes were made to the database")
        print(f"Run without --safe flag to apply these changes")
    else:
        print(f"\n✓ Changes have been applied to the database")
    
    # Close connection
    client.close()
    print("\n" + "="*80)


if __name__ == "__main__":
    main()