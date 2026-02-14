#!/usr/bin/env python3
"""
Convert cluster_titles.json to .vec format by fetching article data from MongoDB.

Usage:
  python convert_to_vec.py <input_json> <output_vec>
"""

import os
import sys
import json
from pymongo import MongoClient
from bson.objectid import ObjectId


def connect_to_mongo():
    """Initialize MongoDB connection using environment variables."""
    mongo_user = os.getenv("MONGO_USER")
    mongo_pass = os.getenv("MONGO_PASS")

    if not mongo_user or not mongo_pass:
        raise ValueError(
            "MONGO_USER and MONGO_PASS environment variables must be set"
        )

    uri = f"mongodb://{mongo_user}:{mongo_pass}@localhost:27017"
    client = MongoClient(uri)
    db = client["rssnews"]
    collection = db["articles"]
    return collection


def format_entities(ner_data):
    """Format entities from NER data into the required text format."""
    if not ner_data or 'entities' not in ner_data:
        return ""
    
    # Group entities by label
    entities_by_type = {}
    for entity in ner_data['entities']:
        label = entity.get('label', '')
        text = entity.get('text', '')
        if label and text:
            if label not in entities_by_type:
                entities_by_type[label] = []
            # Avoid duplicates
            if text not in entities_by_type[label]:
                entities_by_type[label].append(text)
    
    # Format as lines sorted by entity type
    lines = []
    for entity_type in sorted(entities_by_type.keys()):
        values = entities_by_type[entity_type]
        if values:
            values_str = ", ".join(values)
            lines.append(f"{entity_type}: {values_str}")
    
    return "\n".join(lines)


def format_bias(bias_data):
    """Format bias data as a JSON string."""
    if not bias_data:
        return "{}"
    
    # If bias_data is already a string, validate and return it
    if isinstance(bias_data, str):
        try:
            # Parse to validate it's valid JSON, then return compact version
            parsed = json.loads(bias_data)
            return json.dumps(parsed, separators=(',', ':'))
        except json.JSONDecodeError:
            return "{}"
    
    # If it's a dict, convert to compact JSON
    return json.dumps(bias_data, separators=(',', ':'))


def fetch_article_data(collection, article_id):
    """Fetch article data from MongoDB."""
    try:
        doc = collection.find_one({"_id": ObjectId(article_id)})
        if doc is None:
            print(f"Warning: Document with ID '{article_id}' not found", file=sys.stderr)
            return None
        
        # Extract published date - handle MongoDB date object
        published = doc.get('published', '')
        if isinstance(published, dict) and '$date' in published:
            # It's a MongoDB extended JSON date
            published = published['$date']
        # Convert to ISO format string if it's not already
        if hasattr(published, 'isoformat'):
            published = published.isoformat()
        elif isinstance(published, str):
            # Remove 'Z' and milliseconds if present to match expected format
            published = published.replace('.000Z', '').replace('Z', '')
        
        # Extract required fields
        return {
            'id': article_id,
            'title': doc.get('title', ''),
            'published': published,
            'source': doc.get('source', ''),
            'bias': doc.get('bias', {}),
            'ner': doc.get('ner', {}),  # Changed from entities to ner
            'text': doc.get('article', '')  # Changed from text to article
        }
    except Exception as e:
        print(f"Error fetching article {article_id}: {e}", file=sys.stderr)
        return None


def write_category_header(f, category_name, category_id, article_count):
    """Write a category header in the .vec file."""
    f.write("=" * 80 + "\n")
    f.write(f"CATEGORY {category_id}: {category_name}\n")
    f.write(f"Articles: {article_count}\n")
    f.write("=" * 80 + "\n")
    f.write("\n")


def write_vec_entry(f, article_data):
    """Write a single article entry in .vec format."""
    f.write("---\n")
    f.write(f"ID: {article_data['id']}\n")
    f.write(f"Title: {article_data['title']}\n")
    f.write(f"Published: {article_data['published']}\n")
    f.write(f"Source: {article_data['source']}\n")
    f.write(f"Bias: {format_bias(article_data['bias'])}\n")
    
    # Write entities section from NER data
    entities_text = format_entities(article_data['ner'])
    if entities_text:
        f.write("<entities>\n")
        f.write(entities_text + "\n")
        f.write("</entities>\n")
    else:
        f.write("<entities>\n</entities>\n")
    
    # Write text
    f.write(f"Text: {article_data['text']}\n")
    f.write("\n")


def sanitize_filename(filename):
    """Sanitize category name for use as filename."""
    # Replace problematic characters with underscores
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    # Remove multiple consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_to_vec.py <input_json> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input JSON
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Connect to MongoDB
    print("Connecting to MongoDB...")
    collection = connect_to_mongo()
    
    # Process articles
    total_articles = sum(cat['article_count'] for cat in data['categories'])
    total_categories = len(data['categories'])
    print(f"Processing {total_categories} categories with {total_articles} articles...")
    print(f"Output directory: {output_dir}\n")
    
    total_processed = 0
    total_skipped = 0
    created_files = []
    
    for category in data['categories']:
        category_name = category['category_name']
        category_id = category['category_id']
        
        # Create filename from category name
        safe_name = sanitize_filename(category_name)
        output_file = os.path.join(output_dir, f"{category_id:03d}_{safe_name}.vec")
        
        print(f"Category {category_id}: {category_name}")
        print(f"  Output file: {output_file}")
        
        category_processed = 0
        category_skipped = 0
        
        with open(output_file, 'w') as outf:
            # Write category header at the top of the file
            write_category_header(
                outf,
                category_name,
                category_id,
                category['article_count']
            )
            
            for article in category['articles']:
                article_id = article['article_id']
                
                # Fetch article data from MongoDB
                article_data = fetch_article_data(collection, article_id)
                
                if article_data:
                    write_vec_entry(outf, article_data)
                    category_processed += 1
                    total_processed += 1
                else:
                    category_skipped += 1
                    total_skipped += 1
        
        print(f"  Articles: {category_processed} processed, {category_skipped} skipped\n")
        created_files.append(output_file)
    
    print("=" * 80)
    print(f"Complete!")
    print(f"Total categories: {total_categories}")
    print(f"Total articles processed: {total_processed}")
    print(f"Total articles skipped: {total_skipped}")
    print(f"Files created: {len(created_files)}")
    print(f"\nOutput files in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
