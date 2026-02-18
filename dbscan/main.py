#!/usr/bin/env python

"""
Main entry point for the news article categorization system.
Handles CLI arguments and orchestrates the categorization workflow.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from file_handler import load_articles, export_results
from embeddings import generate_embeddings
from chroma_manager import initialize_chroma, add_articles_to_chroma
from categorizer import cluster_articles, generate_category_names, format_results
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    input_path: str,
    output_path: str,
    min_cluster_size: int,
    similarity_threshold: float,
    persist_dir: str
) -> None:
    """
    Execute the news article categorization pipeline.
    
    Args:
        input_path: Path to input TSV file
        output_path: Path to output JSON file
        min_cluster_size: Minimum articles required to form a category
        similarity_threshold: Cosine similarity threshold for grouping
        persist_dir: ChromaDB persistence directory
    """
    try:
        # Step 1: Load articles
        logger.info(f"Loading articles from {input_path}")
        articles_df = load_articles(input_path)
        logger.info(f"Loaded {len(articles_df)} articles")
        
        if articles_df.empty:
            logger.error("No articles found in input file")
            return
        
        # Step 2: Generate embeddings
        logger.info("Generating embeddings for article titles")
        titles = articles_df['title'].tolist()
        embeddings = generate_embeddings(titles)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        # Step 3: Initialize ChromaDB
        logger.info(f"Initializing ChromaDB in {persist_dir}")
        collection = initialize_chroma(
            collection_name=config.COLLECTION_NAME,
            persist_dir=persist_dir
        )
        
        # Step 4: Add articles to ChromaDB
        logger.info("Adding articles to ChromaDB")
        add_articles_to_chroma(collection, articles_df, embeddings)
        
        # Step 5: Cluster articles
        logger.info("Clustering articles")
        cluster_labels = cluster_articles(
            embeddings,
            method="dbscan",
            min_samples=min_cluster_size,
            eps=1 - similarity_threshold  # Convert similarity to distance
        )
        
        # Step 6: Generate category names
        logger.info("Generating category names")
        categories = generate_category_names(
            articles_df,
            cluster_labels,
            min_cluster_size
        )
        
        # Step 7: Format and display results
        results = format_results(categories, articles_df, cluster_labels)
        
        # Console output
        print("\n" + "="*60)
        print("CATEGORIZATION RESULTS")
        print("="*60 + "\n")
        
        for cat_id, cat_data in results['categories'].items():
            print(f"Category {cat_id}: \"{cat_data['category_name']}\" "
                  f"({cat_data['article_count']} articles)")
            for article in cat_data['articles']:
                print(f"  - {article['article_id']}: {article['title']}")
            print()
        
        if results['uncategorized']:
            print(f"Uncategorized: {len(results['uncategorized'])} articles")
            for article in results['uncategorized']:
                print(f"  - {article['article_id']}: {article['title']}")
        
        # Step 8: Export results
        logger.info(f"Exporting results to {output_path}")
        export_results(results, output_path)
        logger.info("Categorization complete!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error during categorization: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Categorize news articles using semantic similarity"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input TSV file"
    )
    parser.add_argument(
        "--output",
        default="categories.json",
        help="Path to output JSON file (default: categories.json)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum articles per category (default: 2)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold (default: 0.75)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="ChromaDB storage directory (default: ./chroma_db)"
    )
    
    args = parser.parse_args()
    
    main(
        input_path=args.input,
        output_path=args.output,
        min_cluster_size=args.min_cluster_size,
        similarity_threshold=args.similarity_threshold,
        persist_dir=args.persist_dir
    )
