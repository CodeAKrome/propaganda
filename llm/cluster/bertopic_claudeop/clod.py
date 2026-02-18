#!/usr/bin/env python
"""
BERTopic News Article Title Clustering
Input: Tab-delimited file with article_id and title
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import argparse
import os
from datetime import datetime


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load tab-delimited file with article_id and title.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        DataFrame with article_id and title columns
    """
    print(f"Loading data from {file_path}...")
    
    df = pd.read_csv(
        file_path, 
        sep='\t', 
        header=None, 
        names=['article_id', 'title'],
        dtype={'article_id': str, 'title': str}
    )
    
    # Clean data
    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str).str.strip()
    df = df[df['title'].str.len() > 0]
    
    print(f"Loaded {len(df)} articles")
    return df


def create_bertopic_model(
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 10,
    nr_topics: str = "auto",
    n_neighbors: int = 15,
    n_components: int = 5,
    min_cluster_size: int = 10,
    language: str = "english"
) -> BERTopic:
    """
    Create and configure BERTopic model with custom parameters.
    
    Args:
        embedding_model: Sentence transformer model name
        min_topic_size: Minimum number of documents per topic
        nr_topics: Number of topics ("auto" or integer)
        n_neighbors: UMAP n_neighbors parameter
        n_components: UMAP n_components parameter
        min_cluster_size: HDBSCAN min_cluster_size parameter
        language: Language for stop words
        
    Returns:
        Configured BERTopic model
    """
    print("Initializing BERTopic model...")
    
    # Embedding model
    sentence_model = SentenceTransformer(embedding_model)
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Vectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words=language,
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics if nr_topics == "auto" else int(nr_topics),
        verbose=True,
        calculate_probabilities=True
    )
    
    return topic_model


def run_clustering(
    df: pd.DataFrame, 
    topic_model: BERTopic
) -> tuple:
    """
    Run BERTopic clustering on article titles.
    
    Args:
        df: DataFrame with titles
        topic_model: Configured BERTopic model
        
    Returns:
        Tuple of (topics, probabilities, topic_model)
    """
    print("Running topic modeling...")
    titles = df['title'].tolist()
    
    # Fit the model
    topics, probabilities = topic_model.fit_transform(titles)
    
    print(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
    print(f"Outliers (topic -1): {topics.count(-1)} documents")
    
    return topics, probabilities, topic_model


def save_results(
    df: pd.DataFrame,
    topics: list,
    probabilities: np.ndarray,
    topic_model: BERTopic,
    output_dir: str
):
    """
    Save clustering results to files.
    
    Args:
        df: Original DataFrame
        topics: Topic assignments
        probabilities: Topic probabilities
        topic_model: Trained BERTopic model
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save article-topic assignments
    results_df = df.copy()
    results_df['topic'] = topics
    results_df['probability'] = probabilities.max(axis=1) if len(probabilities.shape) > 1 else probabilities
    
    # Get topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = dict(zip(topic_info['Topic'], topic_info['Name']))
    results_df['topic_label'] = results_df['topic'].map(topic_labels)
    
    output_file = os.path.join(output_dir, f"article_clusters_{timestamp}.tsv")
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved article clusters to {output_file}")
    
    # 2. Save topic information
    topic_info_file = os.path.join(output_dir, f"topic_info_{timestamp}.tsv")
    topic_info.to_csv(topic_info_file, sep='\t', index=False)
    print(f"Saved topic info to {topic_info_file}")
    
    # 3. Save detailed topic words
    topic_words_data = []
    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            words = topic_model.get_topic(topic_id)
            for word, score in words:
                topic_words_data.append({
                    'topic': topic_id,
                    'word': word,
                    'score': score
                })
    
    topic_words_df = pd.DataFrame(topic_words_data)
    topic_words_file = os.path.join(output_dir, f"topic_words_{timestamp}.tsv")
    topic_words_df.to_csv(topic_words_file, sep='\t', index=False)
    print(f"Saved topic words to {topic_words_file}")
    
    # 4. Save the model
    model_file = os.path.join(output_dir, f"bertopic_model_{timestamp}")
    topic_model.save(model_file, serialization="pickle")
    print(f"Saved model to {model_file}")
    
    return results_df


def generate_visualizations(
    topic_model: BERTopic,
    titles: list,
    topics: list,
    output_dir: str
):
    """
    Generate and save BERTopic visualizations.
    
    Args:
        topic_model: Trained BERTopic model
        titles: List of article titles
        topics: Topic assignments
        output_dir: Output directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Topic visualization
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, f"topic_visualization_{timestamp}.html"))
        print("Saved topic visualization")
    except Exception as e:
        print(f"Could not create topic visualization: {e}")
    
    try:
        # Bar chart of topics
        fig = topic_model.visualize_barchart(top_n_topics=15)
        fig.write_html(os.path.join(output_dir, f"topic_barchart_{timestamp}.html"))
        print("Saved topic barchart")
    except Exception as e:
        print(f"Could not create barchart: {e}")
    
    try:
        # Topic hierarchy
        fig = topic_model.visualize_hierarchy()
        fig.write_html(os.path.join(output_dir, f"topic_hierarchy_{timestamp}.html"))
        print("Saved topic hierarchy")
    except Exception as e:
        print(f"Could not create hierarchy: {e}")
    
    try:
        # Heatmap of topic similarity
        fig = topic_model.visualize_heatmap()
        fig.write_html(os.path.join(output_dir, f"topic_heatmap_{timestamp}.html"))
        print("Saved topic heatmap")
    except Exception as e:
        print(f"Could not create heatmap: {e}")
    
    try:
        # Document visualization (may be slow for large datasets)
        if len(titles) <= 10000:
            embeddings = topic_model._extract_embeddings(titles)
            fig = topic_model.visualize_documents(titles, embeddings=embeddings)
            fig.write_html(os.path.join(output_dir, f"document_visualization_{timestamp}.html"))
            print("Saved document visualization")
    except Exception as e:
        print(f"Could not create document visualization: {e}")


def print_topic_summary(topic_model: BERTopic, top_n: int = 10):
    """
    Print a summary of discovered topics.
    
    Args:
        topic_model: Trained BERTopic model
        top_n: Number of top topics to display
    """
    print("\n" + "="*60)
    print("TOPIC SUMMARY")
    print("="*60)
    
    topic_info = topic_model.get_topic_info()
    
    for idx, row in topic_info.head(top_n + 1).iterrows():
        topic_id = row['Topic']
        count = row['Count']
        name = row['Name']
        
        if topic_id == -1:
            print(f"\nTopic {topic_id} (Outliers): {count} documents")
        else:
            print(f"\nTopic {topic_id}: {count} documents")
            print(f"  Label: {name}")
            
            # Get top words
            words = topic_model.get_topic(topic_id)[:5]
            word_str = ", ".join([f"{w[0]} ({w[1]:.3f})" for w in words])
            print(f"  Top words: {word_str}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Cluster news article titles using BERTopic'
    )
    parser.add_argument(
        'input_file',
        help='Tab-delimited input file (article_id\\ttitle)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./bertopic_output',
        help='Output directory (default: ./bertopic_output)'
    )
    parser.add_argument(
        '--embedding-model', '-e',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--min-topic-size', '-m',
        type=int,
        default=10,
        help='Minimum topic size (default: 10)'
    )
    parser.add_argument(
        '--nr-topics', '-n',
        default='auto',
        help='Number of topics or "auto" (default: auto)'
    )
    parser.add_argument(
        '--min-cluster-size', '-c',
        type=int,
        default=10,
        help='HDBSCAN min cluster size (default: 10)'
    )
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualizations'
    )
    parser.add_argument(
        '--language', '-l',
        default='english',
        help='Language for stop words (default: english)'
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input_file)
    
    if len(df) < args.min_topic_size:
        print(f"Error: Not enough documents ({len(df)}) for min_topic_size ({args.min_topic_size})")
        return
    
    # Create model
    topic_model = create_bertopic_model(
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        min_cluster_size=args.min_cluster_size,
        language=args.language
    )
    
    # Run clustering
    topics, probabilities, topic_model = run_clustering(df, topic_model)
    
    # Print summary
    print_topic_summary(topic_model)
    
    # Save results
    save_results(df, topics, probabilities, topic_model, args.output_dir)
    
    # Generate visualizations
    if not args.no_visualizations:
        print("\nGenerating visualizations...")
        generate_visualizations(
            topic_model, 
            df['title'].tolist(), 
            topics, 
            args.output_dir
        )
    
    print("\nClustering complete!")


if __name__ == "__main__":
    main()
