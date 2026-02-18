#!/usr/bin/env python

"""
BERTopic clustering for news article titles
Input:  tab-delimited file ->  article_id   title
Output:
    - <prefix>_assignments.tsv     →  one row per article with assigned topic + probability + topic name
    - <prefix>_topics.tsv          →  topic summary (size, top words, auto-generated name)
"""

import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


def main():
    parser = argparse.ArgumentParser(description="Cluster news article titles with BERTopic")
    parser.add_argument("input_file",
                        help="Input TSV file: article_id\\ttitle")
    parser.add_argument("--output_prefix",
                        default="bertopic_news",
                        help="Prefix for output files (default: bertopic_news)")
    parser.add_argument("--min_topic_size",
                        type=int, default=10,
                        help="Minimum topic size (default: 10). Lower for smaller datasets.")
    parser.add_argument("--embedding_model",
                        default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    # ===================================
    # 1. Load data
    # ===================================
    print("Loading titles...")
    df = pd.read_csv(args.input_file, sep='\t', header=None,
                     names=['article_id', 'title'], dtype=str)
    
    # Clean: remove empty titles
    initial_count = len(df)
    df = df.dropna(subset=['title'])
    df = df[df['title'].str.strip() != '']
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} titles (dropped {initial_count - len(df)} empty/invalid)")

    titles = df['title'].tolist()

    # ===================================
    # 2. BERTopic setup – tuned for short texts (titles)
    # ===================================
    print("Initializing BERTopic (this may take a moment for embedding download)...")
    
    # Fast & excellent for short English texts
    embedding_model = SentenceTransformer(args.embedding_model)
    
    # Use n-grams and remove very common words – helps a lot with titles
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        min_df=2,           # ignore terms that appear in only 1 document
        max_df=0.95
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,   # controls granularity
        nr_topics="auto",        # automatically merge similar topics & reduce outliers
        calculate_probabilities=True,
        verbose=True
    )

    # ===================================
    # 3. Fit the model
    # ===================================
    print("Fitting BERTopic on titles...")
    topics, probs = topic_model.fit_transform(titles)

    # ===================================
    # 4. Get rich document info (includes probability & auto topic name)
    # ===================================
    print("Extracting topic assignments and metadata...")
    doc_info = topic_model.get_document_info(titles)

    # doc_info preserves original order → safe to attach
    df['topic']          = doc_info['Topic']
    df['topic_name']     = doc_info['Name']          # e.g., 0_trump_biden_election_
    df['top_words']      = doc_info['Top_n_words']
    df['probability']    = doc_info['Probability']
    # Representative_Document may not exist in older BERTopic versions
    if 'Representative_Document' in doc_info.columns:
        df['representative'] = doc_info['Representative_Document']
    else:
        df['representative'] = False

    # ===================================
    # 5. Save results
    # ===================================
    assignments_file = f"{args.output_prefix}_assignments.tsv"
    topics_file      = f"{args.output_prefix}_topics.tsv"

    # Assignments: one article per row
    output_cols = [
        'article_id', 'title', 'topic', 'topic_name',
        'probability', 'top_words'
    ]
    if 'Representative_Document' in doc_info.columns:
        output_cols.append('representative')
    df[output_cols].to_csv(assignments_file, sep='\t', index=False)
    
    # Topic overview
    topic_model.get_topic_info().to_csv(topics_file, sep='\t', index=False)

    print("\n=== Done ===")
    print(f"Assignments saved to: {assignments_file}")
    print(f"Topic overview saved to: {topics_file}")
    print("\nTopic summary:")
    print(topic_model.get_topic_info()[['Topic', 'Count', 'Name', 'Representation']].head(20))

    # Optional: show a few example clusters
    print("\nExample clusters:")
    for t in topic_model.get_topic_info()['Topic'].head(8):
        if t == -1:
            print(f"\nTopic -1 (Outliers): {topic_model.get_topic_info().loc[topic_model.get_topic_info()['Topic']==-1, 'Count'].values[0]} articles")
            continue
        words = [w for w, _ in topic_model.get_topic(t)[:10]]
        print(f"Topic {t} | {topic_model.get_topic_info().loc[topic_model.get_topic_info()['Topic']==t, 'Name'].values[0]} | {' | '.join(words)}")


if __name__ == "__main__":
    main()