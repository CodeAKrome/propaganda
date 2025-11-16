#!/usr/bin/env python
import os
from pymongo import MongoClient
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

class ArticleRecommender:
    """
    Content-based article recommendation system using TF-IDF and cosine similarity.
    Recommendations are based on article title and content similarity to rated articles.
    """
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", 
                 db_name: str = "news_db"):
        """
        Initialize the recommender system.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.ratings_collection = self.db['ratings']
        self.articles_collection = self.db['articles']
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.article_vectors = {}
        self.article_texts = {}
        
    def add_rating(self, user_id: str, article_id: str, rating: float, 
                   timestamp: datetime = None):
        """
        Add a user rating for an article.
        
        Args:
            user_id: User identifier
            article_id: Article identifier (ObjectId string)
            rating: Rating value (e.g., 1-5 scale)
            timestamp: Rating timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        rating_doc = {
            'user_id': user_id,
            'article_id': article_id,
            'rating': rating,
            'timestamp': timestamp
        }
        
        # Update if exists, insert if not
        self.ratings_collection.update_one(
            {'user_id': user_id, 'article_id': article_id},
            {'$set': rating_doc},
            upsert=True
        )
    
    def _get_article_text(self, article_id: str) -> str:
        """
        Retrieve and combine article title and content.
        
        Args:
            article_id: Article ObjectId string
            
        Returns:
            Combined text of title and article content
        """
        # Check cache first
        if article_id in self.article_texts:
            return self.article_texts[article_id]
        
        # Query MongoDB - convert string to ObjectId
        try:
            article = self.articles_collection.find_one({'_id': ObjectId(article_id)})
        except Exception as e:
            print(f"Error querying article {article_id}: {e}")
            return ""
        
        if not article:
            return ""
        
        # Combine title and article text, give more weight to title
        title = article.get('title', '')
        content = article.get('article', '')
        
        # Repeat title to give it more weight in similarity calculation
        combined_text = f"{title} {title} {title} {content}"
        
        # Cache it
        self.article_texts[article_id] = combined_text
        
        return combined_text
    
    def _vectorize_articles(self, article_ids: List[str]):
        """
        Create TF-IDF vectors for articles.
        
        Args:
            article_ids: List of article ObjectId strings
        """
        texts = []
        valid_ids = []
        
        for article_id in article_ids:
            text = self._get_article_text(article_id)
            if text:
                texts.append(text)
                valid_ids.append(article_id)
        
        if not texts:
            return
        
        # Fit and transform texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Store vectors
        for i, article_id in enumerate(valid_ids):
            self.article_vectors[article_id] = tfidf_matrix[i]
    
    def _get_user_profile(self, user_id: str) -> Tuple[np.ndarray, List[str]]:
        """
        Build a user profile based on their rated articles.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (weighted average vector, list of rated article IDs)
        """
        # Get user's ratings
        ratings = list(self.ratings_collection.find({'user_id': user_id}))
        
        if not ratings:
            return None, []
        
        # Get article IDs and ratings
        article_ids = [r['article_id'] for r in ratings]
        rating_values = [r['rating'] for r in ratings]
        
        # Ensure all articles are vectorized
        missing_ids = [aid for aid in article_ids if aid not in self.article_vectors]
        if missing_ids:
            self._vectorize_articles(missing_ids)
        
        # Build weighted profile
        vectors = []
        weights = []
        
        for article_id, rating in zip(article_ids, rating_values):
            if article_id in self.article_vectors:
                vectors.append(self.article_vectors[article_id].toarray().flatten())
                # Weight by rating (higher ratings = more influence)
                weights.append(rating)
        
        if not vectors:
            return None, article_ids
        
        # Compute weighted average
        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted average of vectors
        user_profile = (vectors * weights).sum(axis=0)
        
        return user_profile, article_ids
    
    def recommend_articles(self, user_id: str, article_ids: List[str], 
                          n_recommendations: int = 5, 
                          min_rating: float = 3.0) -> List[Tuple[str, float]]:
        """
        Recommend articles based on content similarity to user's rated articles.
        
        Args:
            user_id: User identifier
            article_ids: List of candidate article ObjectId strings
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating threshold for building user profile
            
        Returns:
            List of tuples (article_id, similarity_score) sorted by score
        """
        # Get user profile
        user_profile, rated_article_ids = self._get_user_profile(user_id)
        
        if user_profile is None:
            raise ValueError(f"User {user_id} has no ratings to build profile from.")
        
        # Filter out already rated articles
        candidate_articles = [aid for aid in article_ids if aid not in rated_article_ids]
        
        if not candidate_articles:
            return []
        
        # Vectorize candidate articles
        missing_ids = [aid for aid in candidate_articles if aid not in self.article_vectors]
        if missing_ids:
            self._vectorize_articles(missing_ids)
        
        # Calculate similarities
        similarities = []
        for article_id in candidate_articles:
            if article_id in self.article_vectors:
                article_vector = self.article_vectors[article_id].toarray().flatten()
                
                # Cosine similarity
                similarity = cosine_similarity(
                    user_profile.reshape(1, -1), 
                    article_vector.reshape(1, -1)
                )[0][0]
                
                similarities.append((article_id, float(similarity)))
        
        # Sort by similarity (descending) and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]
    
    def get_article_details(self, article_id: str) -> Dict:
        """
        Retrieve article details from MongoDB.
        
        Args:
            article_id: Article ObjectId string
            
        Returns:
            Article document or None
        """
        try:
            return self.articles_collection.find_one({'_id': ObjectId(article_id)})
        except Exception as e:
            print(f"Error retrieving article {article_id}: {e}")
            return None
    
    def recommend_with_details(self, user_id: str, article_ids: List[str], 
                               n_recommendations: int = 5) -> List[Dict]:
        """
        Recommend articles with full details.
        
        Args:
            user_id: User identifier
            article_ids: List of candidate article ObjectId strings
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries with article details and similarity scores
        """
        recommendations = self.recommend_articles(user_id, article_ids, n_recommendations)
        
        result = []
        for article_id, similarity in recommendations:
            article = self.get_article_details(article_id)
            
            rec_item = {
                'article_id': article_id,
                'similarity_score': round(similarity, 4)
            }
            
            if article:
                rec_item.update({
                    'title': article.get('title', 'N/A'),
                    'source': article.get('source', 'N/A'),
                    'description': article.get('description', 'N/A'),
                    'link': article.get('link', 'N/A'),
                    'published': article.get('published', {}).get('$date', 'N/A') if isinstance(article.get('published'), dict) else article.get('published', 'N/A')
                })
            
            result.append(rec_item)
        
        return result
    
    def get_user_ratings(self, user_id: str) -> List[Dict]:
        """
        Get all ratings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of rating documents
        """
        return list(self.ratings_collection.find({'user_id': user_id}))
    
    def get_similar_articles(self, article_id: str, candidate_ids: List[str], 
                            n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Find articles similar to a given article.
        
        Args:
            article_id: Reference article ObjectId string
            candidate_ids: List of candidate article ObjectId strings
            n_similar: Number of similar articles to return
            
        Returns:
            List of tuples (article_id, similarity_score)
        """
        # Vectorize articles if needed
        all_ids = [article_id] + candidate_ids
        missing_ids = [aid for aid in all_ids if aid not in self.article_vectors]
        if missing_ids:
            self._vectorize_articles(missing_ids)
        
        if article_id not in self.article_vectors:
            raise ValueError(f"Could not vectorize article {article_id}")
        
        reference_vector = self.article_vectors[article_id].toarray().flatten()
        
        # Calculate similarities
        similarities = []
        for candidate_id in candidate_ids:
            if candidate_id == article_id:
                continue
            
            if candidate_id in self.article_vectors:
                candidate_vector = self.article_vectors[candidate_id].toarray().flatten()
                
                similarity = cosine_similarity(
                    reference_vector.reshape(1, -1),
                    candidate_vector.reshape(1, -1)
                )[0][0]
                
                similarities.append((candidate_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = ArticleRecommender(
        mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017"),
        db_name="star"
    )
    
    # Example: Add some ratings
    # User likes political articles
    recommender.add_rating("user1", "68bc02a9d93ed529e0f83d96", 4.5)  # RFK article
    recommender.add_rating("user1", "article_id_politics_2", 4.0)
    recommender.add_rating("user1", "article_id_sports", 2.0)  # Doesn't like sports
    
    # Get recommendations
    new_article_ids = [
        "article_id_politics_3",
        "article_id_sports_2", 
        "article_id_tech",
        "article_id_politics_4"
    ]
    
    try:
        recommendations = recommender.recommend_with_details(
            user_id="user1",
            article_ids=new_article_ids,
            n_recommendations=3
        )
        
        print("Content-based recommendations for user1:")
        print("(Based on title and article text similarity)\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.get('title', rec['article_id'])}")
            print(f"   Similarity: {rec['similarity_score']}")
            print(f"   Source: {rec.get('source', 'N/A')}")
            print()
            
    except ValueError as e:
        print(f"Recommendation error: {e}")
    
    # Example: Find similar articles
    try:
        similar = recommender.get_similar_articles(
            article_id="68bc02a9d93ed529e0f83d96",
            candidate_ids=new_article_ids,
            n_similar=3
        )
        
        print("\nArticles similar to the RFK article:")
        for article_id, score in similar:
            print(f"  {article_id}: {score:.4f}")
            
    except ValueError as e:
        print(f"Similar articles error: {e}")
    
    # Close connection
    recommender.close()