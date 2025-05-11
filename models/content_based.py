# content_based.py
"""
This module implements content-based recommendation algorithms.
It analyzes item features (e.g., movie genres, tags) to find similar items.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import pickle
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    Content-based recommender that uses item metadata to make recommendations.
    For movies, this uses genres, keywords, and other attributes.
    """
    
    def __init__(self):
        """Initialize the content-based recommender."""
        self.item_features = None  # Will store item features
        self.item_similarity = None  # Will store item similarity matrix
        self.movies_df = None  # Will store movie metadata
        self.tfidf_matrix = None  # Will store TF-IDF matrix
        self.tfidf = None  # Will store TF-IDF vectorizer
    
    def _extract_features(self, movies_df):
        """
        Extract features from movie metadata for content-based recommendations.
        
        Args:
            movies_df (pd.DataFrame): DataFrame with movie metadata
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        logger.info("Extracting features from movie metadata...")
        
        # Create a copy to avoid modifying the original
        df = movies_df.copy()
        
        # Check which columns are available in the dataset
        available_columns = set(df.columns)
        
        # Check if movie_idx exists, if not, try to create it
        if 'movie_idx' not in available_columns:
            if 'movieId' in available_columns:
                logger.info("Creating movie_idx column from movieId")
                # Create a mapping from movieId to sequential indices
                movie_indices = {movie_id: idx for idx, movie_id in enumerate(df['movieId'].unique())}
                df['movie_idx'] = df['movieId'].map(movie_indices)
            else:
                # If neither exists, create a default index
                logger.info("No movieId column found, creating a default movie_idx")
                df['movie_idx'] = range(len(df))
        
        # Features to use (if available)
        feature_columns = []
        
        # Always use genres
        if 'genres' in available_columns:
            feature_columns.append('genres')
            # Clean up genres (replace | with space for better tokenization)
            df['genres'] = df['genres'].str.replace('|', ' ')
        
        # Check for other potential feature columns
        optional_columns = ['overview', 'keywords', 'tagline', 'director', 'cast']
        for col in optional_columns:
            if col in available_columns:
                feature_columns.append(col)
        
        # If no additional columns found beyond genres, create a simple feature string
        if len(feature_columns) == 1 and feature_columns[0] == 'genres':
            df['feature_string'] = df['genres']
        else:
            # Create a combined feature string with different weights
            df['feature_string'] = ''
            
            # Add genres (multiply for higher weight)
            if 'genres' in feature_columns:
                df['feature_string'] += df['genres'] + ' ' + df['genres']
            
            # Add other features
            for col in feature_columns:
                if col != 'genres' and col in available_columns:
                    # Clean and add the column
                    if df[col].dtype == 'object':  # Check if it's a string column
                        df['feature_string'] += ' ' + df[col].fillna('').astype(str)
        
        # Clean up feature string
        df['feature_string'] = df['feature_string'].fillna('')
        df['feature_string'] = df['feature_string'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x.lower()) if isinstance(x, str) else '')
        
        # Ensure all required columns exist
        required_columns = ['movieId', 'movie_idx', 'title', 'feature_string']
        for col in required_columns:
            if col not in df.columns:
                if col == 'title' and 'movieId' in df.columns:
                    # Create a default title if missing
                    df['title'] = 'Movie ' + df['movieId'].astype(str)
                else:
                    # Add an empty column as fallback
                    logger.warning(f"Adding empty column for {col}")
                    df[col] = ""
        
        # Return only needed columns
        result_df = df[required_columns]
        
        logger.info("Feature extraction complete.")
        return result_df
    
    def fit(self, movies_df):
        """
        Build the content-based recommender model.
        
        Args:
            movies_df (pd.DataFrame): DataFrame with movie metadata
        """
        logger.info("Fitting content-based recommender model...")
        
        # Store the movies dataframe
        self.movies_df = movies_df
        
        # Extract features
        feature_df = self._extract_features(movies_df)
        
        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            min_df=2,  # Minimum document frequency
            max_features=5000,  # Maximum number of features
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform feature strings to TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(feature_df['feature_string'])
        
        # Compute item similarity
        self.item_similarity = cosine_similarity(self.tfidf_matrix)
        
        # Create a mapping from movie index to position in the similarity matrix
        self.movie_idx_to_pos = {
            movie_idx: i for i, movie_idx in enumerate(feature_df['movie_idx'])
        }
        
        # Create reverse mapping
        self.pos_to_movie_idx = {
            pos: movie_idx for movie_idx, pos in self.movie_idx_to_pos.items()
        }
        
        logger.info("Model fitting complete.")
    
    def get_similar_items(self, item_idx, n=10):
        """
        Get items similar to a given item.
        
        Args:
            item_idx (int): Item index
            n (int): Number of similar items to retrieve
            
        Returns:
            list: List of (item_idx, similarity_score) tuples
        """
        # Get the position in the similarity matrix
        if item_idx not in self.movie_idx_to_pos:
            logger.warning(f"Item index {item_idx} not found in the model.")
            return []
        
        pos = self.movie_idx_to_pos[item_idx]
        
        # Get similarity scores
        similarity_scores = self.item_similarity[pos]
        
        # Sort scores and get top N (excluding self)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n+1]
        similar_scores = similarity_scores[similar_indices]
        
        # Convert positions back to movie indices
        result = [
            (self.pos_to_movie_idx[idx], score) 
            for idx, score in zip(similar_indices, similar_scores)
        ]
        
        return result
    
    def recommend_similar(self, item_idx, n=10):
        """
        Recommend items similar to a given item.
        
        Args:
            item_idx (int): Item index
            n (int): Number of recommendations
            
        Returns:
            pd.DataFrame: DataFrame with recommendations
        """
        # Get similar items
        similar_items = self.get_similar_items(item_idx, n)
        
        # Create a list of movie indices
        movie_indices = [idx for idx, _ in similar_items]
        similarity_scores = [score for _, score in similar_items]
        
        # Get movie details
        recommendations = self.movies_df[self.movies_df['movie_idx'].isin(movie_indices)].copy()
        
        # Add similarity scores
        score_dict = {idx: score for idx, score in similar_items}
        recommendations['similarity_score'] = recommendations['movie_idx'].map(score_dict)
        
        # Sort by similarity score
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations
    
    def recommend_for_user(self, user_idx, train_matrix, n=10, diversity_level=0.3):
        """
        Recommend items for a user based on their past ratings.
        
        Args:
            user_idx (int): User index
            train_matrix (csr_matrix): User-item interaction matrix
            n (int): Number of recommendations
            diversity_level (float): Level of diversity in recommendations (0-1)
            
        Returns:
            list: List of (item_idx, score) tuples
        """
        # Get user's rated items
        rated_items = train_matrix[user_idx].nonzero()[1]
        
        if len(rated_items) == 0:
            logger.warning(f"User {user_idx} has no ratings. Cannot make content-based recommendations.")
            return []
        
        # Get the ratings
        ratings = [train_matrix[user_idx, item] for item in rated_items]
        
        # Create a dictionary of item -> rating
        item_ratings = dict(zip(rated_items, ratings))
        
        # Find similar items for each rated item
        all_candidates = {}
        
        for item in rated_items:
            # Get similar items
            similar_items = self.get_similar_items(item, n=20)
            
            # Weight by the user's rating of the source item
            rating = item_ratings[item]
            
            for similar_item, similarity in similar_items:
                # Skip if it's an already rated item
                if similar_item in rated_items:
                    continue
                
                # Calculate score (rating * similarity)
                score = rating * similarity
                
                # Update the candidate score (add if new, take max if existing)
                if similar_item in all_candidates:
                    # Apply diversity - higher diversity means we take the average
                    # Lower diversity means we take the maximum
                    current_score = all_candidates[similar_item]
                    all_candidates[similar_item] = (
                        diversity_level * (current_score + score) / 2 +
                        (1 - diversity_level) * max(current_score, score)
                    )
                else:
                    all_candidates[similar_item] = score
        
        # Convert to list and sort by score
        recommendations = [(item, score) for item, score in all_candidates.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n]
    
    def save_model(self, file_path):
        """
        Save the model to a file.
        
        Args:
            file_path (str): Path to save the model
        """
        # Create a dictionary with the essential model components
        model_data = {
            'tfidf_matrix': self.tfidf_matrix,
            'item_similarity': self.item_similarity,
            'movie_idx_to_pos': self.movie_idx_to_pos,
            'pos_to_movie_idx': self.pos_to_movie_idx,
            'tfidf': self.tfidf
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path, movies_df=None):
        """
        Load a saved model.
        
        Args:
            file_path (str): Path to the saved model
            movies_df (pd.DataFrame): Optional DataFrame with movie metadata
            
        Returns:
            ContentBasedRecommender: Loaded model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls()
        
        # Restore model attributes
        model.tfidf_matrix = model_data['tfidf_matrix']
        model.item_similarity = model_data['item_similarity']
        model.movie_idx_to_pos = model_data['movie_idx_to_pos']
        model.pos_to_movie_idx = model_data['pos_to_movie_idx']
        model.tfidf = model_data['tfidf']
        
        # Set movies dataframe if provided
        if movies_df is not None:
            model.movies_df = movies_df
        
        logger.info(f"Model loaded from {file_path}")
        return model