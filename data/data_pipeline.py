# data_pipeline.py
"""
This module handles data collection, processing, and loading for the recommendation system.
It includes functions to:
1. Download and preprocess movie ratings data
2. Split data into train/validation/test sets
3. Create user-item interaction matrices
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import urllib.request
import zipfile
import logging

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, data_dir="data"):
        """Initialize data pipeline with directory paths."""
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

        # Create directories if they do not exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Data file paths
        self.ratings_file = os.path.join(self.raw_dir, "ratings.csv")
        self.movies_file = os.path.join(self.raw_dir, "movies.csv")

    def download_movielens_data(self, dataset_size="small"):
        """
        Download the MovieLens dataset.
        
        Args:
            dataset_size (str): Size of dataset to download ("small" or "full")
        """

        if dataset_size == "small":
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        else:
            url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
        
        # Path for the zip file
        zip_path = os.path.join(self.raw_dir, "movielens.zip")

        logger.info(f"Downloading MovieLens {dataset_size} dataset...")
        urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Downloading MovieLens {dataset_size} dataset...")
        urllib.request.urlretrieve(url, zip_path)


        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)


        # Move files to the correct location
        if dataset_size == "small":
            extract_dir = os.path.join(self.raw_dir, "ml-latest-small")
        else:
            extract_dir = os.path.join(self.raw_dir, "ml-latest")
        
        # Copy files to the raw directory
        if os.path.exists(os.path.join(extract_dir, "ratings.csv")):
            os.replace(
                os.path.join(extract_dir, "ratings.csv"),
                self.ratings_file
            )
        if os.path.exists(os.path.join(extract_dir, "movies.csv")):
            os.replace(
                os.path.join(extract_dir, "movies.csv"),
                self.movies_file
            )
        
        logger.info("Dataset downloaded and extracted successfully.")

    def load_data(self):
        """
        Load ratings and movies data from CSV files.
        
        Returns:
            tuple: (ratings_df, movies_df)
        """
        # Check if files exist, download if not
        if not (os.path.exists(self.ratings_file) and os.path.exists(self.movies_file)):
            logger.info("Data files not found. Downloading...")
            self.download_movielens_data()
        
        # Load data
        logger.info("Loading ratings data...")
        ratings_df = pd.read_csv(self.ratings_file)
        
        logger.info("Loading movies data...")
        movies_df = pd.read_csv(self.movies_file)
        
        return ratings_df, movies_df
    
    def preprocess_data(self, ratings_df, movies_df):
        """
        Preprocess the ratings and movies data.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with ratings data
            movies_df (pd.DataFrame): DataFrame with movies data
            
        Returns:
            tuple: Processed (ratings_df, movies_df)
        """
        logger.info("Preprocessing data...")
        
        # Check for missing values
        if ratings_df.isnull().values.any():
            logger.info("Removing rows with missing values in ratings data...")
            ratings_df = ratings_df.dropna()
        
        if movies_df.isnull().values.any():
            logger.info("Removing rows with missing values in movies data...")
            movies_df = movies_df.dropna()
        
        # Create sequential IDs for users and items (movies)
        logger.info("Creating sequential user and movie IDs...")
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings_df['userId'].unique())}
        movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movies_df['movieId'].unique())}
        
        # Add new ID columns
        ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_map)
        movies_df['movie_idx'] = movies_df['movieId'].map(movie_id_map)
        
        # Save the ID mappings
        pd.DataFrame(list(user_id_map.items()), columns=['userId', 'user_idx']).to_csv(
            os.path.join(self.processed_dir, "user_id_map.csv"), index=False
        )
        pd.DataFrame(list(movie_id_map.items()), columns=['movieId', 'movie_idx']).to_csv(
            os.path.join(self.processed_dir, "movie_id_map.csv"), index=False
        )
        
        # Extract genres as one-hot encoding
        logger.info("Processing movie genres...")
        genres = []
        for genre_list in movies_df['genres']:
            genres.extend(genre_list.split('|'))
        unique_genres = sorted(set(genres))
        
        # One-hot encode the genres
        for genre in unique_genres:
            movies_df[f'genre_{genre}'] = movies_df['genres'].apply(
                lambda x: 1 if genre in x.split('|') else 0
            )
        
        return ratings_df, movies_df
    
    def split_data(self, ratings_df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the ratings data into train, validation, and test sets.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with ratings data
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train, validation, and test sets...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            ratings_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=ratings_df['userId'] if len(ratings_df['userId'].unique()) < 10 else None
        )
        
        # Second split: separate validation set from training set
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['userId'] if len(train_val_df['userId'].unique()) < 10 else None
        )
        
        # Save the splits
        train_df.to_csv(os.path.join(self.processed_dir, "train_ratings.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "val_ratings.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test_ratings.csv"), index=False)
        
        logger.info(f"Data split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_interaction_matrix(self, ratings_df):
        """
        Create a sparse user-item interaction matrix from ratings.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with ratings data
            
        Returns:
            tuple: (interaction_matrix, user_indices, movie_indices)
        """
        logger.info("Creating user-item interaction matrix...")
        
        # Get dimensions - use max index + 1 to get proper dimensions
        n_users = ratings_df['user_idx'].max() + 1
        n_items = ratings_df['movie_idx'].max() + 1
    
        # Create sparse matrix
        rows = ratings_df['user_idx'].values
        cols = ratings_df['movie_idx'].values
        values = ratings_df['rating'].values
        
        interaction_matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_items))
        
        # Ensure indices are within bounds
        if cols.max() >= n_items or rows.max() >= n_users:
            logger.warning(f"Found indices exceeding matrix dimensions. Adjusting dimensions.")
            n_items = max(cols.max() + 1, n_items)
            n_users = max(rows.max() + 1, n_users)
    
        interaction_matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_items))
    
        # Save matrix to disk
        import scipy.sparse as sp
        sp.save_npz(os.path.join(self.processed_dir, "interaction_matrix.npz"), interaction_matrix)
        
        logger.info(f"Created interaction matrix of shape {interaction_matrix.shape}")
        
        return interaction_matrix, np.unique(rows), np.unique(cols)
    
    def process_pipeline(self):
        """
        Run the complete data processing pipeline.
        
        Returns:
            tuple: (train_matrix, val_df, test_df, movies_df)
        """
        # Load data
        ratings_df, movies_df = self.load_data()
        
        # Preprocess data
        ratings_df, movies_df = self.preprocess_data(ratings_df, movies_df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(ratings_df)
        
        # Create interaction matrix for training
        train_matrix, _, _ = self.create_interaction_matrix(train_df)
        
        # Save processed movies data
        movies_df.to_csv(os.path.join(self.processed_dir, "processed_movies.csv"), index=False)
        
        logger.info("Data pipeline completed successfully.")
        
        return train_matrix, val_df, test_df, movies_df


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    train_matrix, val_df, test_df, movies_df = pipeline.process_pipeline()
    
    print(f"Training data shape: {train_matrix.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Movies data shape: {movies_df.shape}")