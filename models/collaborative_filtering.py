# collaborative_filtering.py
"""
This module implements collaborative filtering recommendation algorithms.
It includes:
1. Memory-based methods (user-based, item-based)
2. Matrix factorization methods (SVD, ALS)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryBasedCF:
    """
    Memory-based collaborative filtering using similarity calculations.
    Implements both user-based and item-based approaches.
    """
    
    def __init__(self, method='item', similarity='cosine', k=20):
        """
        Initialize memory-based CF model.
        
        Args:
            method (str): Method to use ('user' or 'item')
            similarity (str): Similarity measure to use ('cosine', 'pearson')
            k (int): Number of nearest neighbors to consider
        """
        self.method = method
        self.similarity = similarity
        self.k = k
        self.sim_matrix = None  # Will store precomputed similarity matrix
        self.train_matrix = None  # Will store training data
        self.mean_ratings = None  # Will store mean ratings (for normalization)
        
    def fit(self, train_matrix):
        """
        Compute similarity matrix from training data.
        
        Args:
            train_matrix (csr_matrix): User-item interaction matrix
        """
        logger.info(f"Fitting {self.method}-based collaborative filtering model...")
        self.train_matrix = train_matrix
        
        # Normalize ratings by user mean
        if self.method == 'user':
            # Calculate mean rating for each user, ignoring zeros
            user_means = np.zeros(train_matrix.shape[0])
            for i in range(train_matrix.shape[0]):
                user_data = train_matrix[i].data
                if user_data.size > 0:
                    user_means[i] = user_data.mean()
                else:
                    user_means[i] = 0
            
            # Store user means for prediction
            self.mean_ratings = user_means

            # For user-based CF, we need to work with a dense matrix for the normalization
            # Convert to a dense matrix for small datasets, or use alternative approach for large ones
            if train_matrix.shape[0] * train_matrix.shape[1] < 10000000:  # Only for reasonably sized matrices
                logger.info("Converting to dense matrix for normalization...")
                normalized_matrix = train_matrix.toarray()
                for i in range(train_matrix.shape[0]):
                    if user_means[i] > 0:
                        # Only normalize if the user has ratings
                        row_indices = np.nonzero(normalized_matrix[i])[0]
                        normalized_matrix[i, row_indices] = normalized_matrix[i, row_indices] - user_means[i]
            else:
                # For large matrices, skip normalization and use the original matrix
                logger.info("Matrix too large for dense conversion, using original matrix...")
                normalized_matrix = train_matrix.toarray()
            
            # Compute user similarity matrix using the processed matrix
            logger.info("Computing user similarity matrix...")
            similarity = cosine_similarity(normalized_matrix)
            self.sim_matrix = similarity
            
        elif self.method == 'item':
            # Calculate mean rating for each item, ignoring zeros
            item_means = np.zeros(train_matrix.shape[1])
            for i in range(train_matrix.shape[1]):
                item_data = train_matrix[:, i].data
                if item_data.size > 0:
                    item_means[i] = item_data.mean()
                else:
                    item_means[i] = 0
            
            # Store item means for prediction
            self.mean_ratings = item_means
            
            # Compute item similarity matrix
            logger.info("Computing item similarity matrix...")
            similarity = cosine_similarity(train_matrix.T)
            self.sim_matrix = similarity
        
        logger.info("Model fitting complete.")
    
    def _get_neighbors(self, idx, k=None):
        """
        Get the k most similar users/items.
        
        Args:
            idx (int): User or item index
            k (int, optional): Number of neighbors, uses self.k if not specified
            
        Returns:
            tuple: (neighbor_indices, similarity_scores)
        """
        if k is None:
            k = self.k
        
        # Get similarities for the target user/item
        similarities = self.sim_matrix[idx]
        
        # Sort similarities and get top k (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:k+1]
        similar_scores = similarities[similar_indices]
        
        return similar_indices, similar_scores
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for a specific user-item pair.
        
        Args:
            user_idx (int): User index
            item_idx (int): Item index
            
        Returns:
            float: Predicted rating
        """
        if self.method == 'user':
            # Get similar users
            similar_users, sim_scores = self._get_neighbors(user_idx)
            
            # Get user mean rating
            user_mean = self.mean_ratings[user_idx]
            
            # Calculate weighted rating
            numerator = 0
            denominator = 0
            
            for neighbor, sim_score in zip(similar_users, sim_scores):
                # Check if neighbor has rated the item
                rating = self.train_matrix[neighbor, item_idx]
                if rating > 0:  # If neighbor has rated the item
                    # Adjusted rating (normalized by neighbor's mean)
                    neighbor_mean = self.mean_ratings[neighbor]
                    numerator += sim_score * (rating - neighbor_mean)
                    denominator += abs(sim_score)
            
            # Return predicted rating
            if denominator > 0:
                return user_mean + (numerator / denominator)
            else:
                return user_mean  # Fallback to user's mean rating
                
        elif self.method == 'item':
            # Get similar items to the target item
            similar_items, sim_scores = self._get_neighbors(item_idx)
            
            # Calculate weighted rating
            numerator = 0
            denominator = 0
            
            for neighbor, sim_score in zip(similar_items, sim_scores):
                # Check if user has rated the neighbor item
                rating = self.train_matrix[user_idx, neighbor]
                if rating > 0:  # If user has rated the neighbor item
                    numerator += sim_score * rating
                    denominator += abs(sim_score)
            
            # Return predicted rating
            if denominator > 0:
                return numerator / denominator
            else:
                return self.mean_ratings[item_idx]  # Fallback to item's mean rating
    
    def recommend(self, user_idx, n=10, exclude_rated=True):
        """
        Generate top N recommendations for a user.
        
        Args:
            user_idx (int): User index
            n (int): Number of recommendations to generate
            exclude_rated (bool): Whether to exclude already rated items
            
        Returns:
            list: List of (item_idx, predicted_rating) tuples
        """
        # Get user's rated items (to exclude them)
        if exclude_rated:
            rated_items = set(self.train_matrix[user_idx].indices)
        else:
            rated_items = set()
        
        # Get predictions for all items
        predictions = []
        for item_idx in range(self.train_matrix.shape[1]):
            if item_idx not in rated_items:
                pred_rating = self.predict(user_idx, item_idx)
                predictions.append((item_idx, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def save_model(self, file_path):
        """
        Save the model to a file.
        
        Args:
            file_path (str): Path to save the model
        """
        model_data = {
            'method': self.method,
            'similarity': self.similarity,
            'k': self.k,
            'sim_matrix': self.sim_matrix,
            'mean_ratings': self.mean_ratings
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a saved model.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            MemoryBasedCF: Loaded model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            method=model_data['method'],
            similarity=model_data['similarity'],
            k=model_data['k']
        )
        
        # Restore model attributes
        model.sim_matrix = model_data['sim_matrix']
        model.mean_ratings = model_data['mean_ratings']
        
        logger.info(f"Model loaded from {file_path}")
        return model


class MatrixFactorizationCF:
    """
    Matrix factorization collaborative filtering using Singular Value Decomposition (SVD).
    """
    
    def __init__(self, n_factors=100, reg=0.1, learning_rate=0.005, n_epochs=20):
        """
        Initialize matrix factorization model.
        
        Args:
            n_factors (int): Number of latent factors
            reg (float): Regularization parameter
            learning_rate (float): Learning rate for gradient descent
            n_epochs (int): Number of epochs for training
        """
        self.n_factors = n_factors
        self.reg = reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.item_biases = None
        self.user_biases = None
        self.global_bias = None
    
    def _initialize_parameters(self, train_matrix):
        """
        Initialize model parameters.
        
        Args:
            train_matrix (csr_matrix): User-item interaction matrix
        """
        n_users, n_items = train_matrix.shape
        
        # Initialize factors with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases to zeros
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Calculate global bias (global mean)
        if train_matrix.nnz > 0:
            self.global_bias = train_matrix.data.mean()
        else:
            self.global_bias = 0.0
    
    def fit(self, train_matrix):
        """
        Train the matrix factorization model using SVD.
        
        Args:
            train_matrix (csr_matrix): User-item interaction matrix
        """
        logger.info("Fitting matrix factorization model...")
        
        # Initialize parameters
        self._initialize_parameters(train_matrix)
        
        n_users, n_items = train_matrix.shape
        
        # Convert to dense for SVD implementation
        # This is for simplicity - for large matrices you'd want a sparse SVD implementation
        if isinstance(train_matrix, csr_matrix):
            # Fill missing values with the global mean
            dense_matrix = train_matrix.toarray()
            mask = (dense_matrix == 0)
            dense_matrix[mask] = self.global_bias
        else:
            dense_matrix = train_matrix.copy()
        
        # Perform SVD
        U, sigma, Vt = svds(dense_matrix, k=self.n_factors)
        
        # Convert sigma to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Store factors
        self.user_factors = U
        self.item_factors = Vt.T
        
        # Scale factors by singular values
        self.user_factors = np.dot(self.user_factors, np.sqrt(sigma_diag))
        self.item_factors = np.dot(self.item_factors, np.sqrt(sigma_diag))
        
        logger.info("Model fitting complete.")
    
    def fit_sgd(self, train_matrix):
        """
        Train the matrix factorization model using Stochastic Gradient Descent.
        This is an alternative to SVD for training.
        
        Args:
            train_matrix (csr_matrix): User-item interaction matrix
        """
        logger.info("Fitting matrix factorization model using SGD...")
        
        # Initialize parameters
        self._initialize_parameters(train_matrix)
        
        n_users, n_items = train_matrix.shape
        
        # Get all non-zero entries for training
        user_indices, item_indices = train_matrix.nonzero()
        n_ratings = len(user_indices)
        
        # Shuffle indices
        idx = np.random.permutation(n_ratings)
        user_indices = user_indices[idx]
        item_indices = item_indices[idx]
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            
            # Iterate through all ratings
            for i in range(n_ratings):
                user_idx = user_indices[i]
                item_idx = item_indices[i]
                rating = train_matrix[user_idx, item_idx]
                
                # Calculate prediction error
                pred = self.global_bias + self.user_biases[user_idx] + self.item_biases[item_idx] + \
                       np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                error = rating - pred
                
                # Update global bias
                self.global_bias += self.learning_rate * (error - self.reg * self.global_bias)
                
                # Update user and item biases
                self.user_biases[user_idx] += self.learning_rate * (error - self.reg * self.user_biases[user_idx])
                self.item_biases[item_idx] += self.learning_rate * (error - self.reg * self.item_biases[item_idx])
                
                # Update user and item factors
                user_factor_update = self.learning_rate * (error * self.item_factors[item_idx] - self.reg * self.user_factors[user_idx])
                item_factor_update = self.learning_rate * (error * self.user_factors[user_idx] - self.reg * self.item_factors[item_idx])
                
                self.user_factors[user_idx] += user_factor_update
                self.item_factors[item_idx] += item_factor_update
                
                # Accumulate loss
                epoch_loss += error ** 2
            
            # Log progress
            avg_loss = epoch_loss / n_ratings
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Model fitting complete.")
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for a specific user-item pair.
        
        Args:
            user_idx (int): User index
            item_idx (int): Item index
            
        Returns:
            float: Predicted rating
        """
        # Check if indices are in bounds
        if user_idx >= self.user_factors.shape[0] or item_idx >= self.item_factors.shape[0]:
            return self.global_bias
        
        # Calculate prediction
        pred = self.global_bias
        
        # Add user and item biases if available
        if self.user_biases is not None:
            pred += self.user_biases[user_idx]
        if self.item_biases is not None:
            pred += self.item_biases[item_idx]
        
        # Add factorized term
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return pred
    
    def recommend(self, user_idx, n=10, exclude_rated=True, train_matrix=None):
        """
        Generate top N recommendations for a user.
        
        Args:
            user_idx (int): User index
            n (int): Number of recommendations to generate
            exclude_rated (bool): Whether to exclude already rated items
            train_matrix (csr_matrix): Optional training matrix to determine rated items
            
        Returns:
            list: List of (item_idx, predicted_rating) tuples
        """
        # Get user's rated items (to exclude them)
        rated_items = set()
        if exclude_rated and train_matrix is not None:
            rated_items = set(train_matrix[user_idx].indices)
        
        # Get predictions for all items
        predictions = []
        for item_idx in range(self.item_factors.shape[0]):
            if item_idx not in rated_items:
                pred_rating = self.predict(user_idx, item_idx)
                predictions.append((item_idx, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def save_model(self, file_path):
        """
        Save the model to a file.
        
        Args:
            file_path (str): Path to save the model
        """
        model_data = {
            'n_factors': self.n_factors,
            'reg': self.reg,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_bias': self.global_bias
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a saved model.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            MatrixFactorizationCF: Loaded model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            n_factors=model_data['n_factors'],
            reg=model_data['reg'],
            learning_rate=model_data['learning_rate'],
            n_epochs=model_data['n_epochs']
        )
        
        # Restore model attributes
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_biases = model_data['user_biases']
        model.item_biases = model_data['item_biases']
        model.global_bias = model_data['global_bias']
        
        logger.info(f"Model loaded from {file_path}")
        return model