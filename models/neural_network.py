# neural_network.py
"""
This module implements neural network-based recommendation algorithms.
It uses PyTorch to create a neural collaborative filtering model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RatingDataset(Dataset):
    """Dataset for training the neural recommender."""
    
    def __init__(self, ratings_df):
        """
        Initialize the dataset from ratings dataframe.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with user ratings
        """
        self.users = ratings_df['user_idx'].values
        self.items = ratings_df['movie_idx'].values
        self.ratings = ratings_df['rating'].values
        
    def __len__(self):
        """Return the number of ratings."""
        return len(self.ratings)
    
    def __getitem__(self, idx):
        """Get a single (user, item, rating) tuple."""
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model.
    Implements a simple MLP-based model for rating prediction.
    """
    
    def __init__(self, n_users, n_items, n_factors=50, hidden_dims=[100, 50]):
        """
        Initialize the NCF model.
        
        Args:
            n_users (int): Number of users
            n_items (int): Number of items
            n_factors (int): Number of latent factors
            hidden_dims (list): Dimensions of hidden layers
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # Layers for MLP
        self.mlp_layers = self._build_mlp(n_factors * 2, hidden_dims)
        
        # Final prediction layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _build_mlp(self, input_dim, hidden_dims):
        """
        Build the MLP part of the network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
            
        Returns:
            nn.Sequential: MLP layers
        """
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, user_indices, item_indices):
        """
        Forward pass of the NCF model.
        
        Args:
            user_indices (torch.Tensor): User indices
            item_indices (torch.Tensor): Item indices
            
        Returns:
            torch.Tensor: Predicted ratings
        """
        # Get embeddings
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Pass through MLP
        mlp_output = self.mlp_layers(concat_embeds)
        
        # Generate prediction
        prediction = self.output_layer(mlp_output)
        
        return prediction.squeeze()

class NeuralRecommender:
    """
    Neural network-based recommender system using PyTorch.
    Trains a neural collaborative filtering model for recommendations.
    """
    
    def __init__(self, n_factors=50, hidden_dims=[100, 50], learning_rate=0.001,
                 batch_size=256, n_epochs=20, device=None):
        """
        Initialize the neural recommender.
        
        Args:
            n_factors (int): Number of latent factors
            hidden_dims (list): Dimensions of hidden layers
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            n_epochs (int): Number of training epochs
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        
        # Additional attributes
        self.n_users = None
        self.n_items = None
        self.train_losses = []
        self.val_losses = []
    
    def _create_data_loaders(self, train_df, val_df=None):
        """
        Create DataLoader objects for training and validation.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = RatingDataset(train_df)
        val_dataset = RatingDataset(val_df) if val_df is not None else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        return train_loader, val_loader
    
    def fit(self, train_df, val_df=None):
        """
        Train the neural network recommender model.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data (optional)
            
        Returns:
            self: The trained model
        """
        logger.info("Training neural recommender model...")
        
        # Get dimensions
        self.n_users = train_df['user_idx'].max() + 1
        self.n_items = train_df['movie_idx'].max() + 1
        
        logger.info(f"Number of users: {self.n_users}, Number of items: {self.n_items}")
        
        # Create model
        self.model = NeuralCollaborativeFiltering(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(train_df, val_df)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for user_idx, item_idx, rating in train_loader:
                # Move to device
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                prediction = self.model(user_idx, item_idx)
                loss = criterion(prediction, rating)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(rating)
            
            # Calculate average training loss
            avg_train_loss = total_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = 0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for user_idx, item_idx, rating in val_loader:
                        # Move to device
                        user_idx = user_idx.to(self.device)
                        item_idx = item_idx.to(self.device)
                        rating = rating.to(self.device)
                        
                        # Forward pass
                        prediction = self.model(user_idx, item_idx)
                        loss = criterion(prediction, rating)
                        
                        val_loss += loss.item() * len(rating)
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader.dataset)
                self.val_losses.append(avg_val_loss)
                
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        logger.info("Model training complete.")
        return self
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for a specific user-item pair.
        
        Args:
            user_idx (int): User index
            item_idx (int): Item index
            
        Returns:
            float: Predicted rating
        """
        # Check if the model is trained
        if self.model is None:
            logger.error("Model is not trained yet.")
            return 0.0
        
        # Check if indices are in bounds
        if user_idx >= self.n_users or item_idx >= self.n_items:
            logger.warning(f"User index {user_idx} or item index {item_idx} out of bounds.")
            return 0.0
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Convert indices to tensors
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
            
            # Get prediction
            prediction = self.model(user_tensor, item_tensor)
            
            # Convert to Python float
            return prediction.item()
    
    def recommend(self, user_idx, n=10, exclude_rated=True, rated_items=None):
        """
        Generate top N recommendations for a user.
        
        Args:
            user_idx (int): User index
            n (int): Number of recommendations
            exclude_rated (bool): Whether to exclude already rated items
            rated_items (list): List of items already rated by the user
            
        Returns:
            list: List of (item_idx, predicted_rating) tuples
        """
        # Check if the model is trained
        if self.model is None:
            logger.error("Model is not trained yet.")
            return []
        
        # Check if user index is in bounds
        if user_idx >= self.n_users:
            logger.warning(f"User index {user_idx} out of bounds.")
            return []
        
        # Items to exclude
        exclude_items = set()
        if exclude_rated and rated_items is not None:
            exclude_items = set(rated_items)
        
        # Generate predictions for all items
        predictions = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Process items in batches for efficiency
            batch_size = 1024
            for start_idx in range(0, self.n_items, batch_size):
                end_idx = min(start_idx + batch_size, self.n_items)
                item_indices = list(range(start_idx, end_idx))
                
                # Filter out already rated items
                item_indices = [idx for idx in item_indices if idx not in exclude_items]
                
                if not item_indices:
                    continue
                
                # Create tensors
                user_tensor = torch.tensor([user_idx] * len(item_indices), 
                                          dtype=torch.long).to(self.device)
                item_tensor = torch.tensor(item_indices, 
                                          dtype=torch.long).to(self.device)
                
                # Get predictions
                batch_predictions = self.model(user_tensor, item_tensor)
                
                # Store predictions
                for i, item_idx in enumerate(item_indices):
                    pred_rating = batch_predictions[i].item()
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
        # Check if the model is trained
        if self.model is None:
            logger.error("Model is not trained yet.")
            return
        
        # Create a dictionary with model parameters
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'n_factors': self.n_factors,
            'hidden_dims': self.hidden_dims,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save to file
        torch.save(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path, device=None):
        """
        Load a saved model.
        
        Args:
            file_path (str): Path to the saved model
            device (str): Device to run on ('cuda' or 'cpu')
            
        Returns:
            NeuralRecommender: Loaded model
        """
        # Determine device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # Load model data
        model_data = torch.load(file_path, map_location=device)
        
        # Create a new instance
        recommender = cls(
            n_factors=model_data['n_factors'],
            hidden_dims=model_data['hidden_dims'],
            device=device
        )
        
        # Set attributes
        recommender.n_users = model_data['n_users']
        recommender.n_items = model_data['n_items']
        recommender.train_losses = model_data['train_losses']
        recommender.val_losses = model_data['val_losses']
        
        # Create and load model
        recommender.model = NeuralCollaborativeFiltering(
            n_users=recommender.n_users,
            n_items=recommender.n_items,
            n_factors=recommender.n_factors,
            hidden_dims=recommender.hidden_dims
        ).to(device)
        
        # Load state dictionary
        recommender.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"Model loaded from {file_path}")
        return recommender