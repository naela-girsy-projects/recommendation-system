import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import List, Dict, Optional, Union
import pickle
import json
import time
from datetime import datetime
import requests

# Import recommendation models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.collaborative_filtering import MemoryBasedCF, MatrixFactorizationCF
from models.content_based import ContentBasedRecommender
from models.neural_network import NeuralRecommender
from data.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TMDB API Configuration
TMDB_API_KEY = "your_api_key_here"  # Replace with your TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_CACHE = {}  # Cache for poster URLs

def get_movie_poster_url(title, year=None):
    """Fetch movie poster URL from TMDB API with caching."""
    cache_key = f"{title}_{year}" if year else title
    
    # Check cache first
    if cache_key in POSTER_CACHE:
        return POSTER_CACHE[cache_key]
    
    # Fetch from API if not in cache
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year
    }
    
    try:
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    # Cache the result
                    POSTER_CACHE[cache_key] = poster_url
                    return poster_url
    except Exception as e:
        logger.error(f"Error fetching poster for {title}: {e}")
    
    # Cache the None result to avoid repeated failed requests
    POSTER_CACHE[cache_key] = None
    return None

# Define API models (request and response schemas)
class RatingRequest(BaseModel):
    """Request model for rating an item."""
    user_id: int
    movie_id: int
    rating: float

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_id: int
    n: int = 10
    algorithm: str = "hybrid"  # Options: "cf", "content", "nn", "hybrid"
    include_rated: bool = False
    diversity: float = 0.3  # 0.0 to 1.0, higher means more diverse recommendations

class UserProfileRequest(BaseModel):
    """Request model for getting or updating a user profile."""
    user_id: int
    preferences: Optional[Dict[str, float]] = None

class Movie(BaseModel):
    """Model for movie data."""
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None
    poster_url: Optional[str] = None
    description: Optional[str] = None

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[Movie]
    algorithm: str
    explanation: Optional[str] = None
    timestamp: str

# Create FastAPI app
app = FastAPI(
    title="Personalized Recommendation System API",
    description="API for serving personalized movie recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables to store loaded data and models
DATA = {
    "ratings_df": None,
    "movies_df": None,
    "user_id_map": None,
    "movie_id_map": None,
    "train_matrix": None
}

MODELS = {
    "cf_item": None,
    "cf_user": None,
    "cf_matrix": None,
    "content_based": None,
    "neural_network": None
}

# Paths to data and models
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
USER_PROFILES_PATH = os.path.join(PROCESSED_DIR, "user_profiles.json")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "collaborative_filtering"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "content_based"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "neural_network"), exist_ok=True)

# Initialize user profiles dictionary
USER_PROFILES = {}

def load_data():
    """Load all necessary data files."""
    logger.info("Loading data...")
    
    # Initialize data pipeline
    pipeline = DataPipeline(data_dir=DATA_DIR)
    
    # Load ratings and movies data
    ratings_df, movies_df = pipeline.load_data()
    
    # Load user and movie ID mappings
    user_id_map_path = os.path.join(PROCESSED_DIR, "user_id_map.csv")
    movie_id_map_path = os.path.join(PROCESSED_DIR, "movie_id_map.csv")
    
    if os.path.exists(user_id_map_path) and os.path.exists(movie_id_map_path):
        user_id_map = pd.read_csv(user_id_map_path)
        movie_id_map = pd.read_csv(movie_id_map_path)
        
        # Convert to dictionaries
        user_id_map = dict(zip(user_id_map['userId'], user_id_map['user_idx']))
        movie_id_map = dict(zip(movie_id_map['movieId'], movie_id_map['movie_idx']))
    else:
        # Process data to create mappings
        ratings_df, movies_df = pipeline.preprocess_data(ratings_df, movies_df)
        
        # Extract mappings
        user_id_map = {
            userId: user_idx for user_idx, userId in 
            enumerate(ratings_df['userId'].unique())
        }
        movie_id_map = {
            movieId: movie_idx for movie_idx, movieId in 
            enumerate(movies_df['movieId'].unique())
        }
    
    # Load interaction matrix
    interaction_matrix_path = os.path.join(PROCESSED_DIR, "interaction_matrix.npz")
    if os.path.exists(interaction_matrix_path):
        import scipy.sparse as sp
        train_matrix = sp.load_npz(interaction_matrix_path)
    else:
        # Process data to create matrix
        pipeline.process_pipeline()
        import scipy.sparse as sp
        train_matrix = sp.load_npz(interaction_matrix_path)
    
    # Load user profiles
    load_user_profiles()
    
    # Store data in global variables
    DATA["ratings_df"] = ratings_df
    DATA["movies_df"] = movies_df
    DATA["user_id_map"] = user_id_map
    DATA["movie_id_map"] = movie_id_map
    DATA["train_matrix"] = train_matrix
    
    logger.info(f"Data loaded successfully. {len(ratings_df)} ratings, {len(movies_df)} movies.")

def load_models():
    """Load all recommendation models."""
    logger.info("Loading recommendation models...")
    
    # Load item-based CF model
    cf_item_path = os.path.join(MODELS_DIR, "collaborative_filtering", "item_based_cf.pkl")
    if os.path.exists(cf_item_path):
        try:
            MODELS["cf_item"] = MemoryBasedCF.load_model(cf_item_path)
            MODELS["cf_item"].train_matrix = DATA["train_matrix"]
            logger.info("Item-based CF model loaded.")
        except Exception as e:
            logger.error(f"Error loading item-based CF model: {e}")
    else:
        # Train a new model
        logger.info("Training new item-based CF model...")
        item_cf = MemoryBasedCF(method='item', k=20)
        item_cf.fit(DATA["train_matrix"])
        MODELS["cf_item"] = item_cf
        
        # Save the model
        try:
            item_cf.save_model(cf_item_path)
            logger.info("Item-based CF model saved.")
        except Exception as e:
            logger.error(f"Error saving item-based CF model: {e}")
    
    # Load user-based CF model
    cf_user_path = os.path.join(MODELS_DIR, "collaborative_filtering", "user_based_cf.pkl")
    if os.path.exists(cf_user_path):
        try:
            MODELS["cf_user"] = MemoryBasedCF.load_model(cf_user_path)
            MODELS["cf_user"].train_matrix = DATA["train_matrix"]
            logger.info("User-based CF model loaded.")
        except Exception as e:
            logger.error(f"Error loading user-based CF model: {e}")
    else:
        # Train a new model
        logger.info("Training new user-based CF model...")
        user_cf = MemoryBasedCF(method='user', k=20)
        user_cf.fit(DATA["train_matrix"])
        MODELS["cf_user"] = user_cf
        
        # Save the model
        try:
            user_cf.save_model(cf_user_path)
            logger.info("User-based CF model saved.")
        except Exception as e:
            logger.error(f"Error saving user-based CF model: {e}")
    
    # Load matrix factorization CF model
    cf_matrix_path = os.path.join(MODELS_DIR, "collaborative_filtering", "matrix_cf.pkl")
    if os.path.exists(cf_matrix_path):
        try:
            MODELS["cf_matrix"] = MatrixFactorizationCF.load_model(cf_matrix_path)
            logger.info("Matrix factorization CF model loaded.")
        except Exception as e:
            logger.error(f"Error loading matrix factorization CF model: {e}")
    else:
        # Train a new model
        logger.info("Training new matrix factorization CF model...")
        matrix_cf = MatrixFactorizationCF(n_factors=50)
        matrix_cf.fit(DATA["train_matrix"])
        MODELS["cf_matrix"] = matrix_cf
        
        # Save the model
        try:
            matrix_cf.save_model(cf_matrix_path)
            logger.info("Matrix factorization CF model saved.")
        except Exception as e:
            logger.error(f"Error saving matrix factorization CF model: {e}")
    
    # Load content-based model
    content_based_path = os.path.join(MODELS_DIR, "content_based", "content_recommender.pkl")
    if os.path.exists(content_based_path):
        try:
            MODELS["content_based"] = ContentBasedRecommender.load_model(
                content_based_path, DATA["movies_df"])
            logger.info("Content-based model loaded.")
        except Exception as e:
            logger.error(f"Error loading content-based model: {e}")
    else:
        # Train a new model
        logger.info("Training new content-based model...")
        content_based = ContentBasedRecommender()
        content_based.fit(DATA["movies_df"])
        MODELS["content_based"] = content_based
        
        # Save the model
        try:
            content_based.save_model(content_based_path)
            logger.info("Content-based model saved.")
        except Exception as e:
            logger.error(f"Error saving content-based model: {e}")
    
    # Load neural network model
    nn_path = os.path.join(MODELS_DIR, "neural_network", "neural_recommender.pt")
    if os.path.exists(nn_path):
        try:
            MODELS["neural_network"] = NeuralRecommender.load_model(nn_path)
            logger.info("Neural network model loaded.")
        except Exception as e:
            logger.error(f"Error loading neural network model: {e}")
    else:
        # Skip training neural network model here as it's more resource-intensive
        # It would typically be trained offline and loaded here
        logger.warning("Neural network model not found. Skipping...")
    
    logger.info("Models loaded successfully.")

def load_user_profiles():
    """Load user profiles from file."""
    global USER_PROFILES
    
    if os.path.exists(USER_PROFILES_PATH):
        try:
            with open(USER_PROFILES_PATH, 'r') as f:
                USER_PROFILES = json.load(f)
            logger.info(f"Loaded {len(USER_PROFILES)} user profiles.")
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
            USER_PROFILES = {}
    else:
        logger.info("No user profiles file found. Starting with empty profiles.")
        USER_PROFILES = {}

def save_user_profiles():
    """Save user profiles to file."""
    try:
        with open(USER_PROFILES_PATH, 'w') as f:
            json.dump(USER_PROFILES, f)
        logger.info(f"Saved {len(USER_PROFILES)} user profiles.")
    except Exception as e:
        logger.error(f"Error saving user profiles: {e}")

def update_models(background_tasks: BackgroundTasks):
    """Update models with new data (to be run in the background)."""
    logger.info("Updating recommendation models...")
    
    try:
        # Reload data
        load_data()
        
        # Retrain models
        # Item-based CF
        item_cf = MemoryBasedCF(method='item', k=20)
        item_cf.fit(DATA["train_matrix"])
        MODELS["cf_item"] = item_cf
        item_cf.save_model(os.path.join(MODELS_DIR, "collaborative_filtering", "item_based_cf.pkl"))
        
        # User-based CF
        user_cf = MemoryBasedCF(method='user', k=20)
        user_cf.fit(DATA["train_matrix"])
        MODELS["cf_user"] = user_cf
        user_cf.save_model(os.path.join(MODELS_DIR, "collaborative_filtering", "user_based_cf.pkl"))
        
        # Matrix factorization CF
        matrix_cf = MatrixFactorizationCF(n_factors=50)
        matrix_cf.fit(DATA["train_matrix"])
        MODELS["cf_matrix"] = matrix_cf
        matrix_cf.save_model(os.path.join(MODELS_DIR, "collaborative_filtering", "matrix_cf.pkl"))
        
        # Content-based model (unchanged by new ratings, skip retraining)
        
        logger.info("Models updated successfully.")
    except Exception as e:
        logger.error(f"Error updating models: {e}")

def external_to_internal_ids(user_id, movie_id=None):
    """
    Convert external IDs to internal indices.
    
    Args:
        user_id (int): External user ID
        movie_id (int, optional): External movie ID
        
    Returns:
        tuple: (user_idx, movie_idx)
    """
    # Convert user ID
    user_idx = DATA["user_id_map"].get(user_id)
    
    # If user doesn't exist, create a new entry
    if user_idx is None:
        user_idx = len(DATA["user_id_map"])
        DATA["user_id_map"][user_id] = user_idx
        
        # Save the updated mapping
        df = pd.DataFrame(list(DATA["user_id_map"].items()), 
                         columns=['userId', 'user_idx'])
        df.to_csv(os.path.join(PROCESSED_DIR, "user_id_map.csv"), index=False)
    
    # Convert movie ID if provided
    movie_idx = None
    if movie_id is not None:
        movie_idx = DATA["movie_id_map"].get(movie_id)
        
        # If movie doesn't exist, return None
        if movie_idx is None:
            # Check if it exists in the movies dataframe
            if movie_id in DATA["movies_df"]["movieId"].values:
                # Add to mapping
                movie_idx = len(DATA["movie_id_map"])
                DATA["movie_id_map"][movie_id] = movie_idx
                
                # Save the updated mapping
                df = pd.DataFrame(list(DATA["movie_id_map"].items()), 
                                 columns=['movieId', 'movie_idx'])
                df.to_csv(os.path.join(PROCESSED_DIR, "movie_id_map.csv"), index=False)
    
    return user_idx, movie_idx

def internal_to_external_ids(user_idx, movie_idx=None):
    """
    Convert internal indices to external IDs.
    
    Args:
        user_idx (int): Internal user index
        movie_idx (int, optional): Internal movie index
        
    Returns:
        tuple: (user_id, movie_id)
    """
    # Get user ID
    user_id = None
    for ext_id, idx in DATA["user_id_map"].items():
        if idx == user_idx:
            user_id = ext_id
            break
    
    # Get movie ID if provided
    movie_id = None
    if movie_idx is not None:
        for ext_id, idx in DATA["movie_id_map"].items():
            if idx == movie_idx:
                movie_id = ext_id
                break
    
    return user_id, movie_id

def get_recommendations(user_idx, n=10, algorithm="hybrid", include_rated=False, diversity=0.3):
    """
    Get recommendations for a user using the specified algorithm.
    
    Args:
        user_idx (int): Internal user index
        n (int): Number of recommendations
        algorithm (str): Algorithm to use
        include_rated (bool): Whether to include already rated items
        diversity (float): Level of diversity in recommendations
        
    Returns:
        list: List of (movie_idx, score) tuples
    """
    # Check if user index is out of range
    if DATA["train_matrix"] is not None and user_idx >= DATA["train_matrix"].shape[0]:
        logger.warning(f"User index {user_idx} is out of range. Using popular items for new user.")
        # For new users, provide popular items
        try:
            # First approach: Try to get the most popular items using movieId
            if DATA["ratings_df"] is not None and 'movieId' in DATA["ratings_df"].columns:
                popular_movies = DATA["ratings_df"].groupby('movieId')['rating'].count().sort_values(ascending=False).head(n*2)
                recommendations = []
                
                for movie_id in popular_movies.index:
                    if movie_id in DATA["movie_id_map"]:
                        movie_idx = DATA["movie_id_map"][movie_id]
                        recommendations.append((int(movie_idx), 5.0))
                    
                    if len(recommendations) >= n:
                        break
                
                if recommendations:
                    return recommendations, "Popular recommendations for new user"
            
            # Second approach: Try using movie_idx directly if it exists
            if DATA["ratings_df"] is not None and 'movie_idx' in DATA["ratings_df"].columns:
                popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                return recommendations, "Popular recommendations for new user"
                
            # Third approach: Get random sample from movies dataframe
            sample_size = min(n, len(DATA["movies_df"]))
            random_movies = DATA["movies_df"].sample(n=sample_size)
            recommendations = []
            
            for _, row in random_movies.iterrows():
                if 'movieId' in row and row['movieId'] in DATA["movie_id_map"]:
                    movie_idx = DATA["movie_id_map"][row['movieId']]
                    recommendations.append((int(movie_idx), 5.0))
            
            if recommendations:
                return recommendations, "Random movie recommendations for new user"
            
            # Last resort: Generate random indices
            matrix_width = DATA["train_matrix"].shape[1]
            random_indices = np.random.choice(matrix_width, size=min(n, matrix_width), replace=False)
            recommendations = [(int(idx), 5.0) for idx in random_indices]
            return recommendations, "Random recommendations for new user"
            
        except Exception as e:
            logger.error(f"Error generating recommendations for new user: {e}")
            # Absolute last resort: hardcoded recommendations for popular movie indices
            recommendations = [(i, 5.0) for i in range(min(n, 100))]  # Use first 100 movie indices
            return recommendations, "Fallback recommendations for new user"
    
    # Get already rated items
    rated_items = None
    if not include_rated:
        if DATA["train_matrix"] is not None:
            try:
                # Get rated items from the interaction matrix
                rated_items = set(DATA["train_matrix"][user_idx].nonzero()[1])
            except IndexError:
                # Handle index error (e.g., user_idx out of bounds)
                rated_items = set()
                logger.warning(f"User index {user_idx} out of bounds. Using empty rated items set.")
    
    recommendations = []
    explanation = ""
    
    if algorithm == "cf":
        # Use item-based collaborative filtering
        if MODELS["cf_item"] is not None:
            try:
                recommendations = MODELS["cf_item"].recommend(
                    user_idx, n=n, exclude_rated=not include_rated)
                explanation = "Based on similar items to those you've enjoyed"
            except Exception as e:
                logger.error(f"Error with item-based CF: {e}. Falling back to popular items.")
                # Fallback to popular items
                if DATA["ratings_df"] is not None:
                    popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                    recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                    explanation = "Popular movies you might enjoy"
        else:
            logger.warning("Item-based CF model not available. Falling back to matrix factorization.")
            if MODELS["cf_matrix"] is not None:
                try:
                    recommendations = MODELS["cf_matrix"].recommend(
                        user_idx, n=n, exclude_rated=not include_rated, train_matrix=DATA["train_matrix"])
                    explanation = "Based on patterns in your ratings"
                except Exception as e:
                    logger.error(f"Error with matrix factorization: {e}. Falling back to popular items.")
                    # Fallback to popular items
                    if DATA["ratings_df"] is not None:
                        popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                        recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                        explanation = "Popular movies you might enjoy"
    
    elif algorithm == "content":
        # Use content-based recommender
        if MODELS["content_based"] is not None:
            try:
                recommendations = MODELS["content_based"].recommend_for_user(
                    user_idx, DATA["train_matrix"], n=n, diversity_level=diversity)
                explanation = "Based on the types of movies you've enjoyed"
            except Exception as e:
                logger.error(f"Error with content-based recommender: {e}. Falling back to popular items.")
                # Fallback to genre-based or popular items
                if len(USER_PROFILES.get(str(user_idx), {}).get("preferences", {})) > 0:
                    # If user has preferences, use them
                    preferences = USER_PROFILES[str(user_idx)]["preferences"]
                    # Filter movies by preferred genres
                    top_genre = max(preferences.items(), key=lambda x: x[1])[0]
                    genre_matches = []
                    for idx, row in DATA["movies_df"].iterrows():
                        if top_genre.lower() in row["genres"].lower():
                            if "movie_idx" in row:
                                genre_matches.append((row["movie_idx"], 5.0))
                            elif "movieId" in row and row["movieId"] in DATA["movie_id_map"]:
                                genre_matches.append((DATA["movie_id_map"][row["movieId"]], 5.0))
                    recommendations = genre_matches[:n]
                    explanation = f"Movies in your preferred genre: {top_genre}"
                else:
                    # Fallback to popular items
                    if DATA["ratings_df"] is not None:
                        popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                        recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                        explanation = "Popular movies you might enjoy"
        else:
            logger.warning("Content-based model not available. Falling back to CF.")
            if MODELS["cf_item"] is not None:
                try:
                    recommendations = MODELS["cf_item"].recommend(
                        user_idx, n=n, exclude_rated=not include_rated)
                    explanation = "Based on similar items to those you've enjoyed"
                except Exception as e:
                    logger.error(f"Error with item-based CF: {e}. Falling back to popular items.")
                    # Fallback to popular items
                    if DATA["ratings_df"] is not None:
                        popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                        recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                        explanation = "Popular movies you might enjoy"
    
    elif algorithm == "nn":
        # Use neural network recommender
        if MODELS["neural_network"] is not None:
            try:
                recommendations = MODELS["neural_network"].recommend(
                    user_idx, n=n, exclude_rated=not include_rated, rated_items=rated_items)
                explanation = "Using deep learning to analyze your preferences"
            except Exception as e:
                logger.error(f"Error with neural network: {e}. Falling back to popular items.")
                # Fallback to popular items
                if DATA["ratings_df"] is not None:
                    popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                    recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                    explanation = "Popular movies you might enjoy"
        else:
            logger.warning("Neural network model not available. Falling back to matrix factorization.")
            if MODELS["cf_matrix"] is not None:
                try:
                    recommendations = MODELS["cf_matrix"].recommend(
                        user_idx, n=n, exclude_rated=not include_rated, train_matrix=DATA["train_matrix"])
                    explanation = "Based on patterns in your ratings"
                except Exception as e:
                    logger.error(f"Error with matrix factorization: {e}. Falling back to popular items.")
                    # Fallback to popular items
                    if DATA["ratings_df"] is not None:
                        popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                        recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                        explanation = "Popular movies you might enjoy"
    
    elif algorithm == "hybrid":
        # Combine recommendations from multiple algorithms
        all_recommendations = {}
        weights = {
            "cf_item": 0.3,
            "cf_matrix": 0.2,
            "content": 0.3,
            "nn": 0.2
        }
        
        # Get recommendations from each available model
        try:
            if MODELS["cf_item"] is not None:
                try:
                    cf_item_recs = MODELS["cf_item"].recommend(
                        user_idx, n=n*2, exclude_rated=not include_rated)
                    for item_idx, score in cf_item_recs:
                        all_recommendations[item_idx] = all_recommendations.get(item_idx, 0) + score * weights["cf_item"]
                except Exception as e:
                    logger.error(f"Error getting CF item recommendations: {e}")
            
            if MODELS["cf_matrix"] is not None:
                try:
                    cf_matrix_recs = MODELS["cf_matrix"].recommend(
                        user_idx, n=n*2, exclude_rated=not include_rated, train_matrix=DATA["train_matrix"])
                    for item_idx, score in cf_matrix_recs:
                        all_recommendations[item_idx] = all_recommendations.get(item_idx, 0) + score * weights["cf_matrix"]
                except Exception as e:
                    logger.error(f"Error getting matrix factorization recommendations: {e}")
            
            if MODELS["content_based"] is not None:
                try:
                    content_recs = MODELS["content_based"].recommend_for_user(
                        user_idx, DATA["train_matrix"], n=n*2, diversity_level=diversity)
                    for item_idx, score in content_recs:
                        all_recommendations[item_idx] = all_recommendations.get(item_idx, 0) + score * weights["content"]
                except Exception as e:
                    logger.error(f"Error getting content-based recommendations: {e}")
            
            if MODELS["neural_network"] is not None:
                try:
                    nn_recs = MODELS["neural_network"].recommend(
                        user_idx, n=n*2, exclude_rated=not include_rated, rated_items=rated_items)
                    for item_idx, score in nn_recs:
                        all_recommendations[item_idx] = all_recommendations.get(item_idx, 0) + score * weights["nn"]
                except Exception as e:
                    logger.error(f"Error getting neural network recommendations: {e}")
            
            # If we got any recommendations, convert to list and sort
            if all_recommendations:
                recommendations = [(item_idx, score) for item_idx, score in all_recommendations.items()]
                recommendations.sort(key=lambda x: x[1], reverse=True)
                recommendations = recommendations[:n]
                explanation = "Using a blend of techniques to provide personalized recommendations"
            else:
                # If no recommendations, fall back to popular items
                if DATA["ratings_df"] is not None and 'movie_idx' in DATA["ratings_df"].columns:
                    popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                    recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                    explanation = "Popular movies you might enjoy"
                elif DATA["ratings_df"] is not None and 'movieId' in DATA["ratings_df"].columns:
                    popular_movies = DATA["ratings_df"].groupby('movieId')['rating'].count().sort_values(ascending=False).head(n*2)
                    recommendations = []
                    for movie_id in popular_movies.index:
                        if movie_id in DATA["movie_id_map"]:
                            movie_idx = DATA["movie_id_map"][movie_id]
                            recommendations.append((int(movie_idx), 5.0))
                        if len(recommendations) >= n:
                            break
                    explanation = "Popular movies you might enjoy"
                
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}. Falling back to popular items.")
            # Fallback to popular items
            if DATA["ratings_df"] is not None and 'movie_idx' in DATA["ratings_df"].columns:
                popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                explanation = "Popular movies you might enjoy"
            elif DATA["ratings_df"] is not None and 'movieId' in DATA["ratings_df"].columns:
                popular_movies = DATA["ratings_df"].groupby('movieId')['rating'].count().sort_values(ascending=False).head(n*2)
                recommendations = []
                for movie_id in popular_movies.index:
                    if movie_id in DATA["movie_id_map"]:
                        movie_idx = DATA["movie_id_map"][movie_id]
                        recommendations.append((int(movie_idx), 5.0))
                    if len(recommendations) >= n:
                        break
                explanation = "Popular movies you might enjoy"
    
    # If no recommendations were generated, try various fallback strategies
    if not recommendations:
        logger.warning(f"No recommendations generated for user {user_idx}. Using fallback strategies.")
        
        # Fallback Strategy 1: Use popular items if available
        try:
            if DATA["ratings_df"] is not None:
                if 'movie_idx' in DATA["ratings_df"].columns:
                    popular_items = DATA["ratings_df"].groupby("movie_idx")["rating"].count().sort_values(ascending=False).head(n)
                    recommendations = [(int(idx), 5.0) for idx in popular_items.index]
                    explanation = "Popular movies you might enjoy"
                elif 'movieId' in DATA["ratings_df"].columns:
                    popular_movies = DATA["ratings_df"].groupby('movieId')['rating'].count().sort_values(ascending=False).head(n*2)
                    recommendations = []
                    for movie_id in popular_movies.index:
                        if movie_id in DATA["movie_id_map"]:
                            movie_idx = DATA["movie_id_map"][movie_id]
                            recommendations.append((int(movie_idx), 5.0))
                        if len(recommendations) >= n:
                            break
                    explanation = "Popular movies you might enjoy"
        except Exception as e:
            logger.error(f"Error using popular items fallback: {e}")
        
        # Fallback Strategy 2: Use random movies from movies dataframe
        if not recommendations:
            try:
                if DATA["movies_df"] is not None:
                    sample_size = min(n, len(DATA["movies_df"]))
                    random_movies = DATA["movies_df"].sample(n=sample_size)
                    recommendations = []
                    
                    for _, row in random_movies.iterrows():
                        if 'movieId' in row and row['movieId'] in DATA["movie_id_map"]:
                            movie_idx = DATA["movie_id_map"][row['movieId']]
                            recommendations.append((int(movie_idx), 5.0))
                    
                    if recommendations:
                        explanation = "Random movie recommendations"
            except Exception as e:
                logger.error(f"Error using random movies fallback: {e}")
        
        # Fallback Strategy 3: Use random indices from train matrix
        if not recommendations and DATA["train_matrix"] is not None:
            try:
                matrix_width = DATA["train_matrix"].shape[1]
                random_indices = np.random.choice(matrix_width, size=min(n, matrix_width), replace=False)
                recommendations = [(int(idx), 5.0) for idx in random_indices]
                explanation = "Random movie recommendations"
            except Exception as e:
                logger.error(f"Error using random indices fallback: {e}")
        
        # Ultimate Fallback: Use hardcoded indices
        if not recommendations:
            logger.warning("Using hardcoded fallback recommendations")
            recommendations = [(i, 5.0) for i in range(min(n, 100))]
            explanation = "Fallback movie recommendations"
    
    return recommendations, explanation

# API Endpoints
@app.on_event("startup")
def startup_event():
    """Load data and models when the API starts."""
    load_data()
    load_models()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Recommendation System API", "status": "active"}


@app.post("/recommendations")
def get_user_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.
    
    Args:
        request (RecommendationRequest): Request with user_id, n, algorithm, etc.
        
    Returns:
        RecommendationResponse: Response with recommendations
    """
    try:
        # Convert user ID to internal index
        user_idx, _ = external_to_internal_ids(request.user_id)
        
        # Get recommendations
        recommendations, explanation = get_recommendations(
            user_idx, 
            n=request.n, 
            algorithm=request.algorithm,
            include_rated=request.include_rated,
            diversity=request.diversity
        )
        
        # Convert recommendations to movies
        movie_recommendations = []
        
        for movie_idx, score in recommendations:
            # Convert internal index to external ID
            _, movie_id = internal_to_external_ids(None, movie_idx)
            
            if movie_id is None:
                logger.warning(f"Could not find external ID for movie_idx {movie_idx}")
                continue
                
            # Get movie details
            movie_data = DATA["movies_df"][DATA["movies_df"]["movieId"] == movie_id]
            
            if not movie_data.empty:
                try:
                    # Extract year from title if available
                    year = None
                    if "(" in movie_data["title"].values[0] and ")" in movie_data["title"].values[0]:
                        try:
                            year = int(movie_data["title"].values[0].split("(")[-1].split(")")[0])
                        except ValueError:
                            year = None
                    
                    # Get movie poster URL
                    poster_url = get_movie_poster_url(
                        movie_data["title"].values[0].split(" (")[0] if "(" in movie_data["title"].values[0] else movie_data["title"].values[0],
                        year=year
                    )
                    
                    # Extract movie details
                    movie = Movie(
                        movie_id=int(movie_id),
                        title=movie_data["title"].values[0],
                        genres=movie_data["genres"].values[0],
                        year=year,
                        poster_url=poster_url
                    )
                    movie_recommendations.append(movie)
                except Exception as e:
                    logger.error(f"Error creating movie object for movie_id {movie_id}: {e}")
            else:
                logger.warning(f"Movie ID {movie_id} not found in movies dataframe")
        
        # Create response
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=movie_recommendations,
            algorithm=request.algorithm,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rate")
def rate_item(rating_request: RatingRequest, background_tasks: BackgroundTasks):
    """
    Add or update a user's rating for an item.
    
    Args:
        rating_request (RatingRequest): Request with user_id, movie_id, rating
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        dict: Status message
    """
    try:
        # Convert IDs to internal indices
        user_idx, movie_idx = external_to_internal_ids(
            rating_request.user_id, rating_request.movie_id)
        
        if user_idx is None or movie_idx is None:
            raise HTTPException(
                status_code=404, 
                detail=f"User ID {rating_request.user_id} or Movie ID {rating_request.movie_id} not found"
            )
        
        # Update ratings dataframe
        new_rating = pd.DataFrame({
            'userId': [rating_request.user_id],
            'movieId': [rating_request.movie_id],
            'rating': [rating_request.rating],
            'timestamp': [int(time.time())],
            'user_idx': [user_idx],
            'movie_idx': [movie_idx]
        })
        
        # Remove existing rating if it exists
        DATA["ratings_df"] = DATA["ratings_df"][
            ~((DATA["ratings_df"]["userId"] == rating_request.user_id) & 
              (DATA["ratings_df"]["movieId"] == rating_request.movie_id))
        ]
        
        # Add new rating
        DATA["ratings_df"] = pd.concat([DATA["ratings_df"], new_rating], ignore_index=True)
        
        # Save updated ratings
        DATA["ratings_df"].to_csv(os.path.join(PROCESSED_DIR, "ratings.csv"), index=False)
        
        # Update interaction matrix
        if DATA["train_matrix"] is not None:
            # Ensure matrix dimensions are sufficient
            if user_idx >= DATA["train_matrix"].shape[0] or movie_idx >= DATA["train_matrix"].shape[1]:
                # Resize matrix
                import scipy.sparse as sp
                old_matrix = DATA["train_matrix"]
                new_shape = (
                    max(user_idx + 1, old_matrix.shape[0]),
                    max(movie_idx + 1, old_matrix.shape[1])
                )
                
                # Create new matrix
                new_matrix = sp.csr_matrix(new_shape)
                
                # Copy old data
                new_matrix[:old_matrix.shape[0], :old_matrix.shape[1]] = old_matrix
                
                # Update reference
                DATA["train_matrix"] = new_matrix
            
            # Update rating
            DATA["train_matrix"][user_idx, movie_idx] = rating_request.rating
            
            # Save matrix
            import scipy.sparse as sp
            sp.save_npz(os.path.join(PROCESSED_DIR, "interaction_matrix.npz"), DATA["train_matrix"])
        
        # Schedule model update in the background
        background_tasks.add_task(update_models, background_tasks)
        
        return {"status": "success", "message": "Rating added successfully"}
    
    except Exception as e:
        logger.error(f"Error adding rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}")
def get_user_profile(user_id: int):
    """
    Get a user's profile and preferences.
    
    Args:
        user_id (int): User ID
        
    Returns:
        dict: User profile
    """
    try:
        # Check if user profile exists
        if str(user_id) in USER_PROFILES:
            return USER_PROFILES[str(user_id)]
        
        # Create a new profile
        profile = {
            "user_id": user_id,
            "preferences": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add profile
        USER_PROFILES[str(user_id)] = profile
        
        # Save profiles
        save_user_profiles()
        
        return profile
    
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile")
def update_user_profile(profile_request: UserProfileRequest):
    """
    Update a user's profile and preferences.
    
    Args:
        profile_request (UserProfileRequest): Request with user_id and preferences
        
    Returns:
        dict: Updated user profile
    """
    try:
        user_id = str(profile_request.user_id)
        
        # Check if user profile exists
        if user_id in USER_PROFILES:
            profile = USER_PROFILES[user_id]
        else:
            # Create a new profile
            profile = {
                "user_id": profile_request.user_id,
                "preferences": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        
        # Update preferences if provided
        if profile_request.preferences:
            profile["preferences"].update(profile_request.preferences)
            profile["last_updated"] = datetime.now().isoformat()
        
        # Save profile
        USER_PROFILES[user_id] = profile
        save_user_profiles()
        
        return profile
    
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}")
def get_movie_details(movie_id: int):
    """
    Get details for a specific movie.
    
    Args:
        movie_id (int): Movie ID
        
    Returns:
        Movie: Movie details
    """
    try:
        # Find movie in dataframe
        movie_data = DATA["movies_df"][DATA["movies_df"]["movieId"] == movie_id]
        
        if movie_data.empty:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found")
        
        # Extract year from title if available
        year = None
        if "(" in movie_data["title"].values[0] and ")" in movie_data["title"].values[0]:
            try:
                year = int(movie_data["title"].values[0].split("(")[-1].split(")")[0])
            except ValueError:
                year = None
        
        # Get movie poster URL
        poster_url = get_movie_poster_url(
            movie_data["title"].values[0].split(" (")[0] if "(" in movie_data["title"].values[0] else movie_data["title"].values[0],
            year=year
        )
        
        # Extract movie details
        movie = Movie(
            movie_id=int(movie_id),
            title=movie_data["title"].values[0],
            genres=movie_data["genres"].values[0],
            year=year,
            poster_url=poster_url
        )
        
        return movie
    
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{movie_id}")
def get_similar_movies(movie_id: int, n: int = 10):
    """
    Get movies similar to a specific movie.
    
    Args:
        movie_id (int): Movie ID
        n (int): Number of similar movies to retrieve
        
    Returns:
        list: List of similar movies
    """
    try:
        # Convert movie ID to internal index
        _, movie_idx = external_to_internal_ids(None, movie_id)
        
        if movie_idx is None:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found")
        
        # Get similar movies using content-based recommender
        if MODELS["content_based"] is not None:
            similar_items = MODELS["content_based"].get_similar_items(movie_idx, n=n)
            
            # Convert to movie objects
            similar_movies = []
            
            for item_idx, similarity in similar_items:
                # Convert internal index to external ID
                _, similar_movie_id = internal_to_external_ids(None, item_idx)
                
                # Get movie details
                movie_data = DATA["movies_df"][DATA["movies_df"]["movieId"] == similar_movie_id]
                
                if not movie_data.empty:
                    # Extract year from title if available
                    year = None
                    if "(" in movie_data["title"].values[0] and ")" in movie_data["title"].values[0]:
                        try:
                            year = int(movie_data["title"].values[0].split("(")[-1].split(")")[0])
                        except ValueError:
                            year = None
                    
                    # Get movie poster URL
                    poster_url = get_movie_poster_url(
                        movie_data["title"].values[0].split(" (")[0] if "(" in movie_data["title"].values[0] else movie_data["title"].values[0],
                        year=year
                    )
                    
                    # Extract movie details
                    movie = Movie(
                        movie_id=int(similar_movie_id),
                        title=movie_data["title"].values[0],
                        genres=movie_data["genres"].values[0],
                        year=year,
                        poster_url=poster_url
                    )
                    similar_movies.append(movie)
            
            return similar_movies
        else:
            raise HTTPException(status_code=503, detail="Content-based recommender not available")
    
    except Exception as e:
        logger.error(f"Error getting similar movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)