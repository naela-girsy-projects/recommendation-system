# evaluation.py
"""
This module implements evaluation metrics and utilities for recommendation systems.
It allows comparing different algorithms using common metrics like RMSE, MAE, precision, recall, etc.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommenderEvaluator:
    """
    Evaluator for recommendation algorithms.
    Implements common evaluation metrics and benchmarking tools.
    """
    
    def __init__(self, metrics=None):
        """
        Initialize the evaluator with specified metrics.
        
        Args:
            metrics (list): List of metrics to compute
        """
        if metrics is None:
            # Default metrics
            self.metrics = ['rmse', 'mae', 'precision', 'recall', 'ndcg']
        else:
            self.metrics = metrics
        
        # Results storage
        self.results = {}
    
    def evaluate_rating_prediction(self, model, test_df, name=None):
        """
        Evaluate a model on rating prediction task.
        
        Args:
            model: Model with predict(user_idx, item_idx) method
            test_df (pd.DataFrame): Test data with user_idx, movie_idx, rating columns
            name (str): Name of the model (for reporting)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if name is None:
            name = type(model).__name__
        
        logger.info(f"Evaluating {name} on rating prediction task...")
        
        # Measure prediction time
        start_time = time.time()
        
        # Get actual and predicted ratings
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            user_idx = row['user_idx']
            movie_idx = row['movie_idx']
            true_rating = row['rating']
            
            try:
                pred_rating = model.predict(user_idx, movie_idx)
                y_true.append(true_rating)
                y_pred.append(pred_rating)
            except Exception as e:
                logger.warning(f"Error predicting rating for user {user_idx}, item {movie_idx}: {e}")
        
        prediction_time = time.time() - start_time
        
        # Compute metrics
        metrics = {}
        
        if 'rmse' in self.metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mae' in self.metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Log results
        logger.info(f"Evaluation results for {name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info(f"  Prediction time: {prediction_time:.2f} seconds for {len(y_true)} ratings")
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'prediction_time': prediction_time,
            'n_predictions': len(y_true)
        }
        
        return metrics
    
    def evaluate_top_n_recommendations(self, model, test_df, train_df, n=10, threshold=3.5, name=None):
        """
        Evaluate a model on top-N recommendation task.
        
        Args:
            model: Model with recommend(user_idx, n) method
            test_df (pd.DataFrame): Test data with user_idx, movie_idx, rating columns
            train_df (pd.DataFrame): Training data (to get already rated items)
            n (int): Number of recommendations to generate
            threshold (float): Rating threshold for relevant items
            name (str): Name of the model (for reporting)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if name is None:
            name = type(model).__name__
        
        logger.info(f"Evaluating {name} on top-{n} recommendation task...")
        
        # Convert test data to dictionary: user -> set of relevant items
        user_relevant_items = defaultdict(set)
        for _, row in test_df.iterrows():
            if row['rating'] >= threshold:  # Only consider items with rating >= threshold as relevant
                user_relevant_items[row['user_idx']].add(row['movie_idx'])
        
        # Convert train data to dictionary: user -> set of rated items
        user_rated_items = defaultdict(set)
        for _, row in train_df.iterrows():
            user_rated_items[row['user_idx']].add(row['movie_idx'])
        
        # Users to evaluate (must have at least one relevant item in test set)
        eval_users = [user for user, items in user_relevant_items.items() if items]
        
        if not eval_users:
            logger.warning("No users with relevant items in test set. Cannot evaluate.")
            return {}
        
        # Measure recommendation time
        start_time = time.time()
        
        # Compute metrics for each user
        precision_at_n = []
        recall_at_n = []
        ndcg_at_n = []
        
        for user_idx in eval_users:
            try:
                # Get user's recommendations
                recommendations = model.recommend(
                    user_idx, 
                    n=n, 
                    exclude_rated=True, 
                    rated_items=user_rated_items[user_idx]
                )
                
                # Extract recommended item IDs
                rec_item_ids = [item_idx for item_idx, _ in recommendations]
                
                # Get relevant items for this user
                relevant_items = user_relevant_items[user_idx]
                
                # Compute precision@N
                n_relevant_and_recommended = len(set(rec_item_ids) & relevant_items)
                precision = n_relevant_and_recommended / len(rec_item_ids) if rec_item_ids else 0
                precision_at_n.append(precision)
                
                # Compute recall@N
                recall = n_relevant_and_recommended / len(relevant_items) if relevant_items else 0
                recall_at_n.append(recall)
                
                # Compute NDCG@N
                if not rec_item_ids:
                    ndcg = 0
                else:
                    # Create relevance list (1 if item is relevant, 0 otherwise)
                    relevance = [1 if item_idx in relevant_items else 0 for item_idx in rec_item_ids]
                    
                    # Calculate DCG
                    dcg = relevance[0] + sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[1:]))
                    
                    # Calculate ideal DCG
                    ideal_relevance = [1] * min(len(relevant_items), n)
                    idcg = ideal_relevance[0] + sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[1:]))
                    
                    # Calculate NDCG
                    ndcg = dcg / idcg if idcg > 0 else 0
                
                ndcg_at_n.append(ndcg)
                
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_idx}: {e}")
        
        recommendation_time = time.time() - start_time
        
        # Compute average metrics
        metrics = {}
        
        if 'precision' in self.metrics:
            metrics[f'precision@{n}'] = np.mean(precision_at_n)
        
        if 'recall' in self.metrics:
            metrics[f'recall@{n}'] = np.mean(recall_at_n)
        
        if 'ndcg' in self.metrics:
            metrics[f'ndcg@{n}'] = np.mean(ndcg_at_n)
        
        # Log results
        logger.info(f"Evaluation results for {name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"  Recommendation time: {recommendation_time:.2f} seconds for {len(eval_users)} users")
        
        # Store results
        if name in self.results:
            self.results[name]['metrics'].update(metrics)
            self.results[name]['recommendation_time'] = recommendation_time
            self.results[name]['n_users'] = len(eval_users)
        else:
            self.results[name] = {
                'metrics': metrics,
                'recommendation_time': recommendation_time,
                'n_users': len(eval_users)
            }
        
        return metrics
    
    def compare_models(self, test_type='rating'):
        """
        Compare multiple models based on evaluation results.
        
        Args:
            test_type (str): Type of test ('rating' or 'recommendation')
            
        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No evaluation results to compare.")
            return None
        
        # Create comparison DataFrame
        data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            
            # Extract relevant metrics based on test type
            if test_type == 'rating':
                row = {
                    'Model': model_name,
                    'RMSE': metrics.get('rmse', float('nan')),
                    'MAE': metrics.get('mae', float('nan')),
                    'Time (s)': result.get('prediction_time', float('nan'))
                }
            else:  # recommendation
                # Find the top-N metrics
                top_n_metrics = {
                    k: v for k, v in metrics.items() 
                    if k.startswith('precision@') or k.startswith('recall@') or k.startswith('ndcg@')
                }
                
                row = {'Model': model_name}
                row.update(top_n_metrics)
                row['Time (s)'] = result.get('recommendation_time', float('nan'))
            
            data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(data)
        
        return comparison_df
    
    def plot_comparison(self, metric, figsize=(10, 6)):
        """
        Plot comparison of models based on a specific metric.
        
        Args:
            metric (str): Metric to compare (e.g., 'rmse', 'precision@10')
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Comparison plot
        """
        # Check if there are results
        if not self.results:
            logger.warning("No evaluation results to plot.")
            return None
        
        # Extract values for the specified metric
        models = []
        values = []
        
        for model_name, result in self.results.items():
            if metric in result['metrics']:
                models.append(model_name)
                values.append(result['metrics'][metric])
        
        if not models:
            logger.warning(f"No results for metric '{metric}'.")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine if lower or higher is better
        is_lower_better = metric.lower() in ['rmse', 'mae']
        
        # Sort by metric value
        sorted_indices = np.argsort(values)
        if not is_lower_better:
            sorted_indices = sorted_indices[::-1]  # Reverse for higher is better
        
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create bar plot
        bars = ax.bar(sorted_models, sorted_values)
        
        # Add labels
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        
        # Add values on bars
        for bar, value in zip(bars, sorted_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{value:.4f}', ha='center', va='bottom')
        
        return fig