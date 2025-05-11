# frontend/app.py
"""
Simple Flask web application for the recommendation system.
Provides a user interface to interact with the recommendation API.
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = 'recommendation_system_key'  # Change in production

# API configuration
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        try:
            user_id = int(request.form.get('user_id'))
            
            # Check if user exists by getting their profile
            response = requests.get(f"{API_URL}/profile/{user_id}")
            
            if response.status_code == 200:
                # Store user ID in session
                session['user_id'] = user_id
                session['user_profile'] = response.json()
                
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('User not found. Please try again.', 'error')
        
        except ValueError:
            flash('Invalid user ID. Please enter a number.', 'error')
        except requests.RequestException as e:
            logger.error(f"API error: {e}")
            flash('Error connecting to the server. Please try again later.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page."""
    if request.method == 'POST':
        try:
            # Get form data
            user_id = int(request.form.get('user_id'))
            preferences = {}
            
            # Extract genre preferences
            genres = ['action', 'comedy', 'drama', 'scifi', 'thriller', 'romance', 'horror']
            for genre in genres:
                if request.form.get(f'genre_{genre}'):
                    preferences[genre] = float(request.form.get(f'genre_{genre}_rating', 3))
            
            # Create user profile
            profile_data = {
                "user_id": user_id,
                "preferences": preferences
            }
            
            response = requests.post(
                f"{API_URL}/profile",
                json=profile_data
            )
            
            if response.status_code == 200:
                # Store user ID in session
                session['user_id'] = user_id
                session['user_profile'] = response.json()
                
                flash('Registration successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash(f'Error: {response.json()["detail"]}', 'error')
        
        except ValueError:
            flash('Invalid user ID. Please enter a number.', 'error')
        except requests.RequestException as e:
            logger.error(f"API error: {e}")
            flash('Error connecting to the server. Please try again later.', 'error')
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard with recommendations."""
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get recommendations
        user_id = session['user_id']
        
        response = requests.post(
            f"{API_URL}/recommendations",
            json={
                "user_id": user_id,
                "n": 10,
                "algorithm": "hybrid",
                "include_rated": False,
                "diversity": 0.3
            }
        )
        
        if response.status_code == 200:
            recommendations = response.json()
            return render_template('dashboard.html', recommendations=recommendations)
        else:
            flash(f'Error getting recommendations: {response.json()["detail"]}', 'error')
            return render_template('dashboard.html', recommendations=None)
    
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        flash('Error connecting to the server. Please try again later.', 'error')
        return render_template('dashboard.html', recommendations=None)

@app.route('/recommendations')
def recommendations():
    """View recommendations with different algorithms."""
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get algorithm from query params (default to hybrid)
        algorithm = request.args.get('algorithm', 'hybrid')
        diversity = float(request.args.get('diversity', 0.3))
        include_rated = request.args.get('include_rated', 'false').lower() == 'true'
        
        # Get recommendations
        user_id = session['user_id']
        
        response = requests.post(
            f"{API_URL}/recommendations",
            json={
                "user_id": user_id,
                "n": 10,
                "algorithm": algorithm,
                "include_rated": include_rated,
                "diversity": diversity
            }
        )
        
        if response.status_code == 200:
            recommendations = response.json()
            return render_template(
                'recommendations.html', 
                recommendations=recommendations, 
                algorithm=algorithm,
                diversity=diversity,
                include_rated=include_rated
            )
        else:
            flash(f'Error getting recommendations: {response.json()["detail"]}', 'error')
            return render_template('recommendations.html', recommendations=None)
    
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        flash('Error connecting to the server. Please try again later.', 'error')
        return render_template('recommendations.html', recommendations=None)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    """View movie details."""
    try:
        # Get movie details
        response = requests.get(f"{API_URL}/movies/{movie_id}")
        
        if response.status_code == 200:
            movie = response.json()
            
            # Get similar movies
            similar_response = requests.get(f"{API_URL}/similar/{movie_id}")
            
            if similar_response.status_code == 200:
                similar_movies = similar_response.json()
            else:
                similar_movies = []
            
            return render_template('movie.html', movie=movie, similar_movies=similar_movies)
        else:
            flash(f'Movie not found.', 'error')
            return redirect(url_for('dashboard'))
    
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        flash('Error connecting to the server. Please try again later.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/rate/<int:movie_id>', methods=['GET', 'POST'])
def rate_movie(movie_id):
    """Rate a movie."""
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    try:
        if request.method == 'POST':
            # Submit rating
            rating = float(request.form.get('rating', 0))
            
            if rating < 0.5 or rating > 5:
                flash('Rating must be between 0.5 and 5.', 'error')
                return redirect(url_for('rate_movie', movie_id=movie_id))
            
            # Send rating to API
            response = requests.post(
                f"{API_URL}/rate",
                json={
                    "user_id": session['user_id'],
                    "movie_id": movie_id,
                    "rating": rating
                }
            )
            
            if response.status_code == 200:
                flash('Rating submitted successfully!', 'success')
                return redirect(url_for('movie_details', movie_id=movie_id))
            else:
                flash(f'Error submitting rating: {response.json()["detail"]}', 'error')
        
        # Get movie details
        response = requests.get(f"{API_URL}/movies/{movie_id}")
        
        if response.status_code == 200:
            movie = response.json()
            return render_template('rate.html', movie=movie)
        else:
            flash('Movie not found.', 'error')
            return redirect(url_for('dashboard'))
    
    except ValueError:
        flash('Invalid rating value. Please enter a number.', 'error')
        return redirect(url_for('rate_movie', movie_id=movie_id))
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        flash('Error connecting to the server. Please try again later.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """View and edit user profile."""
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    try:
        if request.method == 'POST':
            # Update preferences
            preferences = {}
            
            # Extract genre preferences
            genres = ['action', 'comedy', 'drama', 'scifi', 'thriller', 'romance', 'horror']
            for genre in genres:
                if request.form.get(f'genre_{genre}'):
                    preferences[genre] = float(request.form.get(f'genre_{genre}_rating', 3))
            
            # Update user profile
            profile_data = {
                "user_id": session['user_id'],
                "preferences": preferences
            }
            
            response = requests.post(
                f"{API_URL}/profile",
                json=profile_data
            )
            
            if response.status_code == 200:
                session['user_profile'] = response.json()
                flash('Profile updated successfully!', 'success')
            else:
                flash(f'Error updating profile: {response.json()["detail"]}', 'error')
        
        # Get user profile
        user_id = session['user_id']
        profile = session.get('user_profile')
        
        if not profile:
            response = requests.get(f"{API_URL}/profile/{user_id}")
            
            if response.status_code == 200:
                profile = response.json()
                session['user_profile'] = profile
        
        return render_template('profile.html', profile=profile)
    
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        flash('Error connecting to the server. Please try again later.', 'error')
        return render_template('profile.html', profile=None)

@app.route('/logout')
def logout():
    """Log out user."""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)