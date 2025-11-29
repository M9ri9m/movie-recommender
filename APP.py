import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =========================
# 1) Load model & data
# =========================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent  # folder where APP.py lives
    ratings_path = base_dir / "ratings.csv"
    movies_path  = base_dir / "movies.csv"
    
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    return ratings, movies

@st.cache_resource
def load_model():
    base_dir = Path(__file__).parent
    model_path = base_dir / "model.pkl"
    with open(model_path, "rb") as f:
        algo = pickle.load(f)
    return algo

ratings, movies = load_data()
algo = load_model()


# =========================
# 2) Helper: recommendation function
# =========================
def recommend_for_user(user_id, n_recs=10):
    # Movies already rated by this user
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    # Candidate movies (not rated yet)
    candidate_movies = movies[~movies['movieId'].isin(rated_movies)]

    preds = []
    for mid in candidate_movies['movieId'].values:
        est = algo.predict(user_id, mid).est
        preds.append((mid, est))

    # Build dataframe and attach movie metadata
    preds_df = pd.DataFrame(preds, columns=['movieId', 'pred_rating'])
    preds_df = preds_df.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')

    # Sort by predicted rating
    preds_df = preds_df.sort_values('pred_rating', ascending=False).head(n_recs)
    return preds_df

# =========================
# 3) Streamlit UI
# =========================
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("A movie recommendation system based on SVD using the MovieLens dataset.")

st.sidebar.header("Settings")

# Get user IDs from data
all_users = sorted(ratings['userId'].unique())
default_user = int(all_users[0])

user_id = st.sidebar.selectbox("Select User ID", all_users, index=0)
n_recs = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)

st.write(f"### Recommendations for user: {user_id}")

if st.button("Show recommendations"):
    recs = recommend_for_user(user_id, n_recs=n_recs)
    
    if recs.empty:
        st.warning("This user has already rated all movies or no suitable movies were found.")
    else:
        st.subheader("Top Recommendations")
        st.dataframe(
            recs[['title', 'pred_rating', 'genres']].rename(
                columns={
                    'title': 'Movie Title',
                    'pred_rating': 'Predicted Rating',
                    'genres': 'Genres'
                }
            ),
            use_container_width=True
        )

