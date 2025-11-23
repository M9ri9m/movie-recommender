import streamlit as st
import pandas as pd
import numpy as np
import pickle


# --------------------------
# 1) تحميل الموديل والبيانات
# --------------------------

@st.cache_resource
def load_model():
    """Load the trained Surprise model from model.pkl."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_data():
    """Load ratings and movies data."""
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies


model = load_model()
ratings, movies = load_data()


# --------------------------
# 2) دالة التوصية للمستخدم
# --------------------------

def recommend_for_user(user_id, n_recs=5):
    """
    Recommend top-N movies for a given user_id
    using the trained Surprise model.
    """
   
    all_movie_ids = movies["movieId"].unique()

    
    watched_movie_ids = ratings.loc[
        ratings["userId"] == user_id, "movieId"
    ].unique()

   
    candidate_movie_ids = [mid for mid in all_movie_ids if mid not in watched_movie_ids]

    preds = []
    for mid in candidate_movie_ids:
       
        est = model.predict(user_id, mid).est
        preds.append((mid, est))

 
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

    top = preds_sorted[:n_recs]

    
    rows = []
    for mid, score in top:
        title = movies.loc[movies["movieId"] == mid, "title"].iloc[0]
        rows.append(
            {
                "movieId": mid,
                "title": title,
                "predicted_rating": round(score, 2),
            }
        )

    return pd.DataFrame(rows)


# --------------------------
# 3) واجهة Streamlit
# --------------------------

st.title("Movie Recommendation System (SVD)")

st.write("اختر User ID، وهنديك أفضل أفلام متوقّعة تناسب تفضيلاته.")

user_ids = sorted(ratings["userId"].unique())

selected_user = st.selectbox("اختر User ID:", user_ids)

n_recs = st.slider("عدد التوصيات:", min_value=3, max_value=20, value=5, step=1)

if st.button("Get Recommendations"):
    recs_df = recommend_for_user(selected_user, n_recs=n_recs)

    if recs_df.empty:
        st.warning("المستخدم ده ملوش بيانات كفاية (أو مفيش أفلام مرشّحة).")
    else:
        st.subheader(f"Top {n_recs} توصيات للمستخدم {selected_user}:")
        for _, row in recs_df.iterrows():
            st.write(f"🎥 **{row['title']}** — Predicted rating: {row['predicted_rating']}")
