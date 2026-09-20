from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACT_DIR, RATINGS_PATH, MOVIES_PATH
from src.data_loader import load_data


st.set_page_config(
    page_title="CineMatch — Recommendation Intelligence",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_base_data():
    ratings, movies = load_data(RATINGS_PATH, MOVIES_PATH)
    return ratings, movies


@st.cache_resource
def load_artifacts():
    models = {}
    for name in ["popularity", "item_knn", "svd", "hybrid"]:
        path = ARTIFACT_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


ratings, movies = load_base_data()
models = load_artifacts()

st.title("🎬 CineMatch")
st.caption("Recommendation Intelligence Lab • Collaborative + Content + Popularity")

if not models:
    st.error("Models are not trained yet. Run `python download_data.py` and then `python train.py`.")
    st.stop()

movie_map = movies.set_index("movieId")["title"].to_dict()
genre_map = movies.set_index("movieId")["genres"].to_dict()

with st.sidebar:
    st.header("Recommendation Settings")

    user_ids = sorted(ratings["userId"].unique())
    selected_user = st.selectbox(
        "Choose a MovieLens user",
        user_ids,
        index=0
    )

    algorithm = st.selectbox(
        "Algorithm",
        ["Hybrid", "SVD", "Item-kNN", "Popularity"]
    )

    k = st.slider("Recommendations", 5, 20, 10)

    st.divider()
    st.write("**Evaluation-aware design**")
    st.write("• Time-based split")
    st.write("• Ranking metrics")
    st.write("• Cold-start fallback")
    st.write("• Explainable recommendations")

model_key = algorithm.lower().replace("-", "_").replace(" ", "_")
model = models[model_key]

seen = set(ratings.loc[ratings.userId == selected_user, "movieId"])

st.subheader(f"Recommendations for User {selected_user}")

if selected_user not in ratings.userId.values:
    recs = models["popularity"].recommend(k=k)
    explanation = "New-user fallback: popularity-based recommendations."
else:
    recs = model.recommend(selected_user, seen=seen, k=k)
    explanation = "Generated from the selected recommendation model and filtered to exclude previously rated movies."

if not recs:
    st.warning("No recommendations available for this user/model combination.")
else:
    cols = st.columns(2)

    for i, movie_id in enumerate(recs):
        title = movie_map.get(movie_id, "Unknown movie")
        genres = genre_map.get(movie_id, "(unknown)")
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {i + 1}. {title}")
                st.caption(genres)

                if algorithm == "Hybrid":
                    st.write("Why this appears:")
                    st.write("• Combines collaborative taste signals")
                    st.write("• Uses movie-content similarity")
                    st.write("• Includes a popularity prior")
                elif algorithm == "SVD":
                    st.write("Why this appears:")
                    st.write("• Strong latent-factor similarity to the user's learned preference space")
                elif algorithm == "Item-kNN":
                    st.write("Why this appears:")
                    st.write("• Similar users' item-interaction patterns")
                else:
                    st.write("Why this appears:")
                    st.write("• Strong historical rating/popularity signal")

st.info(explanation)

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{ratings.userId.nunique():,}")
c2.metric("Movies", f"{movies.movieId.nunique():,}")
c3.metric("Ratings", f"{len(ratings):,}")
c4.metric("Avg. Rating", f"{ratings.rating.mean():.2f}")

st.subheader("Your Recent Rating History")

history = (
    ratings[ratings.userId == selected_user]
    .sort_values("timestamp", ascending=False)
    .head(10)
    .copy()
)
history["Movie"] = history["movieId"].map(movie_map)
history["Genres"] = history["movieId"].map(genre_map)

st.dataframe(
    history[["Movie", "Genres", "rating", "timestamp"]].rename(
        columns={"rating": "Rating", "timestamp": "Rated At"}
    ),
    use_container_width=True,
    hide_index=True
)

benchmark_path = ARTIFACT_DIR / "benchmark_results.csv"
if benchmark_path.exists():
    st.subheader("Model Benchmark")
    benchmark = pd.read_csv(benchmark_path)
    st.dataframe(
        benchmark.style.format({
            "precision_at_k": "{:.3f}",
            "recall_at_k": "{:.3f}",
            "ndcg_at_k": "{:.3f}",
            "map_at_k": "{:.3f}",
            "catalog_coverage": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True
    )
