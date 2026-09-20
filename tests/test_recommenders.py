import pandas as pd

from models.popularity import PopularityRecommender
from models.svd import SVDRecommender
from src.data_loader import enrich_movies


def sample_data():
    ratings = pd.DataFrame({
        "userId": [1,1,1,2,2,2,3,3,3,4,4,4],
        "movieId": [10,11,12,10,11,13,10,12,13,11,12,13],
        "rating":  [5,4,5,5,4,4,4,5,4,5,5,4],
        "timestamp": pd.date_range("2024-01-01", periods=12, freq="D")
    })
    movies = pd.DataFrame({
        "movieId": [10,11,12,13],
        "title": ["Alpha (2020)", "Beta (2021)", "Gamma (2022)", "Delta (2023)"],
        "genres": ["Action|Sci-Fi", "Comedy", "Drama", "Action|Drama"]
    })
    return ratings, enrich_movies(movies)


def test_popularity_returns_unseen():
    ratings, _ = sample_data()
    model = PopularityRecommender(min_ratings=1).fit(ratings)
    recs = model.recommend(1, seen={10,11}, k=2)
    assert len(recs) <= 2
    assert not ({10,11} & set(recs))


def test_svd_recommender():
    ratings, _ = sample_data()
    model = SVDRecommender(n_components=2).fit(ratings)
    recs = model.recommend(1, seen={10,11}, k=2)
    assert all(x not in {10,11} for x in recs)
