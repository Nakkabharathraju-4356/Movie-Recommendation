import numpy as np
import pandas as pd


class PopularityRecommender:
    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking = None
        self.global_mean = None

    def fit(self, ratings):
        self.global_mean = float(ratings["rating"].mean())

        stats = ratings.groupby("movieId").agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )

        stats = stats[stats["rating_count"] >= self.min_ratings]
        if stats.empty:
            stats = ratings.groupby("movieId").agg(
                mean_rating=("rating", "mean"),
                rating_count=("rating", "count")
            )

        m = stats["rating_count"].quantile(0.80)
        stats["score"] = (
            stats["rating_count"] / (stats["rating_count"] + m) * stats["mean_rating"]
            + m / (stats["rating_count"] + m) * self.global_mean
        )

        self.ranking = stats.sort_values(
            ["score", "rating_count"], ascending=[False, False]
        ).index.to_numpy()
        return self

    def recommend(self, user_id=None, seen=None, k=10):
        seen = set(seen or [])
        result = [int(mid) for mid in self.ranking if int(mid) not in seen]
        return result[:k]
