import numpy as np
import pandas as pd


def movie_lookup(movies):
    return movies.set_index("movieId").to_dict("index")


def top_rated_movies(ratings, movies, n=10, min_ratings=20):
    stats = ratings.groupby("movieId").agg(
        mean_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    stats = stats[stats["rating_count"] >= min_ratings]

    if stats.empty:
        stats = ratings.groupby("movieId").agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )

    C = ratings["rating"].mean()
    m = stats["rating_count"].quantile(0.80)

    stats["score"] = (
        (stats["rating_count"] / (stats["rating_count"] + m)) * stats["mean_rating"]
        + (m / (stats["rating_count"] + m)) * C
    )

    return (
        stats.sort_values("score", ascending=False)
        .head(n)
        .reset_index()
        .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    )


def popularity_distribution(recommendations, ratings):
    counts = ratings.groupby("movieId").size().rename("interaction_count")
    result = recommendations.merge(counts, on="movieId", how="left")
    return result["interaction_count"].fillna(0).to_numpy()
