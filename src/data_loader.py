from pathlib import Path
import pandas as pd


def load_data(ratings_path, movies_path):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}

    missing_r = required_ratings - set(ratings.columns)
    missing_m = required_movies - set(movies.columns)

    if missing_r:
        raise ValueError(f"ratings.csv is missing columns: {missing_r}")
    if missing_m:
        raise ValueError(f"movies.csv is missing columns: {missing_m}")

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
    ratings = ratings.dropna(subset=["timestamp"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    movies["movieId"] = movies["movieId"].astype(int)
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["title"] = movies["title"].fillna("Unknown")

    return ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True), movies


def time_based_split(ratings, test_fraction=0.2, min_interactions=5):
    train_parts = []
    test_parts = []

    for _, group in ratings.groupby("userId", sort=False):
        group = group.sort_values("timestamp")
        if len(group) < min_interactions:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(len(group) * test_fraction)))
        n_test = min(n_test, len(group) - 1)
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()

    return train, test


def enrich_movies(movies):
    result = movies.copy()
    result["genre_tokens"] = result["genres"].str.replace("|", " ", regex=False)
    result["search_text"] = result["title"].str.replace(r"[\(\),:]", " ", regex=True) + " " + result["genre_tokens"]
    return result
