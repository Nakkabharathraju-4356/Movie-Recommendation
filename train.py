import json
from pathlib import Path
import joblib

from src.config import RATINGS_PATH, MOVIES_PATH, ARTIFACT_DIR
from src.data_loader import load_data, time_based_split, enrich_movies
from src.pipeline import build_models


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)

    ratings, movies = load_data(RATINGS_PATH, MOVIES_PATH)
    movies = enrich_movies(movies)
    train, test = time_based_split(ratings)

    print(f"Total interactions: {len(ratings):,}")
    print(f"Training interactions: {len(train):,}")
    print(f"Test interactions: {len(test):,}")
    print(f"Users: {ratings.userId.nunique():,}")
    print(f"Movies: {ratings.movieId.nunique():,}")

    models = build_models(train, movies)

    for name, model in models.items():
        safe = name.lower().replace("-", "_").replace(" ", "_")
        joblib.dump(model, ARTIFACT_DIR / f"{safe}.joblib")
        print(f"Saved {name}")

    metadata = {
        "train_rows": len(train),
        "test_rows": len(test),
        "users": int(ratings.userId.nunique()),
        "movies": int(ratings.movieId.nunique()),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    train.to_csv(ARTIFACT_DIR / "train_split.csv", index=False)
    test.to_csv(ARTIFACT_DIR / "test_split.csv", index=False)

    print("Training complete.")


if __name__ == "__main__":
    main()
