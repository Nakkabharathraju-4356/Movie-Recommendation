import joblib
import pandas as pd

from src.config import ARTIFACT_DIR, RATINGS_PATH, MOVIES_PATH, TOP_K
from src.data_loader import load_data, time_based_split
from src.evaluation import evaluate_model


def main():
    ratings, movies = load_data(RATINGS_PATH, MOVIES_PATH)

    train_path = ARTIFACT_DIR / "train_split.csv"
    test_path = ARTIFACT_DIR / "test_split.csv"

    if train_path.exists() and test_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    else:
        train, test = time_based_split(ratings)

    names = ["Popularity", "Item-kNN", "SVD", "Hybrid"]
    results = []

    for name in names:
        safe = name.lower().replace("-", "_").replace(" ", "_")
        path = ARTIFACT_DIR / f"{safe}.joblib"
        model = joblib.load(path)
        metrics = evaluate_model(model, train, test, k=TOP_K, max_users=300) 
        metrics["model"] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    columns = [
        "model", "precision_at_k", "recall_at_k",
        "ndcg_at_k", "map_at_k", "catalog_coverage", "users_evaluated"
    ]
    df = df[columns]

    out = ARTIFACT_DIR / "benchmark_results.csv"
    df.to_csv(out, index=False)

    print("\nBenchmark Results")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
