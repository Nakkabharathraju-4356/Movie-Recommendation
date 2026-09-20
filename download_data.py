from pathlib import Path
import zipfile
import requests
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "ml-1m.zip"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading MovieLens 1M...")
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)

    extract_dir = DATA_DIR / "ml-1m"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)

    ratings_raw = extract_dir / "ratings.dat"
    movies_raw = extract_dir / "movies.dat"

    ratings = pd.read_csv(
        ratings_raw,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )

    movies = pd.read_csv(
        movies_raw,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    ratings.to_csv(DATA_DIR / "ratings.csv", index=False)
    movies.to_csv(DATA_DIR / "movies.csv", index=False)

    print("Created:")
    print(DATA_DIR / "ratings.csv")
    print(DATA_DIR / "movies.csv")


if __name__ == "__main__":
    main()
