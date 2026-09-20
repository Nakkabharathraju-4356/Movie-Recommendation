from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

RATINGS_PATH = DATA_DIR / "ratings.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"

RANDOM_STATE = 42
TOP_K = 10
MIN_RATINGS_FOR_POPULARITY = 20
SVD_COMPONENTS = 40
KNN_NEIGHBORS = 40
