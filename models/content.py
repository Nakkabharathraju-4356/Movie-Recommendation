import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=15000
        )
        self.movie_ids = None
        self.matrix = None
        self.movie_to_idx = {}
        self.movies = None

    def fit(self, movies):
        self.movies = movies.reset_index(drop=True).copy()
        self.movie_ids = self.movies["movieId"].to_numpy()
        self.movie_to_idx = {int(mid): i for i, mid in enumerate(self.movie_ids)}
        self.matrix = self.vectorizer.fit_transform(self.movies["search_text"])
        return self

    def recommend_similar_to_movie(self, movie_id, k=10, exclude=None):
        exclude = set(exclude or [])
        idx = self.movie_to_idx.get(int(movie_id))
        if idx is None:
            return []

        scores = linear_kernel(self.matrix[idx], self.matrix).ravel()
        order = np.argsort(-scores)
        return [
            int(self.movie_ids[i])
            for i in order
            if int(self.movie_ids[i]) not in exclude and int(self.movie_ids[i]) != int(movie_id)
        ][:k]

    def recommend_for_profile(self, liked_movie_ids, seen=None, k=10):
        seen = set(seen or [])

        indices = [
            self.movie_to_idx[movie_id]
            for movie_id in liked_movie_ids
            if movie_id in self.movie_to_idx
        ]

        if not indices:
            return []

        # Build the user's content profile
        profile = self.matrix[indices].mean(axis=0)

        # Convert numpy matrix to a standard 2D NumPy array
        profile = np.asarray(profile).reshape(1, -1)

        # Calculate similarity between the user profile and all movies
        scores = linear_kernel(
            profile,
            self.matrix
        ).ravel()

        # Exclude movies already seen
        for movie_id in seen:
            idx = self.movie_to_idx.get(movie_id)

            if idx is not None:
                scores[idx] = -np.inf

        candidate_count = min(k, len(scores))

        if candidate_count == 0:
            return []

        top_indices = np.argpartition(
            -scores,
            candidate_count - 1
        )[:candidate_count]

        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        return [
            int(self.movie_ids[idx])
            for idx in top_indices
            if np.isfinite(scores[idx])
        ][:k]