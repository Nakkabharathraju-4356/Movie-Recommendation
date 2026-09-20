import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNRecommender:
    def __init__(self, n_neighbors=40):
        self.n_neighbors = n_neighbors
        self.movie_ids = None
        self.movie_to_idx = {}
        self.similarity = None
        self.rating_matrix = None
        self.user_seen = {}

    def fit(self, ratings):
        user_codes, user_uniques = np.unique(
            ratings["userId"].to_numpy(),
            return_inverse=True
        )

        movie_codes, movie_uniques = np.unique(
            ratings["movieId"].to_numpy(),
            return_inverse=True
        )

        matrix = csr_matrix(
            (
                ratings["rating"].to_numpy(),
                (user_uniques, movie_uniques)
            ),
            shape=(len(user_codes), len(movie_codes))
        )

        matrix = matrix.tocsr().astype(float)

        # Mean-center ratings for each user
        user_means = np.zeros(matrix.shape[0])

        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)

            if row.nnz:
                user_means[i] = row.data.mean()

        centered = matrix.copy().tolil()

        for i in range(centered.shape[0]):
            if centered.rows[i]:
                centered.data[i] = [
                    value - user_means[i]
                    for value in centered.data[i]
                ]

        centered = centered.tocsr()

        # Item-item cosine similarity
        item_matrix = normalize(centered.T, axis=1)

        self.similarity = cosine_similarity(
            item_matrix,
            dense_output=False
        ).tocsr()

        self.movie_ids = movie_codes

        self.movie_to_idx = {
            int(movie_id): index
            for index, movie_id in enumerate(movie_codes)
        }

        self.user_seen = (
            ratings.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend(self, user_id, seen=None, k=10):
        seen = set(
            seen or self.user_seen.get(user_id, set())
        )

        seen_indices = [
            self.movie_to_idx[movie_id]
            for movie_id in seen
            if movie_id in self.movie_to_idx
        ]

        if not seen_indices:
            return []

        # Aggregate similarity scores efficiently
        scores = np.asarray(
            self.similarity[seen_indices].sum(axis=0)
        ).ravel()

        # Never recommend movies already seen
        for index in seen_indices:
            scores[index] = -np.inf

        candidate_count = min(k, len(scores))

        if candidate_count == 0:
            return []

        # Select only the highest scoring candidates
        top_indices = np.argpartition(
            -scores,
            candidate_count - 1
        )[:candidate_count]

        # Sort selected candidates by score
        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        recommendations = [
            int(self.movie_ids[index])
            for index in top_indices
            if np.isfinite(scores[index])
        ]

        return recommendations[:k]