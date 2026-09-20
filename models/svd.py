import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, n_components=40, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.user_seen = {}

    def fit(self, ratings):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())

        self.user_to_idx = {int(v): i for i, v in enumerate(self.user_ids)}
        self.movie_to_idx = {int(v): i for i, v in enumerate(self.movie_ids)}

        rows = ratings["userId"].map(self.user_to_idx).to_numpy()
        cols = ratings["movieId"].map(self.movie_to_idx).to_numpy()
        self.matrix = csr_matrix(
            (ratings["rating"].to_numpy(), (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids))
        )

        max_components = max(2, min(self.matrix.shape) - 1)
        n_components = min(self.n_components, max_components)

        self.model = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state
        )
        self.model.fit(self.matrix)
        self.user_seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
        return self

    def recommend(self, user_id, seen=None, k=10):
        if user_id not in self.user_to_idx:
            return []

        seen = set(seen or self.user_seen.get(user_id, set()))
        user_idx = self.user_to_idx[int(user_id)]

        user_vector = self.matrix[user_idx]
        user_latent = self.model.transform(user_vector)
        item_latent = self.model.components_
        scores = user_latent @ item_latent

        order = np.argsort(-scores.ravel())
        return [
            int(self.movie_ids[i])
            for i in order
            if int(self.movie_ids[i]) not in seen
        ][:k]
