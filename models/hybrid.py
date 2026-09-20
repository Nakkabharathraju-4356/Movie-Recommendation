import numpy as np


class HybridRecommender:
    def __init__(self, collaborative, content, popularity,
                 collaborative_weight=0.55, content_weight=0.30, popularity_weight=0.15):
        self.collaborative = collaborative
        self.content = content
        self.popularity = popularity
        self.weights = np.array([
            collaborative_weight,
            content_weight,
            popularity_weight
        ], dtype=float)

    def recommend(self, user_id, seen=None, k=10):
        seen = set(seen or [])
        candidate_ids = set()

        collab = self.collaborative.recommend(user_id, seen=seen, k=max(k * 5, 50))
        candidate_ids.update(collab)

        if hasattr(self.collaborative, "user_seen"):
            liked = list(self.collaborative.user_seen.get(user_id, set()))
        else:
            liked = []

        content = self.content.recommend_for_profile(liked, seen=seen, k=max(k * 5, 50))
        candidate_ids.update(content)

        pop = self.popularity.recommend(user_id, seen=seen, k=max(k * 5, 50))
        candidate_ids.update(pop)

        if not candidate_ids:
            return pop[:k]

        def rank_scores(items):
            scores = {}
            for rank, item in enumerate(items):
                scores[item] = 1.0 / np.log2(rank + 2)
            return scores

        c_scores = rank_scores(collab)
        t_scores = rank_scores(content)
        p_scores = rank_scores(pop)

        final = []
        for item in candidate_ids:
            score = (
                self.weights[0] * c_scores.get(item, 0.0)
                + self.weights[1] * t_scores.get(item, 0.0)
                + self.weights[2] * p_scores.get(item, 0.0)
            )
            final.append((item, score))

        final.sort(key=lambda x: x[1], reverse=True)
        return [int(item) for item, _ in final[:k]]
