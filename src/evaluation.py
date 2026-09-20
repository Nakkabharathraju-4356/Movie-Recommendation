import math
import numpy as np
import pandas as pd


def precision_at_k(recommended, relevant, k):
    recs = recommended[:k]
    if k == 0:
        return 0.0
    return len(set(recs) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def average_precision_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i

    return score / min(len(relevant), k)


def catalog_coverage(all_recommendations, catalog_size):
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / catalog_size if catalog_size else 0.0


def intra_list_diversity(recommendations, similarity_matrix=None):
    # Generic diversity proxy: 1 - duplicate ratio when no feature matrix is supplied.
    if not recommendations:
        return 0.0
    n = len(recommendations)
    unique = len(set(recommendations))
    return 1.0 - (n - unique) / max(1, n)


def evaluate_model(model, train, test, k=10, max_users=1000):
    relevant_by_user = test.groupby("userId")["movieId"].apply(set).to_dict()
    seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()

    users = list(relevant_by_user.keys())[:max_users]

    rows = []
    all_recs = []

    for user_id in users:
        recs = model.recommend(
            user_id,
            seen=seen_by_user.get(user_id, set()),
            k=k
        )
        relevant = relevant_by_user[user_id]

        rows.append({
            "precision_at_k": precision_at_k(recs, relevant, k),
            "recall_at_k": recall_at_k(recs, relevant, k),
            "ndcg_at_k": ndcg_at_k(recs, relevant, k),
            "map_at_k": average_precision_at_k(recs, relevant, k),
        })
        all_recs.append(recs)

    metrics = pd.DataFrame(rows).mean().to_dict() if rows else {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "map_at_k": 0.0,
    }

    metrics["catalog_coverage"] = catalog_coverage(all_recs, train["movieId"].nunique())
    metrics["users_evaluated"] = len(users)

    return metrics
