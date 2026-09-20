from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
)


def test_precision():
    assert precision_at_k([1, 2, 3], {1, 4}, 3) == 1 / 3


def test_recall():
    assert recall_at_k([1, 2, 3], {1, 4}, 3) == 0.5


def test_ndcg_is_bounded():
    value = ndcg_at_k([1, 2, 3], {1, 3}, 3)
    assert 0 <= value <= 1


def test_map_is_bounded():
    value = average_precision_at_k([1, 2, 3], {1, 3}, 3)
    assert 0 <= value <= 1
