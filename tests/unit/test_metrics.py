from bench.metrics import average_precision_at_k, ndcg_at_k, recall_at_k


def test_metrics_basic():
    qrels = {"d1": 2, "d2": 1, "d3": 0}
    retrieved = ["d1", "d4", "d2", "d3"]
    assert ndcg_at_k(retrieved, qrels, 3) > 0
    assert 0 <= average_precision_at_k(retrieved, qrels, 3) <= 1
    assert recall_at_k(retrieved, qrels, 3) == 1.0
