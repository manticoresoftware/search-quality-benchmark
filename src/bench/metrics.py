from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def ndcg_at_k(retrieved_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = []
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        rel = qrels.get(doc_id, 0)
        gains.append((2**rel - 1) / math.log2(i + 2))
    dcg = sum(gains)
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(retrieved_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant_total = sum(1 for rel in qrels.values() if rel > 0)
    if relevant_total == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            hits += 1
            score += hits / i
    return score / relevant_total


def recall_at_k(retrieved_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    found = sum(1 for d in retrieved_doc_ids[:k] if d in relevant)
    return found / len(relevant)


def precision_at_k(retrieved_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {d for d, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    found = sum(1 for d in retrieved_doc_ids[:k] if d in relevant)
    return found / k


def reciprocal_rank_at_k(retrieved_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0
