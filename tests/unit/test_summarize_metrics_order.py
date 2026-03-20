from pathlib import Path

from bench.summarize import ArtifactRow, _collect_relevance_keys


def test_collect_relevance_keys_includes_mrr_order():
    row = ArtifactRow(
        run_id="r1",
        run_at_unix=1,
        run_at="2026-01-01 00:00:00",
        path=Path("artifacts/r1"),
        engine="manticore",
        mode="fulltext",
        model="x",
        dataset_source="msmarco-passage/dev/small",
        doc_count=100,
        runtime_seconds=0.1,
        concurrency=1,
        relevance={
            "ndcg@10": 0.1,
            "ndcg@20": 0.1,
            "map@10": 0.1,
            "mrr@10": 0.1,
            "p@20": 0.1,
            "recall@100": 0.1,
        },
        performance={},
        status="ok",
    )
    keys = _collect_relevance_keys([row])
    assert keys[:6] == ["ndcg@10", "ndcg@20", "map@10", "mrr@10", "p@20", "recall@100"]
