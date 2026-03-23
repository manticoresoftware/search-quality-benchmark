from pathlib import Path

from bench.summarize import ArtifactRow, discover_artifacts, render_combined_report


def test_discover_artifacts_extracts_doc_count(tmp_path: Path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        """
{
  "run_id": "abc123",
  "run_at_unix": 1770000000,
  "engine": "manticore",
  "mode": "fulltext",
  "dataset_source": "beir/trec-covid",
  "model": "bm25",
  "relevance": {"fulltext": {"ndcg@10": 0.5}},
  "performance": {"fulltext": {"qps": 10.0}},
  "live_state_lines": ["rows (trec_covid_ft): 171,332"]
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    rows = discover_artifacts()
    assert len(rows) == 1
    assert rows[0].doc_count == 171332
    report = render_combined_report(rows, title="X")
    assert "| Docs |" not in report
    assert "manticore / full-text" in report
    assert "manticore / full-text / 171332" not in report


def test_render_combined_report_groups_by_dataset() -> None:
    rows = [
        ArtifactRow(
            run_id="a1",
            profile="p1",
            run_at_unix=1,
            run_at="2026-01-01 00:00:00",
            path=Path("artifacts/a1"),
            engine="manticore",
            mode="fulltext",
            model="bm25",
            dataset_source="beir/nfcorpus/test",
            doc_count=100,
            runtime_seconds=0.1,
            concurrency=1,
            relevance={"ndcg@10": 0.5},
            performance={"qps": 10.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="b1",
            profile="p2",
            run_at_unix=2,
            run_at="2026-01-02 00:00:00",
            path=Path("artifacts/b1"),
            engine="qdrant",
            mode="fulltext",
            model="Qdrant/bm25",
            dataset_source="beir/trec-covid",
            doc_count=200,
            runtime_seconds=0.2,
            concurrency=1,
            relevance={"ndcg@10": 0.4},
            performance={"qps": 9.0},
            status="ok",
        ),
    ]
    report = render_combined_report(rows, title="X")
    assert "## Relevance for [`beir/nfcorpus/test`](https://ir-datasets.com/beir.html#beir/nfcorpus/test)" in report
    assert "## Relevance for [`beir/trec-covid`](https://ir-datasets.com/beir.html#beir/trec-covid)" in report
    assert "## Query Performance for [`beir/nfcorpus/test`](https://ir-datasets.com/beir.html#beir/nfcorpus/test)" in report
    assert "## Query Performance for [`beir/trec-covid`](https://ir-datasets.com/beir.html#beir/trec-covid)" in report


def test_render_combined_report_includes_ndcg_chart_best_per_engine_mode_docs() -> None:
    rows = [
        ArtifactRow(
            run_id="a1",
            profile="p1",
            run_at_unix=1,
            run_at="2026-01-01 00:00:00",
            path=Path("artifacts/a1"),
            engine="manticore",
            mode="fulltext",
            model="bm25",
            dataset_source="msmarco-passage/dev/small",
            doc_count=250000,
            runtime_seconds=0.1,
            concurrency=1,
            relevance={"ndcg@10": 0.30, "mrr@10": 0.20},
            performance={"qps": 10.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="a2",
            profile="p2",
            run_at_unix=2,
            run_at="2026-01-02 00:00:00",
            path=Path("artifacts/a2"),
            engine="manticore",
            mode="fulltext",
            model="bm25f",
            dataset_source="msmarco-passage/dev/small",
            doc_count=250000,
            runtime_seconds=0.2,
            concurrency=1,
            relevance={"ndcg@10": 0.40, "mrr@10": 0.25},
            performance={"qps": 9.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="b1",
            profile="p3",
            run_at_unix=3,
            run_at="2026-01-03 00:00:00",
            path=Path("artifacts/b1"),
            engine="qdrant",
            mode="vector",
            model="openai",
            dataset_source="msmarco-passage/dev/small",
            doc_count=8841823,
            runtime_seconds=0.3,
            concurrency=1,
            relevance={"ndcg@10": 0.55, "mrr@10": 0.50},
            performance={"qps": 8.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="c1",
            profile="p4",
            run_at_unix=4,
            run_at="2026-01-04 00:00:00",
            path=Path("artifacts/c1"),
            engine="meilisearch",
            mode="hybrid",
            model="openai",
            dataset_source="msmarco-passage/dev/small",
            doc_count=250000,
            runtime_seconds=0.4,
            concurrency=1,
            relevance={"ndcg@10": 0.80, "mrr@10": 0.70},
            performance={"qps": 7.0},
            status="ok",
        ),
    ]
    report = render_combined_report(rows, title="X")
    assert "## Relevance for [`msmarco-passage/dev/small`](https://ir-datasets.com/msmarco-passage.html#msmarco-passage/dev/small)" in report
    assert "```mermaid" not in report
    assert "<table>" not in report
    assert "Best NDCG@10:" in report
    assert "Best MRR@10:" in report
    assert "| Category | NDCG@10 |" in report
    assert "| Category | MRR@10 |" in report
    assert "qdrant / vector / 8841823" in report
    assert "manticore / full-text / 250000" in report
    assert report.index("qdrant / vector / 8841823") < report.index("manticore / full-text / 250000")
    assert "0.4000" in report
    assert "0.5500" in report
    assert "| meilisearch / hybrid / 250000 |" not in report


def test_render_combined_report_hides_docs_when_constant_but_keeps_when_mixed() -> None:
    same_docs_rows = [
        ArtifactRow(
            run_id="a1",
            profile="p1",
            run_at_unix=1,
            run_at="2026-01-01 00:00:00",
            path=Path("artifacts/a1"),
            engine="manticore",
            mode="fulltext",
            model="bm25",
            dataset_source="beir/nfcorpus/test",
            doc_count=100,
            runtime_seconds=0.1,
            concurrency=1,
            relevance={"ndcg@10": 0.5},
            performance={"qps": 10.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="a2",
            profile="p2",
            run_at_unix=2,
            run_at="2026-01-02 00:00:00",
            path=Path("artifacts/a2"),
            engine="qdrant",
            mode="fulltext",
            model="bm25",
            dataset_source="beir/nfcorpus/test",
            doc_count=100,
            runtime_seconds=0.1,
            concurrency=1,
            relevance={"ndcg@10": 0.4},
            performance={"qps": 9.0},
            status="ok",
        ),
    ]
    mixed_docs_rows = same_docs_rows + [
        ArtifactRow(
            run_id="b1",
            profile="p3",
            run_at_unix=3,
            run_at="2026-01-03 00:00:00",
            path=Path("artifacts/b1"),
            engine="qdrant",
            mode="vector",
            model="openai",
            dataset_source="beir/trec-covid",
            doc_count=200,
            runtime_seconds=0.2,
            concurrency=1,
            relevance={"ndcg@10": 0.3},
            performance={"qps": 8.0},
            status="ok",
        ),
        ArtifactRow(
            run_id="b2",
            profile="p4",
            run_at_unix=4,
            run_at="2026-01-04 00:00:00",
            path=Path("artifacts/b2"),
            engine="manticore",
            mode="vector",
            model="openai",
            dataset_source="beir/trec-covid",
            doc_count=300,
            runtime_seconds=0.2,
            concurrency=1,
            relevance={"ndcg@10": 0.2},
            performance={"qps": 7.0},
            status="ok",
        ),
    ]
    report = render_combined_report(mixed_docs_rows, title="X")
    nf_rel = "## Relevance for [`beir/nfcorpus/test`](https://ir-datasets.com/beir.html#beir/nfcorpus/test)"
    nf_perf = "## Query Performance for [`beir/nfcorpus/test`](https://ir-datasets.com/beir.html#beir/nfcorpus/test)"
    tc_rel = "## Relevance for [`beir/trec-covid`](https://ir-datasets.com/beir.html#beir/trec-covid)"
    tc_perf = "## Query Performance for [`beir/trec-covid`](https://ir-datasets.com/beir.html#beir/trec-covid)"
    nf_block = report.split(nf_rel, 1)[1].split(nf_perf, 1)[0]
    tc_block = report.split(tc_rel, 1)[1].split(tc_perf, 1)[0]
    assert "| Docs |" not in nf_block
    assert "| Docs |" in tc_block
    assert "/ 100 |" not in nf_block
    assert "/ 200 |" in tc_block
