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
    assert "| Docs |" in report
    assert "171,332" in report


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
    assert "## Relevance for `beir/nfcorpus/test`" in report
    assert "## Relevance for `beir/trec-covid`" in report
    assert "## Query Performance for `beir/nfcorpus/test`" in report
    assert "## Query Performance for `beir/trec-covid`" in report
