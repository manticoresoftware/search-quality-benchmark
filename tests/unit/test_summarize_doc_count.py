from pathlib import Path

from bench.summarize import discover_artifacts, render_combined_report


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
