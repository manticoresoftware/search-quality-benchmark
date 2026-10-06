import json

from bench import runbench
from bench.reporting import build_summary_markdown


def test_build_summary_includes_resolved_config_and_live_state() -> None:
    metrics = {
        "run_id": "abc123",
        "profile": "manticore_bm25",
        "engine": "columnar",
        "engine_version": "29.9.0",
        "engine_image": "manticoresearch/manticore:29.9.0",
        "engine_image_digest": "manticoresearch/manticore@sha256:example",
        "engine_image_created_at": "2026-09-11T14:19:52Z",
        "dataset_source": "msmarco-passage/dev/small",
        "mode": "fulltext",
        "runtime_seconds": 0.12,
        "benchmark_settings": {
            "concurrency": 1,
            "limit": 100,
            "warmup_queries": 5,
            "repeats": 1,
        },
        "resolved_config_lines": [
            "profile: manticore_bm25",
            "engine: manticore",
            "manticore.search_options: ranker=bm25",
        ],
        "live_state_lines": [
            "rows (trec_covid_ft): 171,332",
        ],
        "relevance": {"fulltext": {"ndcg@10": 0.5}},
        "performance": {"fulltext": {"qps": 100.0, "queries_total": 50.0}},
    }
    md = build_summary_markdown(metrics)
    assert "## Resolved Config" in md
    assert "`manticore.search_options: ranker=bm25`" in md
    assert "## Live State" in md
    assert "`rows (trec_covid_ft): 171,332`" in md
    assert "- Engine: `columnar`" in md
    assert "- Engine version: `29.9.0`" in md
    assert "- Engine image: `manticoresearch/manticore:29.9.0`" in md
    assert "- Engine image digest: `manticoresearch/manticore@sha256:example`" in md
    assert "- Engine image created: `2026-09-11T14:19:52Z`" in md
    assert "- Dataset source: `msmarco-passage/dev/small`" in md


def test_write_run_artifacts_records_engine_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runbench,
        "resolve_engine_artifact_metadata",
        lambda engine: {
            "engine_version": "29.9.0",
            "engine_image": "manticoresearch/manticore:29.9.0",
            "engine_image_id": "sha256:image-id",
            "engine_image_digest": "manticoresearch/manticore@sha256:digest",
            "engine_image_created_at": "2026-09-11T14:19:52Z",
            "engine_version_source": "container",
        },
    )
    metrics = {
        "run_id": "abc123",
        "engine": "columnar",
        "mode": "vector",
        "runtime_seconds": 1.0,
        "benchmark_settings": {},
        "relevance": {},
        "performance": {},
    }

    out_dir = runbench.write_run_artifacts("abc123", metrics, [])
    written = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    assert written["engine_version"] == "29.9.0"
    assert written["engine_image"] == "manticoresearch/manticore:29.9.0"
    assert written["engine_image_id"] == "sha256:image-id"
    assert written["engine_image_digest"] == "manticoresearch/manticore@sha256:digest"
    assert written["engine_image_created_at"] == "2026-09-11T14:19:52Z"
    assert written["engine_version_source"] == "container"
