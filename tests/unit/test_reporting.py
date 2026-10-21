import json

from bench import runbench
from bench.reporting import build_summary_markdown, compare_summaries


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
        "query_variant_requested": "long",
        "query_variant": "same",
        "query_variants_distinct": False,
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
    assert "- Query variant: `same` (short and long texts are identical; requested `long`)" in md


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


def test_compare_summaries_rejects_distinct_query_sets() -> None:
    legacy_short = {
        "run_id": "old",
        "dataset_source": "beir/trec-covid",
        "mode": "fulltext",
        "performance": {"fulltext": {"qps": 1.0}},
        "relevance": {"fulltext": {"ndcg@10": 0.1}},
    }
    new_long = {
        "run_id": "new",
        "dataset_source": "beir/trec-covid",
        "mode": "fulltext",
        "query_variant_requested": "long",
        "query_variant": "long",
        "query_variants_distinct": True,
        "performance": {"fulltext": {"qps": 2.0}},
        "relevance": {"fulltext": {"ndcg@10": 0.2}},
    }

    comparison = compare_summaries(new_long, legacy_short)

    assert "not directly comparable" in comparison
    assert "delta" not in comparison


def test_compare_summaries_accepts_same_text_identity() -> None:
    short = {
        "run_id": "old",
        "dataset_source": "beir/scifact/test",
        "mode": "fulltext",
        "performance": {"fulltext": {"qps": 1.0}},
        "relevance": {"fulltext": {"ndcg@10": 0.1}},
    }
    same = {
        "run_id": "new",
        "dataset_source": "beir/scifact/test",
        "mode": "fulltext",
        "query_variant_requested": "long",
        "query_variant": "same",
        "query_variants_distinct": False,
        "performance": {"fulltext": {"qps": 2.0}},
        "relevance": {"fulltext": {"ndcg@10": 0.2}},
    }

    comparison = compare_summaries(same, short)

    assert "not directly comparable" not in comparison
    assert "delta" in comparison
