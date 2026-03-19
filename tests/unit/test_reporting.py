from bench.reporting import build_summary_markdown


def test_build_summary_includes_resolved_config_and_live_state() -> None:
    metrics = {
        "run_id": "abc123",
        "profile": "manticore_bm25",
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

