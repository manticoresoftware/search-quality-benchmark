from pathlib import Path

from bench.run_matrix import build_variants_from_config_paths, filter_runnable_variants


def test_filter_runnable_variants(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "config.test.yaml"
    cfg.write_text(
        """
manticore:
  host: "127.0.0.1"
  sql_port: 9306
  http_port: 9308
qdrant:
  host: "127.0.0.1"
  port: 6333
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_variants_from_config_paths([cfg])
    assert variants

    # Keep only manticore fulltext as runnable in this synthetic test.
    monkeypatch.setattr(
        "bench.run_matrix.is_variant_runnable",
        lambda v: v.engine == "manticore" and v.mode == "fulltext",
    )
    filtered = filter_runnable_variants(variants)
    assert len(filtered) == 3
    assert {v.engine for v in filtered} == {"manticore"}
    assert {v.mode for v in filtered} == {"fulltext"}
    assert {v.benchmark_dataset_id for v in filtered} == {
        "trec-covid-query",
        "trec-covid-question",
        "trec-covid-narrative",
    }
