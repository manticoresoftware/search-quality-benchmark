from pathlib import Path

from bench.run_matrix import build_variants_from_config_paths


def test_build_variants_from_config_paths(tmp_path: Path):
    cfg = tmp_path / "config.test.yaml"
    cfg.write_text(
        """
benchmark:
  concurrency: 3
manticore:
  type: "manticore"
  host: "127.0.0.1"
  sql_port: 9306
  http_port: 9308
qdrant:
  type: "qdrant"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_variants_from_config_paths([cfg])
    # Four engine/mode rows, each expanded into the three logical TREC-COVID datasets.
    assert len(variants) == 12
    assert all(v.config_path == str(cfg) for v in variants)
    assert all(v.entry in {"manticore", "qdrant"} for v in variants)
    assert {v.engine for v in variants} == {"manticore", "qdrant"}
    assert {v.mode for v in variants} == {"fulltext", "vector"}
    assert {v.benchmark_dataset_id for v in variants} == {
        "trec-covid-query",
        "trec-covid-question",
        "trec-covid-narrative",
    }
    assert {v.query_field for v in variants} == {"query", "question", "narrative"}
    assert all(v.default_concurrency == 3 for v in variants)
    assert all(v.model != "" for v in variants)
    ft_by_engine = {v.engine: v.model for v in variants if v.mode == "fulltext"}
    assert "profile:relevance" in ft_by_engine["manticore"]
    assert ft_by_engine["qdrant"] == "Qdrant/bm25"
