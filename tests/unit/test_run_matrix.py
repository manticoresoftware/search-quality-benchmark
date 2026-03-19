from pathlib import Path

from bench.run_matrix import build_variants_from_config_paths


def test_build_variants_from_config_paths(tmp_path: Path):
    cfg = tmp_path / "config.test.yaml"
    cfg.write_text(
        """
manticore:
  host: "127.0.0.1"
  sql_port: 9306
  http_port: 9308
benchmark:
  concurrency: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_variants_from_config_paths([cfg])
    # (manticore:2 modes + qdrant:2 modes) = 4 variants
    assert len(variants) == 4
    assert all(v.config_path == str(cfg) for v in variants)
    assert all(v.entry in {"manticore", "qdrant"} for v in variants)
    assert {v.engine for v in variants} == {"manticore", "qdrant"}
    assert {v.mode for v in variants} == {"fulltext", "vector"}
    assert all(v.default_concurrency == 3 for v in variants)
    assert all(v.model != "" for v in variants)
    ft_by_engine = {v.engine: v.model for v in variants if v.mode == "fulltext"}
    assert "ranker=expr(" in ft_by_engine["manticore"]
    assert ft_by_engine["qdrant"] == "Qdrant/bm25"
