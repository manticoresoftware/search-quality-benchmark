from pathlib import Path

from bench.config import load_engine_targets
from bench.run_matrix import build_variants_from_config_paths


def test_load_engine_targets_from_inheritance_entries(tmp_path: Path):
    cfg = tmp_path / "config.multi.yaml"
    cfg.write_text(
        """
dataset:
  source: "beir/trec-covid"
manticore_default:
  type: "manticore"
  search_options: "ranker=expr('1000*bm25a(1.2,0.75)')"
manticore_bm25:
  base: "manticore_default"
  search_options: "ranker=bm25"
qdrant_main:
  type: "qdrant"
  qdrant:
    host: "127.0.0.1"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    targets = load_engine_targets(cfg)
    assert len(targets) == 3
    by_name = {t.name: t.type for t in targets}
    assert by_name == {
        "manticore_default": "manticore",
        "manticore_bm25": "manticore",
        "qdrant_main": "qdrant",
    }


def test_run_variants_include_each_engine_entry(tmp_path: Path):
    cfg = tmp_path / "config.multi.yaml"
    cfg.write_text(
        """
benchmark:
  concurrency: 2
mant_a:
  type: "manticore"
  search_options: "ranker=bm25"
mant_b:
  base: "mant_a"
  search_options: "ranker=proximity_bm25"
qdr:
  type: "qdrant"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_variants_from_config_paths([cfg])
    # 3 entries * 2 modes each
    assert len(variants) == 6
    assert {v.entry for v in variants} == {"mant_a", "mant_b", "qdr"}
    assert all(v.default_concurrency == 2 for v in variants)
