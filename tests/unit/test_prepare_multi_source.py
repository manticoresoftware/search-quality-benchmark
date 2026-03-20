from pathlib import Path

from bench.config import load_engine_targets
from bench.prepare_matrix import build_prepare_variants_from_config_paths


def test_prepared_dir_is_source_scoped(tmp_path: Path):
    cfg = tmp_path / "config.multi.yaml"
    cfg.write_text(
        """
dataset:
  prepared_dir: "data/prepared"
manticore_trec:
  type: "manticore"
  source: "beir/trec-covid"
qdrant_msmarco:
  type: "qdrant"
  source: "msmarco-passage/dev/small"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    targets = load_engine_targets(cfg)
    by_name = {t.name: t.cfg.dataset.prepared_dir for t in targets}
    assert by_name["manticore_trec"].endswith("data/prepared/beir-trec-covid")
    assert by_name["qdrant_msmarco"].endswith(
        "data/prepared/msmarco-passage-dev-small"
    )


def test_prepare_variants_dedup_by_source_and_dir(tmp_path: Path):
    cfg = tmp_path / "config.multi.yaml"
    cfg.write_text(
        """
dataset:
  prepared_dir: "data/prepared"
manticore_trec:
  type: "manticore"
  source: "beir/trec-covid"
qdrant_trec:
  type: "qdrant"
  source: "beir/trec-covid"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_prepare_variants_from_config_paths([cfg])
    assert len(variants) == 1
    v = variants[0]
    assert v.source == "beir/trec-covid"
    assert v.prepared_dir.endswith("data/prepared/beir-trec-covid")
    assert "manticore_trec" in v.entry
    assert "qdrant_trec" in v.entry


def test_prepared_dir_includes_max_docs_variant(tmp_path: Path):
    cfg = tmp_path / "config.multi.yaml"
    cfg.write_text(
        """
dataset:
  prepared_dir: "data/prepared"
msmarco_full:
  type: "manticore"
  source: "msmarco-passage/dev/small"
msmarco_lite:
  type: "manticore"
  source: "msmarco-passage/dev/small"
  max_docs: 250000
""".strip()
        + "\n",
        encoding="utf-8",
    )
    targets = load_engine_targets(cfg)
    by_name = {t.name: t.cfg.dataset.prepared_dir for t in targets}
    assert by_name["msmarco_full"].endswith("msmarco-passage-dev-small")
    assert by_name["msmarco_lite"].endswith(
        "msmarco-passage-dev-small--maxdocs-250000"
    )
