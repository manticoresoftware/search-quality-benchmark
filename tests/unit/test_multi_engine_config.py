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
meili_main:
  type: "meilisearch"
  meilisearch:
    host: "http://127.0.0.1:7700"
osearch_main:
  type: "opensearch"
  opensearch:
    url: "http://127.0.0.1:9200"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    targets = load_engine_targets(cfg)
    assert len(targets) == 5
    by_name = {t.name: t.type for t in targets}
    assert by_name == {
        "manticore_default": "manticore",
        "manticore_bm25": "manticore",
        "qdrant_main": "qdrant",
        "meili_main": "meilisearch",
        "osearch_main": "opensearch",
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
meili:
  type: "meilisearch"
osearch:
  type: "opensearch"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    variants = build_variants_from_config_paths([cfg])
    # The referenced mant_a entry is an internal base. The remaining ten
    # engine/mode combinations are each expanded into three TREC-COVID views.
    assert len(variants) == 30
    assert {v.entry for v in variants} == {"mant_b", "qdr", "meili", "osearch"}
    assert {v.benchmark_dataset_id for v in variants} == {
        "trec-covid-query",
        "trec-covid-question",
        "trec-covid-narrative",
    }
    assert all(v.default_concurrency == 2 for v in variants)
    assert {v.mode for v in variants if v.entry == "meili"} == {"fulltext", "vector", "hybrid"}
    assert {v.mode for v in variants if v.entry == "osearch"} == {"fulltext", "vector", "hybrid"}
