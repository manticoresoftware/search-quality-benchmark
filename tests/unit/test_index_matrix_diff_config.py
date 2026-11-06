from pathlib import Path

from bench.config import AppConfig
from bench.index_matrix import (
    IndexVariant,
    _common_detail_keys,
    _details_lines,
    build_index_variants_from_config_paths,
)


def _variant(entry: str, *, source: str, table: str) -> IndexVariant:
    cfg = AppConfig()
    cfg.dataset.source = source
    cfg.manticore.fulltext_table = table
    return IndexVariant(
        config_path="config.yaml",
        entry=entry,
        engine="manticore",
        mode="fulltext",
        model="index_field_lengths='1'",
        batch_size=500,
        target=table,
        cfg=cfg,
    )


def test_common_detail_keys_hide_identical_values() -> None:
    v1 = _variant("a", source="beir/trec-covid", table="ft_a")
    v2 = _variant("b", source="beir/trec-covid", table="ft_b")

    hidden = _common_detail_keys([v1, v2], v1)

    assert "dataset.source" in hidden
    assert "manticore.fulltext_table" not in hidden


def test_details_lines_show_only_diff_values_when_common_hidden() -> None:
    v1 = _variant("a", source="beir/trec-covid", table="ft_a")
    v2 = _variant("b", source="beir/trec-covid", table="ft_b")

    hidden = _common_detail_keys([v1, v2], v1)
    lines = _details_lines(v1, hidden_common_keys=hidden, comparison_scope="visible manticore rows (2)")
    joined = "\n".join(lines)

    assert "common across visible manticore rows (2):" in joined
    assert "dataset.source:" not in joined
    assert "manticore.fulltext_table: ft_a" in joined


def test_common_detail_keys_compare_only_same_engine() -> None:
    v1 = _variant("a", source="beir/trec-covid", table="ft_a")
    v2 = _variant("b", source="beir/trec-covid", table="ft_b")
    qcfg = AppConfig()
    q = IndexVariant(
        config_path="config.yaml",
        entry="q",
        engine="qdrant",
        mode="fulltext",
        model="Qdrant/bm25",
        batch_size=500,
        target="bm25_q",
        cfg=qcfg,
    )

    hidden = _common_detail_keys([v1, v2, q], v1)

    assert "dataset.source" in hidden


def test_combined_manticore_target_has_separate_fulltext_and_vector_builds(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
manticore:
  type: manticore
  index_layout: combined
  fulltext_table: docs
  vector_table: docs
manticore_fulltext:
  base: manticore
  type: fulltext
manticore_vector:
  base: manticore
  type: vector
manticore_hybrid:
  base: manticore
  type: hybrid
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = build_index_variants_from_config_paths([config_path])

    assert [(row.mode, row.target_kind, row.entry) for row in rows] == [
        ("fulltext", "fulltext", "manticore_fulltext"),
        ("hybrid", "hybrid", "manticore_hybrid"),
    ]
    assert rows[0].used_by == ("manticore_fulltext",)
    assert rows[1].used_by == ("manticore_hybrid", "manticore_vector")


def test_combined_qdrant_target_has_fulltext_and_native_hybrid_builds(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
qdrant:
  type: qdrant
  index_layout: combined
  bm25_collection: docs
  dense_collection: docs
qdrant_fulltext:
  base: qdrant
  type: fulltext
qdrant_vector:
  base: qdrant
  type: vector
qdrant_hybrid:
  base: qdrant
  type: hybrid
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = build_index_variants_from_config_paths([config_path])

    assert [(row.mode, row.target_kind, row.entry) for row in rows] == [
        ("fulltext", "fulltext", "qdrant_fulltext"),
        ("hybrid", "hybrid", "qdrant_hybrid"),
    ]
    assert rows[0].used_by == ("qdrant_fulltext",)
    assert rows[1].used_by == ("qdrant_hybrid", "qdrant_vector")
    assert "native-rrf" in rows[1].model
