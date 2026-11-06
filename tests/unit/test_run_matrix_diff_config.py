from bench.config import AppConfig
from bench.run_matrix import RunVariant, _common_detail_keys, _details_lines


def _variant(entry: str, *, source: str, table: str) -> RunVariant:
    cfg = AppConfig()
    cfg.dataset.source = source
    cfg.manticore.fulltext_table = table
    return RunVariant(
        config_path="config.yaml",
        entry=entry,
        engine="manticore",
        mode="fulltext",
        model="[search:ranker=bm25]",
        default_concurrency=1,
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
    q = RunVariant(
        config_path="config.yaml",
        entry="q",
        engine="qdrant",
        mode="fulltext",
        model="Qdrant/bm25",
        default_concurrency=1,
        cfg=AppConfig(),
    )

    hidden = _common_detail_keys([v1, v2, q], v1)

    assert "dataset.source" in hidden
