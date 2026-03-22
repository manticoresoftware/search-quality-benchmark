from bench.config import AppConfig
from bench.runbench import _manticore_fulltext_match_query, _manticore_fulltext_ranker_label


def test_manticore_fulltext_query_mode_and() -> None:
    cfg = AppConfig()
    cfg.manticore.fulltext_query_mode = "and"
    assert _manticore_fulltext_match_query(cfg, "Do Cholesterol Statin Drugs Cause Breast Cancer?") == (
        "Do Cholesterol Statin Drugs Cause Breast Cancer"
    )


def test_manticore_fulltext_query_mode_or() -> None:
    cfg = AppConfig()
    cfg.manticore.fulltext_query_mode = "or"
    assert _manticore_fulltext_match_query(cfg, "a b c") == "a | b | c"


def test_manticore_fulltext_query_mode_quorum() -> None:
    cfg = AppConfig()
    cfg.manticore.fulltext_query_mode = "quorum"
    cfg.manticore.fulltext_query_quorum = 0.3
    assert _manticore_fulltext_match_query(cfg, "a b c") == "\"a b c\"/0.3"


def test_manticore_fulltext_query_mode_or_drops_underscore_only_tokens() -> None:
    cfg = AppConfig()
    cfg.manticore.fulltext_query_mode = "or"
    assert _manticore_fulltext_match_query(cfg, "_____ | is | the | answer") == "is | the | answer"


def test_manticore_fulltext_label_includes_query_mode() -> None:
    cfg = AppConfig()
    cfg.manticore.search_options = "ranker=bm25"
    cfg.manticore.fulltext_options = "index_field_lengths='1'"
    cfg.manticore.fulltext_query_mode = "quorum"
    cfg.manticore.fulltext_query_quorum = 0.5
    label = _manticore_fulltext_ranker_label(cfg)
    assert "search:ranker=bm25" in label
    assert "query:quorum/0.5" in label
