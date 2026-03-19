from bench.config import AppConfig
from bench.runbench import _manticore_fulltext_option_clause, _manticore_vector_option_clause


def test_manticore_search_options_normalized_and_appended() -> None:
    cfg = AppConfig()
    cfg.manticore.search_options = "OPTION fuzzy=1, max_query_time=250;"
    fulltext_clause = _manticore_fulltext_option_clause(cfg)
    vector_clause = _manticore_vector_option_clause(cfg)

    assert "idf='plain,tfidf_unnormalized'" in fulltext_clause
    assert "max_matches=" in fulltext_clause
    assert "OPTION " not in fulltext_clause
    assert "fuzzy=1" in fulltext_clause
    assert "max_query_time=250" in fulltext_clause
    assert vector_clause == "fuzzy=1, max_query_time=250"


def test_manticore_search_options_empty_for_vector_clause() -> None:
    cfg = AppConfig()
    cfg.manticore.search_options = ""
    assert _manticore_vector_option_clause(cfg) == ""
