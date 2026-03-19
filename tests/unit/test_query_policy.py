from bench.runbench import query_text_for_mode


def test_query_policy_prefers_short_for_fulltext():
    topic = {"query_short": "short q", "query_long": "longer query text"}
    assert query_text_for_mode(topic, "fulltext") == "short q"


def test_query_policy_prefers_long_for_vector():
    topic = {"query_short": "short q", "query_long": "longer query text"}
    assert query_text_for_mode(topic, "vector") == "longer query text"


def test_query_policy_uses_long_when_short_missing():
    topic = {"query_short": "", "query_long": "longer query text"}
    assert query_text_for_mode(topic, "fulltext") == "longer query text"
    assert query_text_for_mode(topic, "vector") == "longer query text"


def test_query_policy_raises_on_missing_short_and_long():
    topic = {}
    try:
        query_text_for_mode(topic, "fulltext")
    except ValueError as exc:
        assert "missing query_short/query_long" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing query fields")
