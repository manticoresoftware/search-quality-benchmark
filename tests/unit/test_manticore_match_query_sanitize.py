from bench.runbench import _sanitize_manticore_fulltext_query


def test_sanitize_match_query_removes_operator_punctuation():
    q = "what is a 5/5 of 60 score?"
    out = _sanitize_manticore_fulltext_query(q)
    assert "/" not in out
    assert "?" not in out
    assert out == "what is a 5 5 of 60 score"
