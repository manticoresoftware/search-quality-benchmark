from bench.summarize import _parse_selection, _system_label


def test_parse_selection_with_ranges():
    assert _parse_selection("1,3-5,7", 10) == [1, 3, 4, 5, 7]


def test_parse_selection_bounds():
    assert _parse_selection("0,1,9-12", 10) == [1, 9, 10]


def test_system_label():
    assert _system_label("manticore", "fulltext") == "Manticore full-text (manticore)"
    assert _system_label("qdrant", "fulltext") == "Qdrant full-text (qdrant)"
    assert _system_label("meilisearch", "fulltext") == "Meilisearch full-text (meilisearch)"
    assert _system_label("meilisearch", "hybrid") == "Meilisearch hybrid (meilisearch)"
    assert _system_label("opensearch", "fulltext") == "OpenSearch full-text (opensearch)"
    assert _system_label("opensearch", "hybrid") == "OpenSearch hybrid (opensearch)"
