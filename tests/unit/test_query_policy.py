from bench.runbench import query_text_for_mode
from bench.query_policy import dataset_query_variants_distinct, query_variant_metadata


def test_query_policy_uses_long_default_for_fulltext():
    topic = {"query_short": "short q", "query_long": "longer query text"}
    assert query_text_for_mode(topic, "fulltext") == "longer query text"


def test_query_policy_can_select_short_for_fulltext():
    topic = {"query_short": "short q", "query_long": "longer query text"}
    assert query_text_for_mode(topic, "fulltext", fulltext_query_variant="short") == "short q"


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


def test_query_variant_metadata_keeps_distinct_query_sets_separate():
    topics = [{"query_short": "short q", "query_long": "longer query text"}]

    fulltext = query_variant_metadata(
        topics,
        ["fulltext"],
        fulltext_query_variant="long",
    )
    vector = query_variant_metadata(
        topics,
        ["vector"],
        fulltext_query_variant="long",
    )

    assert fulltext["query_variant_requested"] == "long"
    assert fulltext["query_variant"] == "long"
    assert fulltext["query_variants_distinct"] is True
    assert vector["query_variant"] == "long"


def test_query_variant_metadata_collapses_identical_short_and_long_texts():
    topics = [{"query_short": "one query", "query_long": "one query"}]

    fulltext = query_variant_metadata(
        topics,
        ["fulltext"],
        fulltext_query_variant="short",
    )
    vector = query_variant_metadata(
        topics,
        ["vector"],
        fulltext_query_variant="long",
    )

    assert fulltext["query_variant_requested"] == "short"
    assert fulltext["query_variant"] == "same"
    assert fulltext["query_variants_distinct"] is False
    assert vector["query_variant_requested"] == "long"
    assert vector["query_variant"] == "same"


def test_legacy_dataset_fallback_assumes_only_trec_covid_is_distinct():
    assert dataset_query_variants_distinct("beir/trec-covid", None) is True
    assert dataset_query_variants_distinct("beir/scifact/test", None) is False
    assert dataset_query_variants_distinct("future/distinct", True) is True
