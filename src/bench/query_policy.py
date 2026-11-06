from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


QUERY_VARIANT_SAME = "same"
TREC_COVID_DATASET = "beir/trec-covid"
TREC_COVID_FAMILY = "trec-covid"
TREC_COVID_QUERY_FIELDS = ("query", "question", "narrative")
TREC_COVID_BENCHMARKS = {
    "trec-covid-query": "query",
    "trec-covid-question": "question",
    "trec-covid-narrative": "narrative",
}


def is_trec_covid(dataset_source: str) -> bool:
    return str(dataset_source or "").strip().lower() == TREC_COVID_DATASET


def benchmark_views(dataset_source: str) -> tuple[tuple[str, str], ...]:
    """Return logical benchmark ids and topic fields for one physical dataset."""
    if is_trec_covid(dataset_source):
        return tuple(TREC_COVID_BENCHMARKS.items())
    return ((str(dataset_source or "").strip(), ""),)


def query_field_for_benchmark(benchmark_dataset_id: str) -> str:
    return TREC_COVID_BENCHMARKS.get(str(benchmark_dataset_id or "").strip().lower(), "")


def benchmark_id_for_query_field(dataset_source: str, query_field: str) -> str:
    if is_trec_covid(dataset_source):
        field = str(query_field or "").strip().lower()
        for benchmark_id, candidate in TREC_COVID_BENCHMARKS.items():
            if candidate == field:
                return benchmark_id
    return str(dataset_source or "").strip()


def legacy_trec_covid_query_field(mode: str) -> str:
    """Map artifacts from before logical benchmark ids to their actual text field."""
    return "query" if str(mode or "").strip().lower() == "fulltext" else "narrative"


def topic_query_variants(topic: Mapping[str, Any]) -> dict[str, str]:
    """Return explicit topic variants, with compatibility for prepared v1 topics."""
    variants = {
        field: str(topic.get(field) or "").strip()
        for field in TREC_COVID_QUERY_FIELDS
        if str(topic.get(field) or "").strip()
    }
    if variants:
        return variants
    short_query, long_query = resolved_topic_queries(topic)
    out: dict[str, str] = {}
    if short_query:
        out["query"] = short_query
    if long_query:
        out["narrative"] = long_query
    return out


def query_text_for_field(
    topic: Mapping[str, Any],
    query_field: str,
    *,
    mode: str = "",
    fulltext_query_variant: str = "long",
) -> str:
    """Resolve one query text, retaining the old short/long policy as fallback."""
    field = str(query_field or "").strip().lower()
    if field:
        value = str(topic.get(field) or "").strip()
        if value:
            return value
        # Prepared v1 TREC-COVID data can still run query/narrative views. The
        # question view intentionally has no fallback because that text was not
        # preserved and must be regenerated with `bench prepare`.
        short_query, long_query = resolved_topic_queries(topic)
        if field == "query" and short_query:
            return short_query
        if field == "narrative" and long_query:
            return long_query
        raise ValueError(
            f"Prepared topic is missing the '{field}' query field. Please rerun: bench prepare"
        )

    short_query, long_query = resolved_topic_queries(topic)
    if not short_query and not long_query:
        raise ValueError(
            "Prepared topics are missing query_short/query_long or explicit query fields. "
            "Please rerun: bench prepare"
        )
    if str(mode or "").strip().lower() == "fulltext":
        return long_query if str(fulltext_query_variant or "long").strip().lower() == "long" else short_query
    return long_query


def query_texts_for_embedding(
    topic: Mapping[str, Any],
    dataset_source: str,
) -> tuple[str, ...]:
    """Return every query text that vector/hybrid logical views may require."""
    if is_trec_covid(dataset_source):
        variants = topic_query_variants(topic)
        return tuple(variants[field] for field in TREC_COVID_QUERY_FIELDS if variants.get(field))
    short_query, long_query = resolved_topic_queries(topic)
    return (long_query or short_query,) if (long_query or short_query) else ()


def trec_covid_artifact_identity(
    *,
    mode: str,
    benchmark_dataset_id: Any = None,
    query_field: Any = None,
    recorded_variant: Any = None,
) -> tuple[str, str]:
    """Resolve explicit identity, or apply the agreed mode-based legacy map."""
    benchmark_id = str(benchmark_dataset_id or "").strip().lower()
    field = query_field_for_benchmark(benchmark_id)
    if not field:
        candidate = str(query_field or "").strip().lower()
        if candidate in TREC_COVID_QUERY_FIELDS:
            field = candidate
        else:
            # query_variant belonged to the superseded short/long policy and is
            # not a reliable logical-dataset identifier. Only the new explicit
            # fields override the agreed migration rule.
            field = legacy_trec_covid_query_field(mode)
        benchmark_id = benchmark_id_for_query_field(TREC_COVID_DATASET, field)
    return benchmark_id, field


def resolved_topic_queries(topic: Mapping[str, Any]) -> tuple[str, str]:
    """Return the short and long texts after applying benchmark fallbacks."""
    short_query = str(topic.get("query_short") or topic.get("query") or "").strip()
    long_query = str(topic.get("query_long") or topic.get("narrative") or "").strip()
    if not short_query:
        short_query = long_query
    if not long_query:
        long_query = short_query
    return short_query, long_query


def query_variants_distinct(topics: Iterable[Mapping[str, Any]]) -> bool:
    """Whether selecting an available query field changes at least one topic."""
    for topic in topics:
        values = set(topic_query_variants(topic).values())
        if len(values) > 1:
            return True
    return False


def dataset_query_variants_distinct(
    dataset_source: str,
    inspected_distinct: bool | None,
) -> bool:
    """Resolve legacy query identity when an artifact has no variant metadata.

    Prepared-topic inspection takes precedence. If those files are unavailable,
    TREC-COVID is the only current dataset assumed to have distinct short and
    long texts; other supported datasets use the same text for both fields.
    """
    if isinstance(inspected_distinct, bool):
        return inspected_distinct
    return str(dataset_source or "").strip().lower() == TREC_COVID_DATASET


def requested_query_variant(mode: str, fulltext_query_variant: str = "long") -> str:
    if str(mode or "").strip().lower() == "fulltext":
        configured = str(fulltext_query_variant or "long").strip().lower()
        return "long" if configured == "long" else "short"
    return "long"


def query_variant_metadata(
    topics: Iterable[Mapping[str, Any]],
    modes: Iterable[str],
    *,
    fulltext_query_variant: str,
) -> dict[str, Any]:
    """Build artifact metadata describing requested and comparable query sets."""
    topics_list = list(topics)
    distinct = query_variants_distinct(topics_list)
    requested_by_mode = {
        str(mode): requested_query_variant(str(mode), fulltext_query_variant)
        for mode in modes
    }
    effective_by_mode = {
        mode: requested if distinct else QUERY_VARIANT_SAME
        for mode, requested in requested_by_mode.items()
    }
    requested_values = set(requested_by_mode.values())
    effective_values = set(effective_by_mode.values())
    return {
        "query_variant_requested": (
            next(iter(requested_values)) if len(requested_values) == 1 else "mixed"
        ),
        "query_variant": (
            next(iter(effective_values)) if len(effective_values) == 1 else "mixed"
        ),
        "query_variants_distinct": distinct,
        "query_variants_requested_by_mode": requested_by_mode,
        "query_variants_by_mode": effective_by_mode,
    }


def legacy_requested_query_variant(mode: str) -> str:
    """Historical artifacts predate metadata and used short only for full-text."""
    return "short" if str(mode or "").strip().lower() == "fulltext" else "long"


def effective_query_variant(
    *,
    mode: str,
    recorded_variant: Any = None,
    recorded_requested_variant: Any = None,
    variants_distinct: bool | None = None,
) -> tuple[str, str]:
    """Resolve requested/effective variants for new and legacy artifacts."""
    requested = str(recorded_requested_variant or "").strip().lower()
    if requested not in {"short", "long", "query", "question", "narrative", "mixed"}:
        requested = legacy_requested_query_variant(mode)

    recorded = str(recorded_variant or "").strip().lower()
    if recorded in {
        "short",
        "long",
        "query",
        "question",
        "narrative",
        QUERY_VARIANT_SAME,
        "mixed",
    }:
        effective = recorded
    elif variants_distinct is False:
        effective = QUERY_VARIANT_SAME
    else:
        effective = requested

    if variants_distinct is False:
        effective = QUERY_VARIANT_SAME
    return requested, effective
