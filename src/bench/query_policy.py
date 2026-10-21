from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


QUERY_VARIANT_SAME = "same"
TREC_COVID_DATASET = "beir/trec-covid"


def resolved_topic_queries(topic: Mapping[str, Any]) -> tuple[str, str]:
    """Return the short and long texts after applying benchmark fallbacks."""
    short_query = str(topic.get("query_short") or "").strip()
    long_query = str(topic.get("query_long") or "").strip()
    if not short_query:
        short_query = long_query
    if not long_query:
        long_query = short_query
    return short_query, long_query


def query_variants_distinct(topics: Iterable[Mapping[str, Any]]) -> bool:
    """Whether selecting short versus long changes at least one query text."""
    return any(
        short_query != long_query
        for short_query, long_query in map(resolved_topic_queries, topics)
    )


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
    if requested not in {"short", "long", "mixed"}:
        requested = legacy_requested_query_variant(mode)

    recorded = str(recorded_variant or "").strip().lower()
    if recorded in {"short", "long", QUERY_VARIANT_SAME, "mixed"}:
        effective = recorded
    elif variants_distinct is False:
        effective = QUERY_VARIANT_SAME
    else:
        effective = requested

    if variants_distinct is False:
        effective = QUERY_VARIANT_SAME
    return requested, effective
