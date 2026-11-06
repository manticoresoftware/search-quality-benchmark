from __future__ import annotations

from bench.cli import _print_index_results
from bench.indexing import ModeIndexResult


def test_print_index_results_includes_total_elapsed_time(capsys) -> None:
    result = ModeIndexResult(
        mode="fulltext",
        duration_seconds=143.25,
        rows_inserted=8_841_823,
        docs_per_second=61_722.149,
        failed_batches=0,
        retried_batches=0,
    )

    _print_index_results([result], engine="meilisearch")

    assert capsys.readouterr().out == (
        "Index completed [meilisearch:fulltext] inserted=8841823 "
        "total_elapsed=143.2s avg_rate=61722.1 docs/s\n"
    )


def test_print_index_results_keeps_issue_details(capsys) -> None:
    result = ModeIndexResult(
        mode="vector",
        duration_seconds=12.0,
        rows_inserted=99,
        docs_per_second=8.25,
        failed_batches=1,
        retried_batches=2,
        last_error="request failed",
        error_reasons={"timeout": 2},
    )

    _print_index_results([result], engine="qdrant")

    output = capsys.readouterr().out
    assert "Index completed [qdrant:vector]" in output
    assert "total_elapsed=12.0s" in output
    assert "Index issues [vector]: retries=2 failed=1" in output
    assert "2  timeout" in output
