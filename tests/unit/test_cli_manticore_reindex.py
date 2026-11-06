from __future__ import annotations

from typing import Any

import pytest

from bench import cli
from bench.config import AppConfig
from bench.indexing import _run_index_manticore


class _FakeManticoreClient:
    existing_tables: set[str] = set()
    checked_tables: list[str] = []

    def __init__(self, _cfg: Any) -> None:
        pass

    def table_exists(self, table: str) -> bool:
        self.checked_tables.append(table)
        return table in self.existing_tables

    def close(self) -> None:
        pass


class _ExistingSharedTableClient:
    dropped_tables: list[str] = []

    def __init__(self, _cfg: Any) -> None:
        pass

    def table_exists(self, _table: str) -> bool:
        return True

    def drop_table_if_exists(self, table: str) -> None:
        self.dropped_tables.append(table)

    def close(self) -> None:
        pass


class _NonTTYInput:
    def isatty(self) -> bool:
        return False


class _TTYInput:
    def isatty(self) -> bool:
        return True


def _cfg(*, fulltext_table: str, vector_table: str) -> AppConfig:
    cfg = AppConfig()
    cfg.manticore.fulltext_table = fulltext_table
    cfg.manticore.vector_table = vector_table
    return cfg


def test_distinct_fulltext_table_needs_no_destructive_confirmation(monkeypatch: Any) -> None:
    cfg = _cfg(fulltext_table="docs_ft", vector_table="docs_vec")
    monkeypatch.setattr(
        cli,
        "ManticoreClient",
        lambda _cfg: pytest.fail("a distinct full-text target must not be inspected as vector data"),
    )

    assert cli._confirm_manticore_reindex_plans([(cfg, "fulltext")]) is True


def test_shared_fulltext_table_is_blocked_non_interactively(monkeypatch: Any) -> None:
    cfg = _cfg(fulltext_table="docs", vector_table="docs")
    _FakeManticoreClient.existing_tables = {"docs"}
    _FakeManticoreClient.checked_tables = []
    monkeypatch.setattr(cli, "ManticoreClient", _FakeManticoreClient)
    monkeypatch.setattr(cli.sys, "stdin", _NonTTYInput())

    with pytest.raises(RuntimeError, match="may lose its vector index and data"):
        cli._confirm_manticore_reindex_plans([(cfg, "fulltext")])

    assert _FakeManticoreClient.checked_tables == ["docs"]


def test_shared_fulltext_table_requires_explicit_yes(monkeypatch: Any) -> None:
    cfg = _cfg(fulltext_table="docs", vector_table="docs")
    _FakeManticoreClient.existing_tables = {"docs"}
    monkeypatch.setattr(cli, "ManticoreClient", _FakeManticoreClient)
    monkeypatch.setattr(cli.sys, "stdin", _TTYInput())
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "no")

    assert cli._confirm_manticore_reindex_plans([(cfg, "fulltext")]) is False
    assert "removes vector field/data" in prompts[0]
    assert "mode=hybrid" in prompts[0]


def test_yes_allows_shared_fulltext_table_rebuild(monkeypatch: Any) -> None:
    cfg = _cfg(fulltext_table="docs", vector_table="docs")
    _FakeManticoreClient.existing_tables = {"docs"}
    monkeypatch.setattr(cli, "ManticoreClient", _FakeManticoreClient)

    assert cli._confirm_manticore_reindex_plans(
        [(cfg, "fulltext")], auto_yes=True
    ) is True


def test_hybrid_rebuild_is_also_confirmed(monkeypatch: Any) -> None:
    cfg = _cfg(fulltext_table="docs", vector_table="docs")
    _FakeManticoreClient.existing_tables = {"docs"}
    monkeypatch.setattr(cli, "ManticoreClient", _FakeManticoreClient)
    monkeypatch.setattr(cli.sys, "stdin", _NonTTYInput())

    with pytest.raises(RuntimeError, match="Refusing to recreate"):
        cli._confirm_manticore_reindex_plans([(cfg, "hybrid")])


def test_index_backend_does_not_drop_shared_table_without_explicit_opt_in(
    monkeypatch: Any,
) -> None:
    cfg = _cfg(fulltext_table="docs", vector_table="docs")
    _ExistingSharedTableClient.dropped_tables = []
    monkeypatch.setattr(
        "bench.indexing.read_jsonl",
        lambda _path: pytest.fail("the destructive guard must run before loading the corpus"),
    )
    monkeypatch.setattr("bench.indexing.ManticoreClient", _ExistingSharedTableClient)

    with pytest.raises(RuntimeError, match="removes its vector field and data"):
        _run_index_manticore(cfg, mode="fulltext")

    assert _ExistingSharedTableClient.dropped_tables == []
