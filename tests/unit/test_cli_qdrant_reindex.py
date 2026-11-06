from __future__ import annotations

from typing import Any

import pytest

from bench import cli
from bench.config import AppConfig


class _NonTTYInput:
    def isatty(self) -> bool:
        return False


class _TTYInput:
    def isatty(self) -> bool:
        return True


class _FakeQdrantBackend:
    has_vector = True

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def connection_target(self) -> str:
        return "http://127.0.0.1:6333"

    def collection_supports(self, collection_name: str, mode: str) -> bool:
        return self.has_vector and mode == "vector"


def _cfg(*, bm25_collection: str, dense_collection: str) -> AppConfig:
    cfg = AppConfig()
    cfg.qdrant.bm25_collection = bm25_collection
    cfg.qdrant.dense_collection = dense_collection
    return cfg


def test_distinct_qdrant_collections_need_no_confirmation(monkeypatch: Any) -> None:
    cfg = _cfg(bm25_collection="docs-bm25", dense_collection="docs-dense")
    monkeypatch.setattr(
        cli,
        "QdrantBackend",
        lambda _cfg: pytest.fail("distinct targets must not be inspected as shared"),
    )

    assert cli._confirm_qdrant_fulltext_reindex_plans([(cfg, "fulltext")]) == "not-needed"


def test_shared_qdrant_fulltext_rebuild_is_blocked_non_interactively(monkeypatch: Any) -> None:
    cfg = _cfg(bm25_collection="docs", dense_collection="docs")
    monkeypatch.setattr(cli, "QdrantBackend", _FakeQdrantBackend)
    monkeypatch.setattr(cli.sys, "stdin", _NonTTYInput())

    with pytest.raises(RuntimeError, match="removes its dense vector index and data"):
        cli._confirm_qdrant_fulltext_reindex_plans([(cfg, "fulltext")])


def test_shared_qdrant_fulltext_rebuild_offers_skip(monkeypatch: Any) -> None:
    cfg = _cfg(bm25_collection="docs", dense_collection="docs")
    monkeypatch.setattr(cli, "QdrantBackend", _FakeQdrantBackend)
    monkeypatch.setattr(cli.sys, "stdin", _TTYInput())
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "skip")

    assert cli._confirm_qdrant_fulltext_reindex_plans([(cfg, "fulltext")]) == "skip"
    assert "skip all selected Qdrant full-text-only builds" in prompts[0]
    assert "use the hybrid build" in prompts[0]


def test_yes_allows_shared_qdrant_fulltext_rebuild(monkeypatch: Any) -> None:
    cfg = _cfg(bm25_collection="docs", dense_collection="docs")
    monkeypatch.setattr(cli, "QdrantBackend", _FakeQdrantBackend)

    assert (
        cli._confirm_qdrant_fulltext_reindex_plans(
            [(cfg, "fulltext")], auto_yes=True
        )
        == "yes"
    )
