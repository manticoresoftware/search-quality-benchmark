from pathlib import Path
from typing import Any

from bench.config import SereneDBConfig, load_engine_targets
from bench.index_matrix import build_index_variants_from_config_paths
from bench.run_matrix import build_variants_from_config_paths
from bench.serenedb_backend import SereneDBBackend


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection"):
        self.connection = connection

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.connection.statements.append((sql, params))

    def executemany(self, sql: str, params: Any = None) -> None:
        self.connection.statements.append((sql, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.connection.rows)


class _FakeConnection:
    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self.closed = False
        self.rows = rows or []
        self.statements: list[tuple[str, Any]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def test_serenedb_schema_uses_bm25_and_ivf_index() -> None:
    backend = SereneDBBackend(SereneDBConfig(table_name="docs"))
    connection = _FakeConnection()
    backend._conn = connection  # type: ignore[assignment]

    backend.recreate_table(dimension=3)
    backend.create_search_index(with_vector=True)

    sql = "\n".join(statement for statement, _ in connection.statements)
    assert '"embedding" FLOAT[3] NOT NULL' in sql
    assert "CREATE TEXT SEARCH DICTIONARY \"docs_en\"" in sql
    assert '"title" "docs_en"' in sql
    assert '"embedding" ivf (metric = \'cosine\', quant = \'sq8\')' in sql
    assert "VACUUM (REFRESH_TABLE) \"docs\"" in sql


def test_serenedb_bulk_index_binds_float_vector() -> None:
    backend = SereneDBBackend(SereneDBConfig(table_name="docs"))
    connection = _FakeConnection()
    backend._conn = connection  # type: ignore[assignment]

    count = backend.bulk_index(
        [
            {
                "id": 1,
                "doc_id": "doc-1",
                "title": "title",
                "abstract": "abstract",
                "doi": "",
                "date": "",
                "embedding": [1.0, 2.0, 3.0],
            }
        ],
        with_vector=True,
    )

    sql, rows = connection.statements[-1]
    assert count == 1
    assert '"embedding"' in sql
    assert "%s::FLOAT[3]" in sql
    assert rows[0][-1] == [1.0, 2.0, 3.0]


def test_serenedb_hybrid_query_uses_weighted_rrf(monkeypatch) -> None:
    backend = SereneDBBackend(
        SereneDBConfig(
            table_name="docs",
            hybrid_alpha=0.25,
            hybrid_rrf_k=42,
            hybrid_candidates=10,
            vector_query_mode="exact",
        )
    )
    calls: list[tuple[str, int]] = []

    def search_lexical(query: str, *, limit: int) -> list[dict[str, Any]]:
        calls.append(("lexical", limit))
        return [{"doc_id": "lexical"}, {"doc_id": "shared"}]

    def search_vector(vector: list[float], *, limit: int) -> list[dict[str, Any]]:
        calls.append(("vector", limit))
        return [{"doc_id": "vector"}, {"doc_id": "shared"}]

    monkeypatch.setattr(backend, "search_lexical", search_lexical)
    monkeypatch.setattr(backend, "search_vector", search_vector)

    rows = backend.search_hybrid("apple", [1.0, 0.0], limit=3)

    assert calls == [("lexical", 10), ("vector", 10)]
    assert [row["doc_id"] for row in rows] == ["shared", "lexical", "vector"]


def test_serenedb_lexical_query_uses_configured_minimum_match() -> None:
    backend = SereneDBBackend(
        SereneDBConfig(
            table_name="docs",
            fulltext_min_match_ratio=0.25,
        )
    )
    connection = _FakeConnection([{"doc_id": "doc-1", "score": 0.1}])
    backend._conn = connection  # type: ignore[assignment]

    rows = backend.search_lexical("one two three four five", limit=3)

    sql, params = connection.statements[-1]
    assert 'has_any_tokens("title", %s, 2)' in sql
    assert 'has_any_tokens("abstract", %s, 2)' in sql
    assert params == (
        ["one", "two", "three", "four", "five"],
        ["one", "two", "three", "four", "five"],
        3,
    )
    assert rows == [{"doc_id": "doc-1", "score": 0.1}]


def test_serenedb_vector_query_can_use_exact_or_ann_source() -> None:
    connection = _FakeConnection([{"doc_id": "doc-1", "distance": 0.1}])
    exact = SereneDBBackend(
        SereneDBConfig(table_name="docs", vector_query_mode="exact")
    )
    exact._conn = connection  # type: ignore[assignment]
    exact.search_vector([1.0, 0.0], limit=3)
    assert 'FROM "docs" ORDER BY distance' in connection.statements[-1][0]

    indexed = SereneDBBackend(
        SereneDBConfig(table_name="docs", vector_query_mode="index")
    )
    indexed._conn = connection  # type: ignore[assignment]
    indexed.search_vector([1.0, 0.0], limit=3)
    assert 'FROM "docs_idx" ORDER BY distance' in connection.statements[-1][0]


def test_serenedb_profiles_appear_in_index_and_run_matrices(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
serenedb:
  type: serenedb
  table_name: docs
serenedb_fulltext:
  base: serenedb
  type: fulltext
serenedb_vector:
  base: serenedb
  type: vector
serenedb_hybrid:
  base: serenedb
  type: hybrid
""".strip()
        + "\n",
        encoding="utf-8",
    )

    targets = [target for target in load_engine_targets(config_path) if target.type == "serenedb"]
    run_rows = [row for row in build_variants_from_config_paths([config_path]) if row.engine == "serenedb"]
    index_rows = [row for row in build_index_variants_from_config_paths([config_path]) if row.engine == "serenedb"]

    assert {target.mode_filter for target in targets} >= {"fulltext", "vector", "hybrid"}
    assert {row.mode for row in run_rows} == {"fulltext", "vector", "hybrid"}
    assert [(row.mode, row.target_kind, row.target) for row in index_rows] == [
        ("fulltext", "fulltext", "docs"),
        ("vector", "vector", "docs"),
    ]
    assert index_rows[0].used_by == ("serenedb_fulltext",)
    assert index_rows[1].used_by == ("serenedb_hybrid", "serenedb_vector")
