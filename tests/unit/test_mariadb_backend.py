import struct
from pathlib import Path
from typing import Any

from bench.config import MariaDBConfig, load_engine_targets
from bench.index_matrix import build_index_variants_from_config_paths
from bench.mariadb_backend import MariaDBBackend
from bench.run_matrix import build_variants_from_config_paths


class _FakeCursor:
    def __init__(self, statements: list[tuple[str, Any]]):
        self.statements = statements

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> int:
        self.statements.append((sql, params))
        return 0

    def executemany(self, sql: str, params: Any = None) -> int:
        self.statements.append((sql, params))
        return len(params or [])


class _FakeConnection:
    def __init__(self):
        self.statements: list[tuple[str, Any]] = []
        self.commits = 0

    def ping(self, reconnect: bool = True) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.statements)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_mariadb_schema_uses_fulltext_and_vector_indexes() -> None:
    backend = MariaDBBackend(MariaDBConfig(table_name="docs", vector_field="embedding"))
    connection = _FakeConnection()
    backend._conn = connection  # type: ignore[assignment]

    backend.recreate_table(dimension=3)
    backend.create_search_indexes(with_vector=True)

    sql = "\n".join(statement for statement, _ in connection.statements)
    assert "VECTOR(3) NOT NULL" in sql
    assert "ADD FULLTEXT INDEX `bench_fulltext` (`title`, `abstract`)" in sql
    assert "ADD VECTOR INDEX `bench_vector` (`embedding`) M=8 DISTANCE=cosine" in sql
    assert connection.commits == 2


def test_mariadb_connection_can_override_query_timeout(monkeypatch) -> None:
    connect_args: dict[str, Any] = {}
    connection = _FakeConnection()

    def _connect(**kwargs: Any) -> _FakeConnection:
        connect_args.update(kwargs)
        return connection

    monkeypatch.setattr("bench.mariadb_backend.pymysql.connect", _connect)
    backend = MariaDBBackend(
        MariaDBConfig(query_timeout_seconds=60),
        timeout_seconds=3600,
    )

    assert backend._connection() is connection
    assert connect_args["read_timeout"] == 3600
    assert connect_args["write_timeout"] == 3600


def test_mariadb_connection_can_disable_socket_timeout(monkeypatch) -> None:
    connect_args: dict[str, Any] = {}
    connection = _FakeConnection()

    def _connect(**kwargs: Any) -> _FakeConnection:
        connect_args.update(kwargs)
        return connection

    monkeypatch.setattr("bench.mariadb_backend.pymysql.connect", _connect)
    backend = MariaDBBackend(MariaDBConfig(), timeout_seconds=0)

    assert backend._connection() is connection
    assert connect_args["read_timeout"] is None
    assert connect_args["write_timeout"] is None


def test_mariadb_bulk_index_binds_vector_as_packed_float32() -> None:
    backend = MariaDBBackend(MariaDBConfig(table_name="docs", vector_field="embedding"))
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
    assert "VEC_FromText" not in sql
    assert "CAST(%s AS BINARY)" in sql
    assert rows[0][-1] == struct.pack("<3f", 1.0, 2.0, 3.0)


def test_mariadb_process_status_converts_stage_progress_to_overall_progress() -> None:
    class _StatusCursor(_FakeCursor):
        def fetchone(self) -> dict[str, Any]:
            return {
                "connection_id": 42,
                "server_elapsed_seconds": 30,
                "database_state": "copy to tmp table",
                "database_stage": 1,
                "database_max_stage": 4,
                "stage_progress_percent": 32.0,
            }

    class _StatusConnection(_FakeConnection):
        def cursor(self) -> _StatusCursor:
            return _StatusCursor(self.statements)

    backend = MariaDBBackend(MariaDBConfig())
    backend._conn = _StatusConnection()  # type: ignore[assignment]

    status = backend._process_status(42)

    assert status is not None
    assert status["progress_percent"] == 8.0
    assert status["database_state"] == "copy to tmp table"


def test_mariadb_hybrid_uses_weighted_rrf(monkeypatch) -> None:
    cfg = MariaDBConfig(hybrid_alpha=0.5, hybrid_rrf_k=60, hybrid_candidates=10)
    backend = MariaDBBackend(cfg)
    monkeypatch.setattr(
        backend,
        "search_lexical",
        lambda query_text, limit: [{"doc_id": "lexical"}, {"doc_id": "shared"}],
    )
    monkeypatch.setattr(
        backend,
        "search_vector",
        lambda query_vector, limit: [{"doc_id": "vector"}, {"doc_id": "shared"}],
    )

    rows = backend.search_hybrid("query", [1.0, 0.0], limit=3)

    assert [row["doc_id"] for row in rows] == ["shared", "lexical", "vector"]


def test_mariadb_profiles_appear_in_index_and_run_matrices(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mariadb:
  type: mariadb
  mariadb:
    table_name: docs
mariadb_fulltext:
  base: mariadb
  type: fulltext
mariadb_vector:
  base: mariadb
  type: vector
mariadb_hybrid:
  base: mariadb
  type: hybrid
""".strip()
        + "\n",
        encoding="utf-8",
    )

    targets = [target for target in load_engine_targets(config_path) if target.type == "mariadb"]
    run_rows = [row for row in build_variants_from_config_paths([config_path]) if row.engine == "mariadb"]
    index_rows = [row for row in build_index_variants_from_config_paths([config_path]) if row.engine == "mariadb"]

    assert {target.mode_filter for target in targets} >= {"fulltext", "vector", "hybrid"}
    assert {row.mode for row in run_rows} == {"fulltext", "vector", "hybrid"}
    assert [(row.mode, row.target_kind, row.target) for row in index_rows] == [
        ("fulltext", "fulltext", "docs"),
        ("vector", "hybrid", "docs"),
    ]
    assert index_rows[0].used_by == ("mariadb_fulltext",)
    assert index_rows[1].used_by == ("mariadb_hybrid", "mariadb_vector")
