from typing import Any

from bench.config import AppConfig
from bench.indexing import _run_index_mariadb


def test_mariadb_index_ddl_uses_unbounded_socket_and_forwards_progress(monkeypatch) -> None:
    instances: list[Any] = []

    class _Backend:
        def __init__(self, cfg: Any, *, timeout_seconds: int | None = None):
            self.timeout_seconds = timeout_seconds
            instances.append(self)

        def ensure_accessible(self) -> None:
            return None

        def recreate_table(self, *, dimension: int | None = None) -> None:
            return None

        def bulk_index(self, docs: list[dict[str, Any]], *, with_vector: bool) -> int:
            return len(docs)

        def create_search_indexes(
            self,
            *,
            with_vector: bool,
            progress_callback: Any = None,
            progress_interval_seconds: float = 5.0,
        ) -> None:
            assert with_vector is False
            assert progress_interval_seconds == 2.0
            progress_callback(
                {
                    "index_name": "fulltext",
                    "connection_id": 42,
                    "progress_percent": 25.0,
                    "database_state": "copy to tmp table",
                    "database_stage": 1,
                    "database_max_stage": 4,
                    "stage_progress_percent": 100.0,
                }
            )

        def close(self) -> None:
            return None

    cfg = AppConfig()
    cfg.mariadb.index_timeout_seconds = 3600
    cfg.mariadb.index_progress_interval_seconds = 2.0
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr("bench.indexing.read_jsonl", lambda path: [{"id": 1}])
    monkeypatch.setattr("bench.indexing.MariaDBBackend", _Backend)

    results = _run_index_mariadb(cfg, mode="fulltext", progress_callback=updates.append)

    assert results[0].rows_inserted == 1
    assert any(instance.timeout_seconds == 0 for instance in instances)
    build = next(update for update in updates if update["stage"] == "build_fulltext")
    assert build["progress_percent"] == 25.0
    assert build["database_state"] == "copy to tmp table"
