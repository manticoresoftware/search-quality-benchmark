from __future__ import annotations

import json
import re
import struct
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable

import pymysql
from pymysql.cursors import DictCursor

from .config import MariaDBConfig


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOCUMENT_FIELDS = {"title", "abstract"}


def _identifier(value: str, *, label: str) -> str:
    name = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Invalid MariaDB {label}: {value!r}")
    return f"`{name}`"


class MariaDBBackend:
    def __init__(self, cfg: MariaDBConfig, *, timeout_seconds: int | None = None):
        self.cfg = cfg
        self._timeout_seconds = timeout_seconds
        self._conn: pymysql.connections.Connection | None = None

    def _connection(self) -> pymysql.connections.Connection:
        if self._conn is None:
            configured_timeout = (
                self.cfg.query_timeout_seconds if self._timeout_seconds is None else self._timeout_seconds
            )
            timeout_value = int(configured_timeout)
            timeout = None if timeout_value <= 0 else max(1, timeout_value)
            self._conn = pymysql.connect(
                host=str(self.cfg.host or "127.0.0.1"),
                port=int(self.cfg.port),
                user=str(self.cfg.user or ""),
                password=str(self.cfg.password or ""),
                database=str(self.cfg.database or ""),
                charset="utf8mb4",
                autocommit=False,
                connect_timeout=max(1, int(self.cfg.connect_timeout_seconds)),
                read_timeout=timeout,
                write_timeout=timeout,
                cursorclass=DictCursor,
            )
        else:
            self._conn.ping(reconnect=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def connection_target(self) -> str:
        return f"{self.cfg.host}:{int(self.cfg.port)}/{self.cfg.database}"

    def ensure_accessible(self) -> None:
        try:
            with self._connection().cursor() as cur:
                cur.execute("SELECT VERSION() AS version")
                row = cur.fetchone() or {}
            version = str(row.get("version", ""))
            match = re.match(r"^(\d+)\.(\d+)", version)
            if match and (int(match.group(1)), int(match.group(2))) < (11, 7):
                raise RuntimeError(f"MariaDB 11.7+ is required for vector search; server is {version}")
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"MariaDB is not reachable at {self.connection_target()}: {exc}") from exc

    def table_name(self) -> str:
        return str(self.cfg.table_name or "").strip()

    def _table_sql(self) -> str:
        return _identifier(self.table_name(), label="table name")

    def _vector_field_sql(self) -> str:
        return _identifier(self.cfg.vector_field, label="vector field")

    def _text_fields(self) -> list[str]:
        fields = [str(x or "").strip() for x in (self.cfg.text_fields or ["title", "abstract"])]
        if not fields or any(field not in _DOCUMENT_FIELDS for field in fields):
            raise ValueError("MariaDB text_fields must contain title and/or abstract")
        return list(dict.fromkeys(fields))

    def _text_fields_sql(self) -> str:
        return ", ".join(_identifier(field, label="text field") for field in self._text_fields())

    def table_exists(self, table_name: str) -> bool:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s LIMIT 1",
                (str(self.cfg.database), str(table_name)),
            )
            return cur.fetchone() is not None

    def table_has_vector(self, table_name: str) -> bool:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns AS c "
                "JOIN information_schema.statistics AS s "
                "ON s.table_schema = c.table_schema AND s.table_name = c.table_name "
                "AND s.column_name = c.column_name "
                "WHERE c.table_schema = %s AND c.table_name = %s AND c.column_name = %s "
                "AND c.data_type = 'vector' AND s.index_type = 'VECTOR' LIMIT 1",
                (str(self.cfg.database), str(table_name), str(self.cfg.vector_field)),
            )
            return cur.fetchone() is not None

    def table_has_fulltext(self, table_name: str) -> bool:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.statistics "
                "WHERE table_schema = %s AND table_name = %s AND index_type = 'FULLTEXT' LIMIT 1",
                (str(self.cfg.database), str(table_name)),
            )
            return cur.fetchone() is not None

    def recreate_table(self, *, dimension: int | None = None) -> None:
        table = self._table_sql()
        with self._connection().cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            columns = [
                "`id` BIGINT UNSIGNED NOT NULL",
                "`doc_id` VARCHAR(2048) NOT NULL",
                "`title` LONGTEXT NOT NULL",
                "`abstract` LONGTEXT NOT NULL",
                "`doi` TEXT NOT NULL",
                "`date` VARCHAR(255) NOT NULL",
            ]
            if dimension is not None:
                if int(dimension) <= 0:
                    raise ValueError("MariaDB vector dimension must be positive")
                columns.append(f"{self._vector_field_sql()} VECTOR({int(dimension)}) NOT NULL")
            columns.append("PRIMARY KEY (`id`)")
            cur.execute(
                f"CREATE TABLE {table} ({', '.join(columns)}) "
                "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
            )
        self._connection().commit()

    def _connection_id(self) -> int:
        with self._connection().cursor() as cur:
            cur.execute("SELECT CONNECTION_ID() AS connection_id")
            row = cur.fetchone() or {}
        return int(row.get("connection_id", 0))

    def _process_status(self, connection_id: int) -> dict[str, Any] | None:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT ID AS connection_id, TIME AS server_elapsed_seconds, STATE AS database_state, "
                "STAGE AS database_stage, MAX_STAGE AS database_max_stage, "
                "PROGRESS AS stage_progress_percent "
                "FROM information_schema.PROCESSLIST WHERE ID = %s",
                (int(connection_id),),
            )
            row = cur.fetchone()
        if row is None:
            return None
        status = dict(row)
        stage = int(status.get("database_stage") or 0)
        max_stage = int(status.get("database_max_stage") or 0)
        stage_progress = float(status.get("stage_progress_percent") or 0.0)
        if stage > 0 and max_stage > 0:
            progress = ((stage - 1) + stage_progress / 100.0) / max_stage * 100.0
        else:
            progress = stage_progress
        status["progress_percent"] = min(100.0, max(0.0, progress))
        return status

    def _execute_monitored_ddl(
        self,
        sql: str,
        *,
        index_name: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        progress_interval_seconds: float,
    ) -> None:
        if progress_callback is None:
            with self._connection().cursor() as cur:
                cur.execute(sql)
            return

        connection_id = self._connection_id()
        started = perf_counter()
        stop = Event()

        progress_callback(
            {
                "index_name": index_name,
                "connection_id": connection_id,
                "progress_percent": 0.0,
                "database_state": "starting",
                "database_stage": 0,
                "database_max_stage": 0,
                "stage_progress_percent": 0.0,
                "server_elapsed_seconds": 0,
                "elapsed_seconds": 0.0,
            }
        )

        def _monitor() -> None:
            monitor = MariaDBBackend(self.cfg)
            interval = max(0.1, float(progress_interval_seconds))
            try:
                while not stop.wait(interval):
                    try:
                        status = monitor._process_status(connection_id)
                        if status is None:
                            continue
                        status["index_name"] = index_name
                        status["elapsed_seconds"] = perf_counter() - started
                        progress_callback(status)
                    except Exception as exc:
                        progress_callback(
                            {
                                "index_name": index_name,
                                "connection_id": connection_id,
                                "progress_percent": 0.0,
                                "database_state": f"progress probe failed: {exc}",
                                "database_stage": 0,
                                "database_max_stage": 0,
                                "stage_progress_percent": 0.0,
                                "server_elapsed_seconds": 0,
                                "elapsed_seconds": perf_counter() - started,
                            }
                        )
            finally:
                monitor.close()

        monitor_thread = Thread(target=_monitor, name=f"mariadb-{index_name}-progress", daemon=True)
        monitor_thread.start()
        try:
            with self._connection().cursor() as cur:
                cur.execute(sql)
        finally:
            stop.set()
            monitor_thread.join()
        progress_callback(
            {
                "index_name": index_name,
                "connection_id": connection_id,
                "progress_percent": 100.0,
                "database_state": "complete",
                "database_stage": 0,
                "database_max_stage": 0,
                "stage_progress_percent": 100.0,
                "server_elapsed_seconds": 0,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    def create_search_indexes(
        self,
        *,
        with_vector: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_seconds: float = 5.0,
    ) -> None:
        table = self._table_sql()
        self._execute_monitored_ddl(
            f"ALTER TABLE {table} ADD FULLTEXT INDEX `bench_fulltext` ({self._text_fields_sql()})",
            index_name="fulltext",
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
        )
        if with_vector:
            distance = str(self.cfg.vector_distance or "cosine").strip().lower()
            if distance not in {"cosine", "euclidean"}:
                raise ValueError("MariaDB vector_distance must be cosine or euclidean")
            vector_m = int(self.cfg.vector_m)
            if vector_m < 3 or vector_m > 200:
                raise ValueError("MariaDB vector_m must be between 3 and 200")
            self._execute_monitored_ddl(
                f"ALTER TABLE {table} ADD VECTOR INDEX `bench_vector` ({self._vector_field_sql()}) "
                f"M={vector_m} DISTANCE={distance}",
                index_name="vector",
                progress_callback=progress_callback,
                progress_interval_seconds=progress_interval_seconds,
            )
        self._connection().commit()

    def bulk_index(self, docs: list[dict[str, Any]], *, with_vector: bool) -> int:
        if not docs:
            return 0
        table = self._table_sql()
        columns = "`id`, `doc_id`, `title`, `abstract`, `doi`, `date`"
        placeholders = "%s, %s, %s, %s, %s, %s"
        rows: list[tuple[Any, ...]] = []
        if with_vector:
            columns += f", {self._vector_field_sql()}"
            # PyMySQL escapes bytes as a string literal without a binary
            # introducer. The explicit cast preserves the packed VECTOR value.
            placeholders += ", CAST(%s AS BINARY)"
            for doc in docs:
                vector = [float(x) for x in doc[self.cfg.vector_field]]
                rows.append(
                    (
                        int(doc["id"]),
                        str(doc["doc_id"]),
                        str(doc.get("title", "")),
                        str(doc.get("abstract", "")),
                        str(doc.get("doi", "")),
                        str(doc.get("date", "")),
                        struct.pack(f"<{len(vector)}f", *vector),
                    )
                )
        else:
            for doc in docs:
                rows.append(
                    (
                        int(doc["id"]),
                        str(doc["doc_id"]),
                        str(doc.get("title", "")),
                        str(doc.get("abstract", "")),
                        str(doc.get("doi", "")),
                        str(doc.get("date", "")),
                    )
                )
        try:
            with self._connection().cursor() as cur:
                cur.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", rows)
            self._connection().commit()
        except Exception:
            self._connection().rollback()
            raise
        return len(rows)

    def get_document_count(self, table_name: str) -> int:
        table = _identifier(table_name, label="table name")
        with self._connection().cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
            row = cur.fetchone() or {}
        return int(row.get("count", 0))

    def _fulltext_modifier(self) -> str:
        mode = str(self.cfg.fulltext_search_mode or "natural").strip().lower()
        if mode == "natural":
            return "IN NATURAL LANGUAGE MODE"
        if mode == "boolean":
            return "IN BOOLEAN MODE"
        if mode in {"query_expansion", "query-expansion"}:
            return "IN NATURAL LANGUAGE MODE WITH QUERY EXPANSION"
        raise ValueError("MariaDB fulltext_search_mode must be natural, boolean, or query_expansion")

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        match = f"MATCH ({self._text_fields_sql()}) AGAINST (%s {self._fulltext_modifier()})"
        sql = (
            f"SELECT `doc_id`, {match} AS score FROM {self._table_sql()} "
            f"WHERE {match} ORDER BY score DESC LIMIT {max(1, int(limit))}"
        )
        with self._connection().cursor() as cur:
            cur.execute(sql, (str(query_text), str(query_text)))
            rows = list(cur.fetchall())
        return [
            {"doc_id": str(row.get("doc_id", "")), "score": float(row.get("score") or 0.0)}
            for row in rows
        ]

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        vector_text = json.dumps([float(x) for x in query_vector], separators=(",", ":"))
        sql = (
            f"SELECT `doc_id`, VEC_DISTANCE({self._vector_field_sql()}, VEC_FromText(%s)) AS distance "
            f"FROM {self._table_sql()} ORDER BY distance ASC LIMIT {max(1, int(limit))}"
        )
        with self._connection().cursor() as cur:
            cur.execute(sql, (vector_text,))
            rows = list(cur.fetchall())
        return [
            {"doc_id": str(row.get("doc_id", "")), "score": -float(row.get("distance") or 0.0)}
            for row in rows
        ]

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        candidates = max(int(limit), int(self.cfg.hybrid_candidates))
        lexical = self.search_lexical(query_text, candidates)
        vector = self.search_vector(query_vector, candidates)
        rrf_k = max(1, int(self.cfg.hybrid_rrf_k))
        alpha = min(1.0, max(0.0, float(self.cfg.hybrid_alpha)))
        scores: dict[str, float] = {}
        for rank, row in enumerate(lexical, start=1):
            doc_id = str(row.get("doc_id", ""))
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) / (rrf_k + rank)
        for rank, row in enumerate(vector, start=1):
            doc_id = str(row.get("doc_id", ""))
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0.0) + alpha / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(limit))]
        return [{"doc_id": doc_id, "score": score} for doc_id, score in ranked]
