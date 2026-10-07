from __future__ import annotations

import re
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import SereneDBConfig


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOCUMENT_FIELDS = {"title", "abstract"}
_VECTOR_OPERATORS = {
    "cosine": "<=>",
    "l2": "<->",
    "l1": "<+>",
    "ip": "<#>",
}


def _identifier(value: str, *, label: str) -> str:
    name = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Invalid SereneDB {label}: {value!r}")
    return f'"{name}"'


class SereneDBBackend:
    """PostgreSQL-wire client for SereneDB full-text, vector and hybrid search."""

    def __init__(self, cfg: SereneDBConfig):
        self.cfg = cfg
        self._conn: psycopg.Connection[dict[str, Any]] | None = None

    def _connection(self) -> psycopg.Connection[dict[str, Any]]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                host=str(self.cfg.host or "127.0.0.1"),
                port=int(self.cfg.port),
                user=str(self.cfg.user or "postgres"),
                password=str(self.cfg.password or ""),
                dbname=str(self.cfg.database or "postgres"),
                connect_timeout=max(1, int(self.cfg.connect_timeout_seconds)),
                autocommit=True,
                row_factory=dict_row,
            )
            self._apply_search_settings(self._conn)
        return self._conn

    def _apply_search_settings(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        with conn.cursor() as cur:
            if int(self.cfg.vector_search_nprobe) > 0:
                cur.execute(
                    f"SET sdb_ivf_search_nprobe = {int(self.cfg.vector_search_nprobe)}"
                )
            if int(self.cfg.vector_rerank_factor) >= 0:
                cur.execute(
                    f"SET sdb_rerank_factor = {int(self.cfg.vector_rerank_factor)}"
                )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def connection_target(self) -> str:
        return f"{self.cfg.host}:{int(self.cfg.port)}/{self.cfg.database}"

    def ensure_accessible(self) -> None:
        try:
            with self._connection().cursor() as cur:
                cur.execute("SELECT version() AS version")
                cur.fetchone()
        except Exception as exc:
            raise RuntimeError(
                f"SereneDB is not reachable at {self.connection_target()}: {exc}"
            ) from exc

    def table_name(self) -> str:
        return str(self.cfg.table_name or "").strip()

    def index_name(self) -> str:
        return f"{self.table_name()}_idx"

    def dictionary_name(self) -> str:
        return f"{self.table_name()}_en"

    def _table_sql(self) -> str:
        return _identifier(self.table_name(), label="table name")

    def _index_sql(self) -> str:
        return _identifier(self.index_name(), label="index name")

    def _dictionary_sql(self) -> str:
        return _identifier(self.dictionary_name(), label="dictionary name")

    def _vector_field_sql(self) -> str:
        return _identifier(self.cfg.vector_field, label="vector field")

    def _text_fields(self) -> list[str]:
        fields = [str(value or "").strip() for value in self.cfg.text_fields]
        if not fields or any(field not in _DOCUMENT_FIELDS for field in fields):
            raise ValueError("SereneDB text_fields must contain title and/or abstract")
        return list(dict.fromkeys(fields))

    def _vector_metric(self) -> str:
        metric = str(self.cfg.vector_metric or "cosine").strip().lower()
        if metric not in _VECTOR_OPERATORS:
            raise ValueError("SereneDB vector_metric must be cosine, l2, l1, or ip")
        return metric

    def _vector_operator(self) -> str:
        return _VECTOR_OPERATORS[self._vector_metric()]

    def _vector_source_sql(self) -> str:
        mode = str(self.cfg.vector_query_mode or "exact").strip().lower()
        if mode == "exact":
            return self._table_sql()
        if mode == "index":
            return self._index_sql()
        raise ValueError("SereneDB vector_query_mode must be exact or index")

    def table_exists(self, table_name: str | None = None) -> bool:
        name = str(table_name or self.table_name())
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = %s LIMIT 1",
                (name,),
            )
            return cur.fetchone() is not None

    def index_exists(self) -> bool:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_indexes "
                "WHERE schemaname = current_schema() AND indexname = %s LIMIT 1",
                (self.index_name(),),
            )
            return cur.fetchone() is not None

    def table_has_fulltext(self, _table_name: str | None = None) -> bool:
        return self.index_exists()

    def table_has_vector(self, _table_name: str | None = None) -> bool:
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = current_schema() AND indexname = %s LIMIT 1",
                (self.index_name(),),
            )
            row = cur.fetchone()
        if not row:
            return False
        indexdef = str(row.get("indexdef") or "").lower()
        return self.cfg.vector_field.lower() in indexdef and any(
            marker in indexdef for marker in (" ivf", " hnsw")
        )

    def recreate_table(self, *, dimension: int | None = None) -> None:
        table = self._table_sql()
        index = self._index_sql()
        dictionary = self._dictionary_sql()
        with self._connection().cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {index}")
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(f"DROP TEXT SEARCH DICTIONARY IF EXISTS {dictionary}")
            columns = [
                '"id" BIGINT PRIMARY KEY',
                '"doc_id" VARCHAR NOT NULL',
                '"title" VARCHAR NOT NULL',
                '"abstract" VARCHAR NOT NULL',
                '"doi" VARCHAR NOT NULL',
                '"date" VARCHAR NOT NULL',
            ]
            if dimension is not None:
                if int(dimension) <= 0:
                    raise ValueError("SereneDB vector dimension must be positive")
                columns.append(
                    f"{self._vector_field_sql()} FLOAT[{int(dimension)}] NOT NULL"
                )
            cur.execute(f"CREATE TABLE {table} ({', '.join(columns)})")

    def create_search_index(self, *, with_vector: bool) -> None:
        table = self._table_sql()
        index = self._index_sql()
        dictionary = self._dictionary_sql()
        text_specs = [
            f"{_identifier(field, label='text field')} {dictionary}"
            for field in self._text_fields()
        ]
        specs = ['"id"', '"doc_id"', *text_specs]
        if with_vector:
            index_type = str(self.cfg.vector_index_type or "ivf").strip().lower()
            if index_type not in {"ivf", "hnsw"}:
                raise ValueError("SereneDB vector_index_type must be ivf or hnsw")
            vector_options = [f"metric = '{self._vector_metric()}'"]
            quantization = str(self.cfg.vector_quantization or "").strip().lower()
            if quantization:
                if quantization not in {"none", "sq8", "sq4", "pq", "rabitq", "tq"}:
                    raise ValueError("Unsupported SereneDB vector_quantization")
                vector_options.append(f"quant = '{quantization}'")
            specs.append(
                f"{self._vector_field_sql()} {index_type} ({', '.join(vector_options)})"
            )
        with self._connection().cursor() as cur:
            cur.execute(
                f"CREATE TEXT SEARCH DICTIONARY {dictionary} AS "
                "split_text(case := 'lower') "
                "| normalize_tokens('en_US.UTF-8', accent := false) "
                "| stem_words('en_US.UTF-8') "
                "WITH (frequency, position)"
            )
            cur.execute(
                f"CREATE INDEX {index} ON {table} USING inverted ({', '.join(specs)})"
            )
            cur.execute(f"VACUUM (REFRESH_TABLE) {table}")

    def bulk_index(self, docs: list[dict[str, Any]], *, with_vector: bool) -> int:
        if not docs:
            return 0
        columns = ['"id"', '"doc_id"', '"title"', '"abstract"', '"doi"', '"date"']
        placeholders = ["%s"] * len(columns)
        vector_dimension = 0
        if with_vector:
            vector_dimension = len(docs[0].get(self.cfg.vector_field, []))
            if vector_dimension <= 0:
                raise ValueError("SereneDB vector document is missing its embedding")
            columns.append(self._vector_field_sql())
            placeholders.append(f"%s::FLOAT[{vector_dimension}]")
        rows: list[tuple[Any, ...]] = []
        for doc in docs:
            values: list[Any] = [
                int(doc.get("id", 0)),
                str(doc.get("doc_id", "")),
                str(doc.get("title", "")),
                str(doc.get("abstract", "")),
                str(doc.get("doi", "")),
                str(doc.get("date", "")),
            ]
            if with_vector:
                vector = [float(value) for value in doc.get(self.cfg.vector_field, [])]
                if not vector:
                    raise ValueError("SereneDB vector document is missing its embedding")
                if len(vector) != vector_dimension:
                    raise ValueError("SereneDB vector dimensions are inconsistent")
                values.append(vector)
            rows.append(tuple(values))
        sql = (
            f"INSERT INTO {self._table_sql()} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)})"
        )
        with self._connection().cursor() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def get_document_count(self, table_name: str | None = None) -> int:
        table = _identifier(table_name or self.table_name(), label="table name")
        with self._connection().cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
            row = cur.fetchone() or {}
        return int(row.get("count") or 0)

    def _lexical_predicate(self) -> str:
        return " OR ".join(
            f"{_identifier(field, label='text field')} @@ %s"
            for field in self._text_fields()
        )

    def search_lexical(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        text = str(query or "").strip()
        if not text or int(limit) <= 0:
            return []
        predicate = self._lexical_predicate()
        sql = (
            f"SELECT doc_id, BM25({self._index_sql()}.tableoid) AS score "
            f"FROM {self._index_sql()} WHERE {predicate} "
            "ORDER BY score DESC, id LIMIT %s"
        )
        params = tuple([text] * len(self._text_fields()) + [int(limit)])
        with self._connection().cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_vector(self, vector: list[float], *, limit: int) -> list[dict[str, Any]]:
        values = [float(value) for value in vector]
        if not values or int(limit) <= 0:
            return []
        operator = self._vector_operator()
        sql = (
            f"SELECT doc_id, {self._vector_field_sql()} {operator} "
            f"%s::FLOAT[{len(values)}] AS distance FROM {self._vector_source_sql()} "
            "ORDER BY distance, id LIMIT %s"
        )
        with self._connection().cursor() as cur:
            cur.execute(sql, (values, int(limit)))
            return [dict(row) for row in cur.fetchall()]

    def search_hybrid(
        self,
        query: str,
        vector: list[float],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        text = str(query or "").strip()
        values = [float(value) for value in vector]
        if not text or not values or int(limit) <= 0:
            return []
        candidates = max(int(limit), int(self.cfg.hybrid_candidates))
        alpha = min(1.0, max(0.0, float(self.cfg.hybrid_alpha)))
        rrf_k = max(1, int(self.cfg.hybrid_rrf_k))
        predicate = self._lexical_predicate()
        operator = self._vector_operator()
        vector_source = self._vector_source_sql()
        sql = (
            "WITH fused AS ("
            "SELECT doc_id, %s::DOUBLE AS weight, "
            "RANK() OVER (ORDER BY score DESC) AS rank FROM ("
            f"SELECT doc_id, BM25({self._index_sql()}.tableoid) AS score "
            f"FROM {self._index_sql()} WHERE {predicate} "
            "ORDER BY score DESC LIMIT %s) lexical "
            "UNION ALL "
            "SELECT doc_id, %s::DOUBLE AS weight, "
            "RANK() OVER (ORDER BY distance) AS rank FROM ("
            f"SELECT doc_id, {self._vector_field_sql()} {operator} "
            f"%s::FLOAT[{len(values)}] AS distance FROM {vector_source} "
            "ORDER BY distance LIMIT %s) semantic) "
            "SELECT doc_id, SUM(weight / (%s + rank)) AS score FROM fused "
            "GROUP BY doc_id ORDER BY score DESC, doc_id LIMIT %s"
        )
        params: list[Any] = [1.0 - alpha]
        params.extend([text] * len(self._text_fields()))
        params.extend([candidates, alpha, values, candidates, rrf_k, int(limit)])
        with self._connection().cursor() as cur:
            cur.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]
