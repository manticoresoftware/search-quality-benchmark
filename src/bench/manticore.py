from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import time
from typing import Any, Callable, Iterable

import pymysql
from pymysql.cursors import DictCursor
import requests

from .config import ManticoreConfig


@dataclass
class IngestStats:
    rows_inserted: int = 0
    failed_batches: int = 0
    retried_batches: int = 0
    last_error: str = ""
    retry_reasons: dict[str, int] = field(default_factory=dict)

    def note_error(self, exc: Exception) -> None:
        msg = str(exc).strip().replace("\n", " ")
        if not msg:
            msg = exc.__class__.__name__
        reason = f"{exc.__class__.__name__}: {msg}"
        if len(reason) > 220:
            reason = reason[:219] + "…"
        self.last_error = reason
        self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1


class ManticoreClient:
    def __init__(self, cfg: ManticoreConfig):
        self.cfg = cfg
        self.conn = pymysql.connect(
            host=cfg.host,
            port=cfg.sql_port,
            user=cfg.user,
            password=cfg.password,
            database="",
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=cfg.connect_timeout_seconds,
            read_timeout=cfg.query_timeout_seconds,
            write_timeout=cfg.query_timeout_seconds,
            cursorclass=DictCursor,
        )
        with self.conn.cursor() as cur:
            cur.execute("SET profiling=1")

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> int:
        sql = sql.strip()
        with self.conn.cursor() as cur:
            if params is None:
                affected = cur.execute(sql)
            else:
                affected = cur.execute(sql, params)
        return affected

    def query(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        sql = sql.strip()
        with self.conn.cursor() as cur:
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql, params)
            rows = list(cur.fetchall())
        return rows

    def show_profile(self) -> list[dict[str, Any]]:
        return self.query("SHOW PROFILE")

    def show_plan(self) -> list[dict[str, Any]]:
        return self.query("SHOW PLAN")

    def drop_table_if_exists(self, table_name: str) -> None:
        self.execute(f"DROP TABLE IF EXISTS {table_name}")

    def table_exists(self, table_name: str) -> bool:
        safe_name = table_name.replace("'", "''")
        rows = self.query(f"SHOW TABLES LIKE '{safe_name}'")
        return len(rows) > 0

    def create_fulltext_table(
        self,
        table_name: str,
        fulltext_options: str = "morphology='stem_en' index_field_lengths='1'",
    ) -> None:
        opts = (fulltext_options or "").strip()
        options_clause = f" {opts}" if opts else ""
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGINT,
                doc_id STRING,
                title TEXT,
                abstract TEXT,
                doi STRING,
                date STRING
            ){options_clause}
            """
        )

    def create_vector_table(
        self,
        table_name: str,
        vector_field: str,
        model_name: str = "",
        vector_from: str = "",
        knn_dims: int = 0,
        hnsw_similarity: str = "cosine",
        quantization: str = "",
        api_key: str = "",
        api_key_file: str = "",
    ) -> None:
        safe_model = model_name.replace("'", "''")
        safe_from = vector_from.replace("'", "''")
        safe_similarity = (hnsw_similarity or "cosine").replace("'", "''")
        safe_quantization = (quantization or "").strip().lower().replace("'", "''")
        resolved_api_key = (api_key or "").strip()
        if not resolved_api_key and (api_key_file or "").strip():
            resolved_api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
        safe_api_key = resolved_api_key.replace("'", "''")
        api_key_clause = f"\n                    API_KEY='{safe_api_key}'" if safe_api_key else ""
        quantization_clause = f"\n                    quantization='{safe_quantization}'" if safe_quantization else ""
        if safe_model:
            vector_opts = (
                f"KNN_TYPE='hnsw'\n"
                f"                    HNSW_SIMILARITY='{safe_similarity}'\n"
                f"                    MODEL_NAME='{safe_model}'\n"
                f"                    FROM='{safe_from}'\n"
                f"                    {quantization_clause}\n"
                f"                    {api_key_clause}"
            )
        else:
            if int(knn_dims) <= 0:
                raise ValueError("knn_dims must be > 0 when model_name is not provided")
            vector_opts = (
                f"KNN_TYPE='hnsw'\n"
                f"                    KNN_DIMS='{int(knn_dims)}'\n"
                f"                    HNSW_SIMILARITY='{safe_similarity}'\n"
                f"                    {quantization_clause}"
            )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGINT,
                doc_id STRING,
                title TEXT,
                abstract TEXT,
                doi STRING,
                date STRING,
                {vector_field} FLOAT_VECTOR
                    {vector_opts}
            )
            """
        )

    def bulk_insert_docs(
        self,
        table_name: str,
        docs: Iterable[dict[str, Any]],
        batch_size: int,
        retries: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_meta: dict[str, Any] | None = None,
    ) -> IngestStats:
        stats = IngestStats()
        batch: list[dict[str, Any]] = []
        processed = 0
        total = len(docs) if hasattr(docs, "__len__") else None
        sql = f"""
            INSERT INTO {table_name} (id, doc_id, title, abstract, doi, date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_size:
                inserted = self._flush_batch(sql, batch, retries, stats)
                processed += len(batch)
                if progress_callback is not None:
                    payload = {
                        "table": table_name,
                        "processed": processed,
                        "total": total,
                        "inserted": stats.rows_inserted,
                        "last_batch_inserted": inserted,
                        "failed_batches": stats.failed_batches,
                        "retried_batches": stats.retried_batches,
                        "last_error": stats.last_error,
                    }
                    if progress_meta:
                        payload.update(progress_meta)
                    progress_callback(payload)
                batch = []
        if batch:
            inserted = self._flush_batch(sql, batch, retries, stats)
            processed += len(batch)
            if progress_callback is not None:
                payload = {
                    "table": table_name,
                    "processed": processed,
                    "total": total,
                    "inserted": stats.rows_inserted,
                    "last_batch_inserted": inserted,
                    "failed_batches": stats.failed_batches,
                    "retried_batches": stats.retried_batches,
                    "last_error": stats.last_error,
                }
                if progress_meta:
                    payload.update(progress_meta)
                progress_callback(payload)
        return stats

    def _flush_batch(
        self,
        sql: str,
        batch: list[dict[str, Any]],
        retries: int,
        stats: IngestStats,
    ) -> int:
        values = [
            (
                d["id"],
                d["doc_id"],
                d.get("title", ""),
                d.get("abstract", ""),
                d.get("doi", ""),
                d.get("date", ""),
            )
            for d in batch
        ]
        attempt = 0
        while True:
            try:
                with self.conn.cursor() as cur:
                    cur.executemany(sql, values)
                stats.rows_inserted += len(values)
                return len(values)
            except Exception as exc:
                stats.note_error(exc)
                attempt += 1
                if attempt > retries:
                    stats.failed_batches += 1
                    return 0
                stats.retried_batches += 1

    def bulk_insert_docs_with_vectors(
        self,
        table_name: str,
        vector_field: str,
        docs: Iterable[dict[str, Any]],
        batch_size: int,
        retries: int,
        minimal_fields: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_meta: dict[str, Any] | None = None,
    ) -> IngestStats:
        stats = IngestStats()
        batch: list[dict[str, Any]] = []
        processed = 0
        total = len(docs) if hasattr(docs, "__len__") else None
        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_size:
                inserted = self._flush_vector_batch(
                    table_name, vector_field, batch, retries, stats, minimal_fields=minimal_fields
                )
                processed += len(batch)
                if progress_callback is not None:
                    payload = {
                        "table": table_name,
                        "processed": processed,
                        "total": total,
                        "inserted": stats.rows_inserted,
                        "last_batch_inserted": inserted,
                        "failed_batches": stats.failed_batches,
                        "retried_batches": stats.retried_batches,
                        "last_error": stats.last_error,
                    }
                    if progress_meta:
                        payload.update(progress_meta)
                    progress_callback(payload)
                batch = []
        if batch:
            inserted = self._flush_vector_batch(
                table_name, vector_field, batch, retries, stats, minimal_fields=minimal_fields
            )
            processed += len(batch)
            if progress_callback is not None:
                payload = {
                    "table": table_name,
                    "processed": processed,
                    "total": total,
                    "inserted": stats.rows_inserted,
                    "last_batch_inserted": inserted,
                    "failed_batches": stats.failed_batches,
                    "retried_batches": stats.retried_batches,
                    "last_error": stats.last_error,
                }
                if progress_meta:
                    payload.update(progress_meta)
                progress_callback(payload)
        return stats

    def _flush_vector_batch(
        self,
        table_name: str,
        vector_field: str,
        batch: list[dict[str, Any]],
        retries: int,
        stats: IngestStats,
        minimal_fields: bool = False,
    ) -> int:
        attempt = 0
        while True:
            try:
                values_sql: list[str] = []
                for d in batch:
                    vec = [float(x) for x in (d.get("vector") or [])]
                    if not vec:
                        continue
                    vec_lit = "(" + ",".join(f"{x:.8f}" for x in vec) + ")"
                    if minimal_fields:
                        values_sql.append(
                            "("
                            f"{int(d['id'])},"
                            f"{self.conn.escape(str(d.get('doc_id', '')))},"
                            f"{vec_lit}"
                            ")"
                        )
                    else:
                        values_sql.append(
                            "("
                            f"{int(d['id'])},"
                            f"{self.conn.escape(str(d.get('doc_id', '')))},"
                            f"{self.conn.escape(str(d.get('title', '') or ''))},"
                            f"{self.conn.escape(str(d.get('abstract', '') or ''))},"
                            f"{self.conn.escape(str(d.get('doi', '') or ''))},"
                            f"{self.conn.escape(str(d.get('date', '') or ''))},"
                            f"{vec_lit}"
                            ")"
                        )
                if not values_sql:
                    return 0
                if minimal_fields:
                    sql = (
                        f"INSERT INTO {table_name} (id, doc_id, {vector_field}) VALUES "
                        + ", ".join(values_sql)
                    )
                else:
                    sql = (
                        f"INSERT INTO {table_name} (id, doc_id, title, abstract, doi, date, {vector_field}) VALUES "
                        + ", ".join(values_sql)
                    )
                with self.conn.cursor() as cur:
                    cur.execute(sql)
                stats.rows_inserted += len(values_sql)
                return len(values_sql)
            except Exception as exc:
                stats.note_error(exc)
                attempt += 1
                if attempt > retries:
                    stats.failed_batches += 1
                    return 0
                stats.retried_batches += 1

    @staticmethod
    def parse_float_vector(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")
        raw = str(value).strip()
        if not raw:
            return []
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1]
        if not raw:
            return []
        out: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out

    def count_rows(self, table_name: str) -> int:
        rows = self.query(f"SELECT COUNT(*) AS c FROM {table_name}")
        if not rows:
            return 0
        return int(rows[0]["c"])

    def iter_vectors(
        self,
        table_name: str,
        vector_field: str,
        batch_size: int = 500,
    ) -> Iterable[dict[str, Any]]:
        total_rows = self.count_rows(table_name)
        offset = 0
        base_url = f"http://{self.cfg.host}:{self.cfg.http_port}/search"
        while True:
            payload = {
                "table": table_name,
                "query": {"match_all": {}},
                "_source": ["doc_id", "title", "abstract", "doi", "date", vector_field],
                "limit": batch_size,
                "offset": offset,
                "sort": [{"id": "asc"}],
                "options": {"max_matches": max(1000, total_rows + 100)},
            }
            resp = requests.post(base_url, json=payload, timeout=max(self.cfg.query_timeout_seconds, 30))
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                src = hit.get("_source", {})
                yield {
                    "id": int(hit.get("_id")),
                    "doc_id": str(src.get("doc_id", "")),
                    "title": src.get("title", ""),
                    "abstract": src.get("abstract", ""),
                    "doi": src.get("doi", ""),
                    "date": src.get("date", ""),
                    "vector": self.parse_float_vector(src.get(vector_field)),
                }
            offset += len(hits)

    def embed_texts(
        self,
        texts: list[str],
        vector_field: str,
        model_name: str,
        vector_from: str,
        api_key: str = "",
        api_key_file: str = "",
    ) -> list[list[float]]:
        if not texts:
            return []
        tmp_table = f"tmp_embed_{os.getpid()}_{int(time() * 1000)}"
        self.create_vector_table(
            tmp_table,
            vector_field=vector_field,
            model_name=model_name,
            vector_from=vector_from,
            api_key=api_key,
            api_key_file=api_key_file,
        )
        docs = [
            {
                "id": idx + 1,
                "doc_id": f"q{idx + 1}",
                "title": text,
                "abstract": "",
                "doi": "",
                "date": "",
            }
            for idx, text in enumerate(texts)
        ]
        try:
            self.bulk_insert_docs(tmp_table, docs, batch_size=min(100, len(docs)), retries=1)
            payload = {
                "table": tmp_table,
                "query": {"match_all": {}},
                "_source": [vector_field],
                "limit": len(texts),
                "offset": 0,
                "sort": [{"id": "asc"}],
            }
            base_url = f"http://{self.cfg.host}:{self.cfg.http_port}/search"
            resp = requests.post(base_url, json=payload, timeout=max(self.cfg.query_timeout_seconds, 30))
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            vectors_by_id = {
                int(h.get("_id")): self.parse_float_vector((h.get("_source") or {}).get(vector_field))
                for h in hits
            }
            return [vectors_by_id.get(i + 1, []) for i in range(len(texts))]
        finally:
            self.drop_table_if_exists(tmp_table)
