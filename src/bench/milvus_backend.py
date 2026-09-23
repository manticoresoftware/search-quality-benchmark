from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
from typing import Any

from .config import MilvusConfig


_pymilvus_log_state = threading.local()
_pymilvus_log_filter_lock = threading.Lock()


class _PymilvusProbeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(_pymilvus_log_state, "probe_depth", 0))


_pymilvus_probe_log_filter = _PymilvusProbeLogFilter()


def _install_pymilvus_probe_log_filter() -> None:
    """Install a thread-aware filter on handlers owned by pymilvus."""
    with _pymilvus_log_filter_lock:
        loggers: list[logging.Logger] = [logging.getLogger("pymilvus")]
        loggers.extend(
            logger
            for name, logger in list(logging.Logger.manager.loggerDict.items())
            if name.startswith("pymilvus.") and isinstance(logger, logging.Logger)
        )
        for logger in loggers:
            for handler in list(logger.handlers):
                if _pymilvus_probe_log_filter not in handler.filters:
                    handler.addFilter(_pymilvus_probe_log_filter)


@contextmanager
def suppress_pymilvus_probe_logs() -> Iterator[None]:
    """Keep expected probe failures from writing tracebacks over a curses UI."""
    _install_pymilvus_probe_log_filter()
    previous_depth = int(getattr(_pymilvus_log_state, "probe_depth", 0))
    _pymilvus_log_state.probe_depth = previous_depth + 1
    try:
        yield
    finally:
        _pymilvus_log_state.probe_depth = previous_depth


class MilvusBackend:
    def __init__(self, cfg: MilvusConfig):
        try:
            from pymilvus import (
                AnnSearchRequest,
                DataType,
                Function,
                FunctionType,
                MilvusClient,
                RRFRanker,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without dependencies
            raise RuntimeError(
                "pymilvus is required for Milvus benchmarking. "
                "Install dependencies with `pip install -e .[dev]`."
            ) from exc

        # pymilvus configures its stderr handlers as part of the lazy import above.
        # Re-scan them here so the first quiet probe is filtered too.
        _install_pymilvus_probe_log_filter()

        self.cfg = cfg
        self.AnnSearchRequest = AnnSearchRequest
        self.DataType = DataType
        self.Function = Function
        self.FunctionType = FunctionType
        self.RRFRanker = RRFRanker

        kwargs: dict[str, Any] = {
            "uri": str(cfg.uri or "http://127.0.0.1:19530"),
            "db_name": str(cfg.database or "default"),
            "timeout": max(1, int(cfg.timeout_seconds)),
        }
        token = self._resolve_token()
        if token:
            kwargs["token"] = token
        self.client = MilvusClient(**kwargs)

    def _resolve_token(self) -> str:
        direct = str(self.cfg.token or "").strip()
        if direct:
            return direct
        path = str(self.cfg.token_file or "").strip()
        if not path:
            return ""
        return Path(path).read_text(encoding="utf-8").strip()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def connection_target(self) -> str:
        database = str(self.cfg.database or "default")
        return f"{self.cfg.uri} database={database}"

    def ensure_accessible(self) -> None:
        try:
            self.client.list_collections(timeout=max(1, int(self.cfg.timeout_seconds)))
        except Exception as exc:
            raise RuntimeError(
                f"Milvus is not reachable at {self.connection_target()}: {exc}"
            ) from exc

    def collection_name(self) -> str:
        return str(self.cfg.collection_name or "").strip()

    def collection_exists(self, collection_name: str | None = None) -> bool:
        return bool(
            self.client.has_collection(
                collection_name=collection_name or self.collection_name(),
                timeout=max(1, int(self.cfg.timeout_seconds)),
            )
        )

    def delete_collection_if_exists(self, collection_name: str | None = None) -> None:
        name = collection_name or self.collection_name()
        if self.collection_exists(name):
            self.client.drop_collection(
                collection_name=name,
                timeout=max(1, int(self.cfg.timeout_seconds)),
            )

    def collection_fields(self, collection_name: str | None = None) -> set[str]:
        name = collection_name or self.collection_name()
        if not self.collection_exists(name):
            return set()
        desc = self.client.describe_collection(
            collection_name=name,
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        out: set[str] = set()
        for field in list((desc or {}).get("fields", []) or []):
            if not isinstance(field, dict):
                continue
            field_name = str(field.get("name") or field.get("field_name") or "").strip()
            if field_name:
                out.add(field_name)
        return out

    def collection_supports(self, mode: str, collection_name: str | None = None) -> bool:
        fields = self.collection_fields(collection_name)
        if not fields:
            return False
        sparse = str(self.cfg.sparse_vector_field or "sparse")
        dense = str(self.cfg.dense_vector_field or "embedding")
        if mode == "fulltext":
            return sparse in fields
        if mode == "vector":
            return dense in fields
        if mode == "hybrid":
            return sparse in fields and dense in fields
        return False

    @staticmethod
    def _varchar_max_length(value: int, *, field: str) -> int:
        limit = int(value)
        if limit < 1 or limit > 65_535:
            raise ValueError(
                f"Milvus VARCHAR max_length for {field!r} must be between 1 and 65535; "
                f"got {limit}."
            )
        return limit

    def recreate_collection(self, *, dimension: int | None = None) -> None:
        name = self.collection_name()
        content_field = str(self.cfg.content_field or "content")
        sparse_field = str(self.cfg.sparse_vector_field or "sparse")
        dense_field = str(self.cfg.dense_vector_field or "embedding")
        doc_id_max_length = self._varchar_max_length(
            self.cfg.doc_id_max_length,
            field="doc_id",
        )
        self.delete_collection_if_exists(name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=self.DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            field_name="doc_id",
            datatype=self.DataType.VARCHAR,
            max_length=doc_id_max_length,
        )
        schema.add_field(
            field_name=content_field,
            datatype=self.DataType.TEXT,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name=sparse_field,
            datatype=self.DataType.SPARSE_FLOAT_VECTOR,
        )
        if dimension is not None:
            if int(dimension) <= 0:
                raise ValueError("Milvus dense vector dimension must be positive.")
            schema.add_field(
                field_name=dense_field,
                datatype=self.DataType.FLOAT_VECTOR,
                dim=int(dimension),
            )

        schema.add_function(
            self.Function(
                name=f"{content_field}_bm25",
                input_field_names=[content_field],
                output_field_names=[sparse_field],
                function_type=self.FunctionType.BM25,
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=sparse_field,
            index_name=f"{sparse_field}_bm25_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": str(
                    self.cfg.sparse_index_algorithm or "DAAT_MAXSCORE"
                ).upper(),
                "bm25_k1": float(self.cfg.bm25_k1),
                "bm25_b": float(self.cfg.bm25_b),
            },
        )
        if dimension is not None:
            index_params.add_index(
                field_name=dense_field,
                index_name=f"{dense_field}_dense_idx",
                index_type=str(self.cfg.dense_index_type or "AUTOINDEX").upper(),
                metric_type=str(self.cfg.dense_metric_type or "COSINE").upper(),
                params=dict(self.cfg.dense_index_params or {}),
            )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level=str(self.cfg.consistency_level or "Strong"),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )

    def _content_for_document(self, doc: dict[str, Any]) -> str:
        fields = list(self.cfg.text_fields or ["title", "abstract"])
        return " ".join(str(doc.get(field, "") or "").strip() for field in fields).strip()

    @staticmethod
    def _validate_varchar(value: str, *, max_bytes: int, field: str, doc_id: str) -> None:
        actual = len(value.encode("utf-8"))
        if actual > max_bytes:
            raise ValueError(
                f"Milvus VARCHAR field {field!r} exceeds max_length for doc_id={doc_id!r}: "
                f"{actual} bytes > {max_bytes} bytes"
            )

    def import_documents(self, docs: list[dict[str, Any]], *, with_vector: bool) -> int:
        if not docs:
            return 0
        content_field = str(self.cfg.content_field or "content")
        dense_field = str(self.cfg.dense_vector_field or "embedding")
        doc_id_max_length = self._varchar_max_length(
            self.cfg.doc_id_max_length,
            field="doc_id",
        )
        rows: list[dict[str, Any]] = []
        for doc in docs:
            doc_id = str(doc.get("doc_id", "") or "")
            content = self._content_for_document(doc)
            self._validate_varchar(
                doc_id,
                max_bytes=doc_id_max_length,
                field="doc_id",
                doc_id=doc_id,
            )
            row: dict[str, Any] = {
                "id": int(doc.get("bench_id", doc.get("id", 0)) or 0),
                "doc_id": doc_id,
                content_field: content,
            }
            if with_vector:
                vector = doc.get(dense_field)
                if not isinstance(vector, list) or not vector:
                    continue
                row[dense_field] = [float(x) for x in vector]
            rows.append(row)
        if not rows:
            return 0
        result = self.client.insert(
            collection_name=self.collection_name(),
            data=rows,
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        if isinstance(result, dict) and result.get("insert_count") is not None:
            return int(result["insert_count"])
        count = getattr(result, "insert_count", None)
        return int(count) if count is not None else len(rows)

    def finalize_collection(self) -> None:
        self.client.flush(
            collection_name=self.collection_name(),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        self.client.load_collection(
            collection_name=self.collection_name(),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )

    def get_document_count(self, collection_name: str | None = None) -> int:
        stats = self.client.get_collection_stats(
            collection_name=collection_name or self.collection_name(),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        return int((stats or {}).get("row_count", 0))

    def _dense_search_params(self) -> dict[str, Any]:
        return {
            "metric_type": str(self.cfg.dense_metric_type or "COSINE").upper(),
            "params": dict(self.cfg.dense_search_params or {}),
        }

    def _sparse_search_params(self) -> dict[str, Any]:
        return {
            "metric_type": "BM25",
            "params": {
                "drop_ratio_search": float(self.cfg.sparse_drop_ratio_search),
            },
        }

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        result = self.client.search(
            collection_name=self.collection_name(),
            data=[str(query_text or "")],
            anns_field=str(self.cfg.sparse_vector_field or "sparse"),
            limit=int(limit),
            output_fields=["doc_id"],
            search_params=self._sparse_search_params(),
            consistency_level=str(self.cfg.consistency_level or "Strong"),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        return self._extract_hits(result)

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        result = self.client.search(
            collection_name=self.collection_name(),
            data=[[float(x) for x in query_vector]],
            anns_field=str(self.cfg.dense_vector_field or "embedding"),
            limit=int(limit),
            output_fields=["doc_id"],
            search_params=self._dense_search_params(),
            consistency_level=str(self.cfg.consistency_level or "Strong"),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        return self._extract_hits(result)

    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        candidates = max(int(limit), int(self.cfg.hybrid_candidates))
        requests = [
            self.AnnSearchRequest(
                data=[str(query_text or "")],
                anns_field=str(self.cfg.sparse_vector_field or "sparse"),
                param=self._sparse_search_params(),
                limit=candidates,
            ),
            self.AnnSearchRequest(
                data=[[float(x) for x in query_vector]],
                anns_field=str(self.cfg.dense_vector_field or "embedding"),
                param=self._dense_search_params(),
                limit=candidates,
            ),
        ]
        result = self.client.hybrid_search(
            collection_name=self.collection_name(),
            reqs=requests,
            ranker=self.RRFRanker(int(self.cfg.hybrid_rrf_k)),
            limit=int(limit),
            output_fields=["doc_id"],
            consistency_level=str(self.cfg.consistency_level or "Strong"),
            timeout=max(1, int(self.cfg.timeout_seconds)),
        )
        return self._extract_hits(result)

    @staticmethod
    def _extract_hits(result: Any) -> list[dict[str, Any]]:
        groups = list(result or [])
        hits = list(groups[0] or []) if groups else []
        out: list[dict[str, Any]] = []
        for hit in hits:
            if isinstance(hit, dict):
                entity = hit.get("entity") or {}
                if not isinstance(entity, dict):
                    entity = {}
                doc_id = entity.get("doc_id", hit.get("doc_id", ""))
                score = hit.get("distance", hit.get("score", 0.0))
            else:
                entity = getattr(hit, "entity", None) or {}
                get = getattr(entity, "get", None)
                doc_id = get("doc_id", "") if callable(get) else getattr(entity, "doc_id", "")
                score = getattr(hit, "distance", getattr(hit, "score", 0.0))
            out.append({"doc_id": str(doc_id or ""), "score": float(score or 0.0)})
        return out
