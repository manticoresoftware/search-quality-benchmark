from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import OpenSearchConfig


class OpenSearchBackend:
    def __init__(self, cfg: OpenSearchConfig):
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opensearch-py is required for OpenSearch benchmarking. Install dependencies with `pip install -e '.[dev]'`."
            ) from exc
        self.cfg = cfg
        parsed = urlparse(str(cfg.url or "http://127.0.0.1:9200"))
        auth = self._resolve_auth()
        self.client = OpenSearch(
            hosts=[
                {
                    "host": parsed.hostname or "127.0.0.1",
                    "port": parsed.port or (443 if parsed.scheme == "https" else 9200),
                    "scheme": parsed.scheme or "http",
                }
            ],
            http_auth=auth,
            verify_certs=bool(cfg.verify_certs),
            timeout=int(cfg.timeout_sec),
        )

    def _resolve_auth(self) -> Any:
        api_key = str(self.cfg.api_key or "").strip()
        if not api_key:
            path = str(self.cfg.api_key_file or "").strip()
            if path:
                api_key = Path(path).read_text(encoding="utf-8").strip()
        if api_key:
            return ("api_key", api_key)
        username = str(self.cfg.username or "").strip()
        password = str(self.cfg.password or "").strip()
        if username or password:
            return (username, password)
        return None

    def connection_target(self) -> str:
        return str(self.cfg.url)

    def ensure_accessible(self) -> None:
        try:
            self.client.info()
        except Exception as exc:
            raise RuntimeError(
                f"OpenSearch is not reachable at {self.connection_target()}: {exc}"
            ) from exc

    def lexical_index_name(self) -> str:
        return str(self.cfg.index_name or "").strip()

    def vector_index_name(self) -> str:
        return str(self.cfg.index_name or "").strip()

    def index_exists(self, index_name: str) -> bool:
        return bool(self.client.indices.exists(index=index_name))

    def delete_index_if_exists(self, index_name: str) -> None:
        if self.index_exists(index_name):
            self.client.indices.delete(index=index_name)

    def recreate_lexical_index(self) -> None:
        index_name = self.lexical_index_name()
        self.delete_index_if_exists(index_name)
        self.client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "long"},
                        "doc_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "abstract": {"type": "text"},
                        "doi": {"type": "keyword"},
                        "date": {"type": "keyword"},
                    }
                }
            },
        )

    def recreate_vector_index(self, dimension: int) -> None:
        index_name = self.vector_index_name()
        self.delete_index_if_exists(index_name)
        self.client.indices.create(
            index=index_name,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "id": {"type": "long"},
                        "doc_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "abstract": {"type": "text"},
                        "doi": {"type": "keyword"},
                        "date": {"type": "keyword"},
                        self.cfg.vector_field: {
                            "type": "knn_vector",
                            "dimension": int(dimension),
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": str(self.cfg.space_type or "cosinesimil"),
                                "parameters": {"ef_construction": 128, "m": 16},
                            },
                        },
                    }
                },
            },
        )

    def bulk_index(self, index_name: str, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        try:
            from opensearchpy import helpers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opensearch-py is required for OpenSearch benchmarking. Install dependencies with `pip install -e '.[dev]'`."
            ) from exc
        actions = [
            {
                "_index": index_name,
                "_id": int(doc["id"]),
                "_source": doc,
            }
            for doc in docs
        ]
        helpers.bulk(self.client, actions, refresh=False)
        return len(docs)

    def refresh(self, index_name: str) -> None:
        self.client.indices.refresh(index=index_name)

    def get_document_count(self, index_name: str) -> int:
        resp = self.client.count(index=index_name)
        return int(resp.get("count", 0))

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        resp = self.client.search(
            index=self.lexical_index_name(),
            body={
                "size": int(limit),
                "_source": ["doc_id"],
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": list(self.cfg.text_fields or ["title", "abstract"]),
                    }
                },
            },
        )
        return self._extract_hits(resp)

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        resp = self.client.search(
            index=self.vector_index_name(),
            body={
                "size": int(limit),
                "_source": ["doc_id"],
                "query": {
                    "knn": {
                        self.cfg.vector_field: {
                            "vector": [float(x) for x in query_vector],
                            "k": int(limit),
                        }
                    }
                },
            },
        )
        return self._extract_hits(resp)

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        resp = self.client.search(
            index=self.vector_index_name(),
            body={
                "size": int(limit),
                "_source": ["doc_id"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": list(self.cfg.text_fields or ["title", "abstract"]),
                                    "boost": float(self.cfg.hybrid_text_boost),
                                }
                            },
                            {
                                "knn": {
                                    self.cfg.vector_field: {
                                        "vector": [float(x) for x in query_vector],
                                        "k": int(limit),
                                        "boost": float(self.cfg.hybrid_vector_boost),
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
            },
        )
        return self._extract_hits(resp)

    @staticmethod
    def _extract_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        hits = list(((result or {}).get("hits") or {}).get("hits", []) or [])
        out: list[dict[str, Any]] = []
        for hit in hits:
            src = hit.get("_source") or {}
            out.append(
                {
                    "doc_id": str(src.get("doc_id", "")),
                    "score": float(hit.get("_score") or 0.0),
                }
            )
        return out
