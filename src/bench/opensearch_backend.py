from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

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
        self._configured_hybrid_pipelines: set[str] = set()

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

    def _hybrid_weights(self) -> tuple[float, float]:
        text_weight = float(self.cfg.hybrid_text_boost)
        vector_weight = float(self.cfg.hybrid_vector_boost)
        if text_weight < 0 or vector_weight < 0 or text_weight + vector_weight <= 0:
            raise ValueError(
                "OpenSearch hybrid RRF weights must be non-negative and have a positive sum"
            )
        total = text_weight + vector_weight
        return text_weight / total, vector_weight / total

    def _hybrid_rrf_k(self) -> int:
        value = int(self.cfg.hybrid_rrf_k)
        if not 1 <= value <= 10000:
            raise ValueError("OpenSearch hybrid_rrf_k must be between 1 and 10000")
        return value

    def hybrid_pipeline_name(self) -> str:
        text_weight, vector_weight = self._hybrid_weights()
        signature = (
            f"{self.vector_index_name()}|{self._hybrid_rrf_k()}|"
            f"{text_weight:.12g}|{vector_weight:.12g}"
        )
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        return f"search-quality-benchmark-hybrid-rrf-{digest}"

    def ensure_hybrid_pipeline(self) -> str:
        pipeline_name = self.hybrid_pipeline_name()
        configured = getattr(self, "_configured_hybrid_pipelines", set())
        if pipeline_name in configured:
            return pipeline_name
        text_weight, vector_weight = self._hybrid_weights()
        self.client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{quote(pipeline_name, safe='')}",
            body={
                "description": "Search Bench native weighted RRF hybrid pipeline",
                "phase_results_processors": [
                    {
                        "score-ranker-processor": {
                            "combination": {
                                "technique": "rrf",
                                "rank_constant": self._hybrid_rrf_k(),
                                "parameters": {
                                    "weights": [text_weight, vector_weight]
                                },
                            }
                        }
                    }
                ],
            },
        )
        configured.add(pipeline_name)
        self._configured_hybrid_pipelines = configured
        return pipeline_name

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

    def search_hybrid(
        self, query_text: str, query_vector: list[float], limit: int
    ) -> list[dict[str, Any]]:
        pipeline_name = self.ensure_hybrid_pipeline()
        candidates = max(int(limit), int(self.cfg.hybrid_candidates))
        resp = self.client.search(
            index=self.vector_index_name(),
            params={"search_pipeline": pipeline_name},
            body={
                "size": int(limit),
                "_source": ["doc_id"],
                "query": {
                    "hybrid": {
                        "pagination_depth": candidates,
                        "queries": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": list(self.cfg.text_fields or ["title", "abstract"]),
                                }
                            },
                            {
                                "knn": {
                                    self.cfg.vector_field: {
                                        "vector": [float(x) for x in query_vector],
                                        "k": candidates,
                                    }
                                }
                            },
                        ]
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
