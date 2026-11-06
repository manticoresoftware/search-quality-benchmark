from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MeilisearchConfig


class MeilisearchBackend:
    def __init__(self, cfg: MeilisearchConfig):
        try:
            from meilisearch import Client
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "meilisearch is required for Meilisearch benchmarking. Install dependencies with `pip install -e '.[dev]'`."
            ) from exc
        self.cfg = cfg
        api_key = self._resolve_api_key()
        self.client = Client(cfg.host, api_key or None)

    def _resolve_api_key(self) -> str:
        direct = str(self.cfg.api_key or "").strip()
        if direct:
            return direct
        path = str(self.cfg.api_key_file or "").strip()
        if not path:
            return ""
        return Path(path).read_text(encoding="utf-8").strip()

    def connection_target(self) -> str:
        return str(self.cfg.host)

    def ensure_accessible(self) -> None:
        try:
            self.client.health()
        except Exception as exc:
            raise RuntimeError(
                f"Meilisearch is not reachable at {self.connection_target()}: {exc}"
            ) from exc

    def lexical_index_name(self) -> str:
        return str(self.cfg.index_name or "").strip()

    def vector_index_name(self) -> str:
        return str(self.cfg.index_name or "").strip()

    def index_exists(self, index_name: str) -> bool:
        try:
            self.client.get_index(index_name)
            return True
        except Exception:
            return False

    def delete_index_if_exists(self, index_name: str) -> None:
        if not self.index_exists(index_name):
            return
        task = self.client.delete_index(index_name)
        self.wait_for_task(task.task_uid)

    def wait_for_task(self, task_uid: int, *, timeout_ms: int | None = None) -> Any:
        return self.client.wait_for_task(
            task_uid,
            timeout_in_ms=timeout_ms if timeout_ms is not None else max(3_600_000, int(self.cfg.timeout_sec) * 1000),
            interval_in_ms=50,
        )

    def _base_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "searchableAttributes": list(self.cfg.searchable_attributes or ["title", "abstract"]),
            "rankingRules": list(
                self.cfg.ranking_rules or ["words", "typo", "proximity", "attribute", "sort", "exactness"]
            ),
            "stopWords": list(self.cfg.stop_words or []),
            "synonyms": dict(self.cfg.synonyms or {}),
            "typoTolerance": {"enabled": bool(self.cfg.typo_tolerance)},
        }
        return settings

    def recreate_lexical_index(self) -> None:
        index_name = self.lexical_index_name()
        self.delete_index_if_exists(index_name)
        task = self.client.create_index(index_name, {"primaryKey": "id"})
        self.wait_for_task(task.task_uid)
        index = self.client.index(index_name)
        task = index.update_settings(self._base_settings())
        self.wait_for_task(task.task_uid)

    def recreate_vector_index(self, dimension: int) -> None:
        index_name = self.vector_index_name()
        self.delete_index_if_exists(index_name)
        task = self.client.create_index(index_name, {"primaryKey": "id"})
        self.wait_for_task(task.task_uid)
        index = self.client.index(index_name)
        task = index.update_settings(self._base_settings())
        self.wait_for_task(task.task_uid)
        embedder: dict[str, Any] = {
            "source": "userProvided",
            "dimensions": int(dimension),
        }
        if bool(self.cfg.binary_quantized):
            embedder["binaryQuantized"] = True
        task = index.update_embedders({self.cfg.embedder_name: embedder})
        self.wait_for_task(task.task_uid)

    def add_documents(self, index_name: str, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        index = self.client.index(index_name)
        task = index.add_documents(docs, primary_key="id")
        self.wait_for_task(task.task_uid)
        return len(docs)

    def enqueue_documents(self, index_name: str, docs: list[dict[str, Any]]) -> int | None:
        if not docs:
            return None
        index = self.client.index(index_name)
        task = index.add_documents(docs, primary_key="id")
        return int(task.task_uid)

    def get_document_count(self, index_name: str) -> int:
        stats = self.client.index(index_name).get_stats()
        return int(getattr(stats, "number_of_documents", 0))

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        result = self.client.index(self.lexical_index_name()).search(
            query_text,
            {"limit": int(limit), "showRankingScore": True},
        )
        return self._extract_hits(result)

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        result = self.client.index(self.vector_index_name()).search(
            "",
            {
                "limit": int(limit),
                "vector": [float(x) for x in query_vector],
                "hybrid": {
                    "embedder": self.cfg.embedder_name,
                    "semanticRatio": 1.0,
                },
                "showRankingScore": True,
            },
        )
        return self._extract_hits(result)

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        result = self.client.index(self.vector_index_name()).search(
            query_text,
            {
                "limit": int(limit),
                "vector": [float(x) for x in query_vector],
                "hybrid": {
                    "embedder": self.cfg.embedder_name,
                    "semanticRatio": float(self.cfg.hybrid_semantic_ratio),
                },
                "showRankingScore": True,
            },
        )
        return self._extract_hits(result)

    @staticmethod
    def _extract_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        hits = list(result.get("hits", []) or [])
        out: list[dict[str, Any]] = []
        for hit in hits:
            out.append(
                {
                    "doc_id": str(hit.get("doc_id", "")),
                    "score": float(hit.get("_rankingScore") or hit.get("_semanticScore") or 0.0),
                }
            )
        return out
