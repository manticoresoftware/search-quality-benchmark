from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
import uuid

from .config import WeaviateConfig


class WeaviateBackend:
    def __init__(self, cfg: WeaviateConfig):
        self.cfg = cfg

    def _resolve_api_key(self) -> str:
        direct = str(self.cfg.api_key or "").strip()
        if direct:
            return direct
        path = str(self.cfg.api_key_file or "").strip()
        if not path:
            return ""
        return Path(path).read_text(encoding="utf-8").strip()

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        headers = {"Content-Type": content_type}
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any | None = None,
    ) -> Any:
        base = str(self.cfg.url or "http://127.0.0.1:8080").rstrip("/")
        query = f"?{urlparse.urlencode(params, doseq=True)}" if params else ""
        url = f"{base}{path}{query}"
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=body, method=method.upper(), headers=self._headers())
        try:
            with urlrequest.urlopen(req, timeout=max(int(self.cfg.timeout_sec), 30)) as resp:
                data = resp.read()
        except urlerror.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code == 404:
                raise
            raise RuntimeError(f"Weaviate HTTP {exc.code}: {body_text}") from exc
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def _graphql(self, query: str) -> dict[str, Any]:
        resp = self._request("POST", "/v1/graphql", payload={"query": query})
        errs = list((resp or {}).get("errors", []) or [])
        if errs:
            raise RuntimeError(f"Weaviate GraphQL error: {json.dumps(errs, ensure_ascii=True)}")
        return dict((resp or {}).get("data") or {})

    def connection_target(self) -> str:
        return str(self.cfg.url or "http://127.0.0.1:8080")

    def ensure_accessible(self) -> None:
        try:
            self._request("GET", "/v1/meta")
        except Exception as exc:
            raise RuntimeError(f"Weaviate is not reachable at {self.connection_target()}: {exc}") from exc

    def collection_name(self) -> str:
        return str(self.cfg.collection_name or "").strip()

    def class_name(self) -> str:
        raw = re.sub(r"[^A-Za-z0-9]+", " ", self.collection_name()).strip()
        if not raw:
            return "BenchWeaviate"
        parts = [p.capitalize() for p in raw.split() if p]
        name = "".join(parts) or "BenchWeaviate"
        if name[0].isdigit():
            name = f"Bench{name}"
        return name

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self._request("GET", f"/v1/schema/{urlparse.quote(self.class_name(), safe='')}")
            return True
        except urlerror.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def delete_collection_if_exists(self, collection_name: str) -> None:
        if not self.collection_exists(collection_name):
            return
        self._request("DELETE", f"/v1/schema/{urlparse.quote(self.class_name(), safe='')}")

    def recreate_collection(self, *, dimension: int | None = None) -> None:
        self.delete_collection_if_exists(self.collection_name())
        props = [
            {"name": "doc_id", "dataType": ["text"]},
            {"name": "bench_id", "dataType": ["int"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "abstract", "dataType": ["text"]},
            {"name": "doi", "dataType": ["text"]},
            {"name": "date", "dataType": ["text"]},
        ]
        schema: dict[str, Any] = {
            "class": self.class_name(),
            "vectorizer": "none",
            "properties": props,
            "vectorIndexType": "hnsw",
        }
        if dimension is not None:
            schema["vectorIndexConfig"] = {
                "distance": str(self.cfg.vector_distance or "cosine"),
            }
        self._request("POST", "/v1/schema", payload=schema)

    def import_documents(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        objects: list[dict[str, Any]] = []
        for doc in docs:
            props = {
                "doc_id": str(doc.get("doc_id", "")),
                "bench_id": int(doc.get("bench_id", doc.get("id", 0)) or 0),
                "title": str(doc.get("title", "")),
                "abstract": str(doc.get("abstract", "")),
                "doi": str(doc.get("doi", "")),
                "date": str(doc.get("date", "")),
            }
            obj: dict[str, Any] = {
                "class": self.class_name(),
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, props["doc_id"] or str(props["bench_id"]))),
                "properties": props,
            }
            vector = doc.get(self.cfg.vector_field)
            if isinstance(vector, list) and vector:
                obj["vector"] = [float(x) for x in vector]
            objects.append(obj)
        resp = self._request("POST", "/v1/batch/objects", payload={"objects": objects})
        errs = (((resp or {}).get("errors") or {}).get("error") or [])
        if errs:
            raise RuntimeError(f"Weaviate batch import error: {json.dumps(errs, ensure_ascii=True)}")
        return len(docs)

    def get_document_count(self, collection_name: str) -> int:
        query = f'{{Aggregate{{{self.class_name()}{{meta{{count}}}}}}}}'
        data = self._graphql(query)
        rows = list(((data.get("Aggregate") or {}).get(self.class_name()) or []) or [])
        if not rows:
            return 0
        meta = rows[0].get("meta") or {}
        return int(meta.get("count", 0))

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        props = json.dumps(list(self.cfg.text_fields or ["title", "abstract"]))
        query = (
            f'{{Get{{{self.class_name()}('
            f'limit:{int(limit)},'
            f'bm25:{{query:{json.dumps(str(query_text or ""))},properties:{props}}}'
            f'){{doc_id _additional{{score}}}}}}}}'
        )
        return self._extract_hits(self._graphql(query))

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        vec = json.dumps([float(x) for x in query_vector], separators=(",", ":"))
        query = (
            f'{{Get{{{self.class_name()}('
            f'limit:{int(limit)},'
            f'nearVector:{{vector:{vec}}}'
            f'){{doc_id _additional{{distance}}}}}}}}'
        )
        return self._extract_hits(self._graphql(query))

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        vec = json.dumps([float(x) for x in query_vector], separators=(",", ":"))
        props = json.dumps(list(self.cfg.text_fields or ["title", "abstract"]))
        query = (
            f'{{Get{{{self.class_name()}('
            f'limit:{int(limit)},'
            f'hybrid:{{query:{json.dumps(str(query_text or ""))},alpha:{float(self.cfg.hybrid_alpha):.6g},vector:{vec},properties:{props}}}'
            f'){{doc_id _additional{{score distance}}}}}}}}'
        )
        return self._extract_hits(self._graphql(query))

    def _extract_hits(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        rows = list(((data.get("Get") or {}).get(self.class_name()) or []) or [])
        out: list[dict[str, Any]] = []
        for row in rows:
            extra = row.get("_additional") or {}
            score = extra.get("score")
            if score is None:
                dist = extra.get("distance")
                score = 0.0 if dist is None else -float(dist)
            out.append(
                {
                    "doc_id": str(row.get("doc_id", "")),
                    "score": float(score or 0.0),
                }
            )
        return out
