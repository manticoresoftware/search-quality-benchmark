from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .config import TypesenseConfig


class TypesenseBackend:
    def __init__(self, cfg: TypesenseConfig):
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
        headers = {
            "Content-Type": content_type,
            "X-TYPESENSE-API-KEY": self._resolve_api_key(),
        }
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any | None = None,
        raw_body: bytes | None = None,
        content_type: str = "application/json",
    ) -> Any:
        base = str(self.cfg.url or "http://127.0.0.1:8108").rstrip("/")
        query = f"?{urlparse.urlencode(params, doseq=True)}" if params else ""
        url = f"{base}{path}{query}"
        body: bytes | None = raw_body
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=body, method=method.upper(), headers=self._headers(content_type))
        try:
            with urlrequest.urlopen(req, timeout=max(int(self.cfg.timeout_sec), 30)) as resp:
                data = resp.read()
        except urlerror.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code == 404:
                raise
            raise RuntimeError(f"Typesense HTTP {exc.code}: {body_text}") from exc
        if not data:
            return {}
        text = data.decode("utf-8")
        if content_type == "text/plain":
            return text
        return json.loads(text)

    def connection_target(self) -> str:
        return str(self.cfg.url or "http://127.0.0.1:8108")

    def ensure_accessible(self) -> None:
        try:
            self._request("GET", "/health")
        except Exception as exc:
            raise RuntimeError(f"Typesense is not reachable at {self.connection_target()}: {exc}") from exc

    def collection_name(self) -> str:
        return str(self.cfg.collection_name or "").strip()

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self._request("GET", f"/collections/{urlparse.quote(collection_name, safe='')}")
            return True
        except urlerror.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def delete_collection_if_exists(self, collection_name: str) -> None:
        if not self.collection_exists(collection_name):
            return
        self._request("DELETE", f"/collections/{urlparse.quote(collection_name, safe='')}")

    def recreate_collection(self, *, dimension: int | None = None) -> None:
        name = self.collection_name()
        self.delete_collection_if_exists(name)
        fields: list[dict[str, Any]] = [
            {"name": "id", "type": "int64"},
            {"name": "doc_id", "type": "string"},
            {"name": "title", "type": "string", "optional": True},
            {"name": "abstract", "type": "string", "optional": True},
            {"name": "doi", "type": "string", "optional": True},
            {"name": "date", "type": "string", "optional": True},
        ]
        if dimension is not None:
            fields.append(
                {
                    "name": str(self.cfg.vector_field or "embedding"),
                    "type": "float[]",
                    "num_dim": int(dimension),
                    "vec_dist": str(self.cfg.vector_distance or "cosine"),
                    "optional": True,
                }
            )
        self._request(
            "POST",
            "/collections",
            payload={
                "name": name,
                "fields": fields,
                "default_sorting_field": "id",
            },
        )

    def import_documents(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        body = ("\n".join(json.dumps(doc, separators=(",", ":")) for doc in docs)).encode("utf-8")
        raw = self._request(
            "POST",
            f"/collections/{urlparse.quote(self.collection_name(), safe='')}/documents/import",
            params={"action": "upsert"},
            raw_body=body,
            content_type="text/plain",
        )
        inserted = 0
        for line in str(raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if bool(row.get("success")):
                inserted += 1
        return inserted

    def get_document_count(self, collection_name: str) -> int:
        resp = self._request("GET", f"/collections/{urlparse.quote(collection_name, safe='')}")
        return int(resp.get("num_documents", 0))

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        resp = self._request(
            "GET",
            f"/collections/{urlparse.quote(self.collection_name(), safe='')}/documents/search",
            params={
                "q": str(query_text or "*") or "*",
                "query_by": ",".join(self.cfg.text_fields or ["title", "abstract"]),
                "include_fields": "doc_id",
                "per_page": int(limit),
            },
        )
        return self._extract_hits(resp)

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        vector_literal = ",".join(format(float(x), ".8g") for x in query_vector)
        resp = self._request(
            "GET",
            f"/collections/{urlparse.quote(self.collection_name(), safe='')}/documents/search",
            params={
                "q": "*",
                "query_by": ",".join(self.cfg.text_fields or ["title", "abstract"]),
                "include_fields": "doc_id",
                "per_page": int(limit),
                "vector_query": f"{self.cfg.vector_field}:([{vector_literal}], k:{int(limit)})",
            },
        )
        return self._extract_hits(resp)

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        vector_literal = ",".join(format(float(x), ".8g") for x in query_vector)
        resp = self._request(
            "GET",
            f"/collections/{urlparse.quote(self.collection_name(), safe='')}/documents/search",
            params={
                "q": str(query_text or "*") or "*",
                "query_by": ",".join(self.cfg.text_fields or ["title", "abstract"]),
                "include_fields": "doc_id",
                "per_page": int(limit),
                "vector_query": (
                    f"{self.cfg.vector_field}:([{vector_literal}], k:{int(limit)}, "
                    f"alpha:{float(self.cfg.hybrid_alpha)})"
                ),
            },
        )
        return self._extract_hits(resp)

    @staticmethod
    def _extract_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        hits = list((result or {}).get("hits", []) or [])
        out: list[dict[str, Any]] = []
        for hit in hits:
            doc = hit.get("document") or {}
            out.append(
                {
                    "doc_id": str(doc.get("doc_id", "")),
                    "score": float(hit.get("text_match") or hit.get("vector_distance") or 0.0),
                }
            )
        return out
