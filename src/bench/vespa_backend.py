from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from zipfile import ZIP_DEFLATED, ZipFile

from .config import VespaConfig


class VespaBackend:
    def __init__(self, cfg: VespaConfig):
        self.cfg = cfg

    def connection_target(self) -> str:
        return str(self.cfg.url or "http://127.0.0.1:8081")

    def deploy_target(self) -> str:
        return str(self.cfg.deploy_url or "http://127.0.0.1:19071")

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any | bytes | None = None,
        content_type: str = "application/json",
    ) -> Any:
        base = str(base_url).rstrip("/")
        query = f"?{urlparse.urlencode(params, doseq=True)}" if params else ""
        url = f"{base}{path}{query}"
        body: bytes | None = None
        if isinstance(payload, bytes):
            body = payload
        elif payload is not None:
            body = json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {}
        if payload is not None:
            headers["Content-Type"] = content_type
        req = urlrequest.Request(url, data=body, method=method.upper(), headers=headers)
        try:
            with urlrequest.urlopen(req, timeout=max(int(self.cfg.timeout_sec), 30)) as resp:
                data = resp.read()
        except urlerror.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code == 404:
                raise
            raise RuntimeError(f"Vespa HTTP {exc.code}: {body_text}") from exc
        if not data:
            return {}
        text = data.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except Exception:
            return text

    def _query(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request("POST", self.connection_target(), "/search/", payload=payload)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected Vespa query response: {resp!r}")
        return resp

    def ensure_accessible(self) -> None:
        try:
            resp = self._request("GET", self.deploy_target(), "/state/v1/health")
            status = dict((resp or {}).get("status") or {})
            code = str(status.get("code") or "").strip().lower()
            if code not in {"up", "initializing"}:
                raise RuntimeError(f"Unexpected Vespa health status: {json.dumps(resp, ensure_ascii=True)}")
        except Exception as exc:
            raise RuntimeError(f"Vespa is not reachable at {self.connection_target()}: {exc}") from exc

    def collection_name(self) -> str:
        return str(self.cfg.collection_name or "").strip()

    def schema_name(self) -> str:
        raw = re.sub(r"[^a-z0-9_]+", "_", str(self.collection_name()).strip().lower())
        raw = re.sub(r"_+", "_", raw).strip("_")
        if not raw:
            raw = "trec_covid_vespa"
        if raw[0].isdigit():
            raw = f"s_{raw}"
        return raw

    def document_type(self) -> str:
        return self.schema_name()

    def collection_exists(self, collection_name: str) -> bool:
        return self._schema_file_path(self.schema_name()).exists()

    def delete_collection_if_exists(self, collection_name: str) -> None:
        schema = self.schema_name()
        path = self._schema_file_path(schema)
        if path.exists():
            path.unlink()
        registry = self._load_registry()
        registry.pop(schema, None)
        self._save_registry(registry)
        self._deploy_current_package()

    def recreate_collection(self, *, dimension: int | None = None) -> None:
        registry = self._load_registry()
        schema = self.schema_name()
        existing = dict(registry.get(schema) or {})
        dim = int(dimension or existing.get("dimension") or 1)
        registry[schema] = {
            "schema": schema,
            "dimension": dim,
            "text_fields": list(self.cfg.text_fields or ["title", "abstract"]),
            "vector_field": str(self.cfg.vector_field or "embedding"),
            "vector_distance": str(self.cfg.vector_distance or "angular"),
            "hybrid_alpha": float(self.cfg.hybrid_alpha),
        }
        self._save_registry(registry)
        self._write_schema_files(registry)
        self._deploy_current_package()

    def import_documents(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        inserted = 0
        namespace = str(self.cfg.namespace or "doc")
        doc_type = self.document_type()
        for doc in docs:
            doc_id = str(doc.get("doc_id", doc.get("id", ""))).strip()
            if not doc_id:
                continue
            fields = {
                "id": int(doc.get("bench_id", doc.get("id", 0)) or 0),
                "doc_id": doc_id,
                "title": str(doc.get("title", "")),
                "abstract": str(doc.get("abstract", "")),
                "doi": str(doc.get("doi", "")),
                "date": str(doc.get("date", "")),
            }
            vector = doc.get(self.cfg.vector_field)
            if isinstance(vector, list) and vector:
                fields[self.cfg.vector_field] = {"values": [float(x) for x in vector]}
            self._request(
                "POST",
                self.connection_target(),
                f"/document/v1/{namespace}/{doc_type}/docid/{urlparse.quote(doc_id, safe='')}",
                payload={"fields": fields},
            )
            inserted += 1
        return inserted

    def get_document_count(self, collection_name: str) -> int:
        resp = self._query(
            {
                "yql": f"select doc_id from sources {self.document_type()} where true;",
                "hits": 0,
                "ranking.profile": "bm25",
            }
        )
        root = dict(resp.get("root") or {})
        fields = dict(root.get("fields") or {})
        total = fields.get("totalCount")
        try:
            return int(total or 0)
        except Exception:
            return 0

    def search_lexical(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        resp = self._query(
            {
                "yql": f"select doc_id from sources {self.document_type()} where userQuery();",
                "query": str(query_text or ""),
                "hits": int(limit),
                "ranking.profile": "bm25",
            }
        )
        return self._extract_hits(resp)

    def search_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        resp = self._query(
            {
                "yql": (
                    f"select doc_id from sources {self.document_type()} "
                    f"where ({{targetHits:{int(limit)}}}nearestNeighbor({self.cfg.vector_field}, q));"
                ),
                "hits": int(limit),
                "ranking.profile": "vector",
                "input.query(q)": [float(x) for x in query_vector],
            }
        )
        return self._extract_hits(resp)

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        resp = self._query(
            {
                "yql": (
                    f"select doc_id from sources {self.document_type()} "
                    f"where userQuery() or ({{targetHits:{int(limit)}}}nearestNeighbor({self.cfg.vector_field}, q));"
                ),
                "query": str(query_text or ""),
                "hits": int(limit),
                "ranking.profile": "hybrid",
                "input.query(q)": [float(x) for x in query_vector],
            }
        )
        return self._extract_hits(resp)

    def _extract_hits(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        root = dict(result.get("root") or {})
        children = list(root.get("children") or [])
        out: list[dict[str, Any]] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            fields = dict(child.get("fields") or {})
            out.append(
                {
                    "doc_id": str(fields.get("doc_id", "")),
                    "score": float(child.get("relevance") or 0.0),
                }
            )
        return out

    def _package_dir(self) -> Path:
        return Path(str(self.cfg.package_dir or "data/vespa_app"))

    def _registry_path(self) -> Path:
        return self._package_dir() / "registry.json"

    def _schema_file_path(self, schema: str) -> Path:
        return self._package_dir() / "schemas" / f"{schema}.sd"

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        path = self._registry_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        out: dict[str, dict[str, Any]] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    out[str(key)] = value
        return out

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        pkg = self._package_dir()
        (pkg / "schemas").mkdir(parents=True, exist_ok=True)
        self._registry_path().write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")

    def _write_schema_files(self, registry: dict[str, dict[str, Any]]) -> None:
        pkg = self._package_dir()
        schemas_dir = pkg / "schemas"
        schemas_dir.mkdir(parents=True, exist_ok=True)
        for old in schemas_dir.glob("*.sd"):
            if old.stem not in registry:
                old.unlink()
        for schema, meta in registry.items():
            path = schemas_dir / f"{schema}.sd"
            path.write_text(self._render_schema(schema, meta), encoding="utf-8")
        (pkg / "services.xml").write_text(self._render_services_xml(registry), encoding="utf-8")

    def _render_services_xml(self, registry: dict[str, dict[str, Any]]) -> str:
        documents = "\n".join(
            f'      <document type="{schema}" mode="index" />'
            for schema in sorted(registry.keys())
        )
        return (
            '<services version="1.0">\n'
            '  <container id="default" version="1.0">\n'
            '    <search />\n'
            '    <document-api />\n'
            '    <document-processing />\n'
            '    <nodes count="1" />\n'
            '  </container>\n'
            '  <content id="content" version="1.0">\n'
            '    <redundancy>1</redundancy>\n'
            '    <documents>\n'
            f'{documents}\n'
            '    </documents>\n'
            '    <nodes count="1" />\n'
            '  </content>\n'
            '</services>\n'
        )

    def _render_schema(self, schema: str, meta: dict[str, Any]) -> str:
        dim = int(meta.get("dimension") or 1)
        text_fields = list(meta.get("text_fields") or ["title", "abstract"])
        vector_field = str(meta.get("vector_field") or "embedding")
        distance = str(meta.get("vector_distance") or "angular")
        alpha = float(meta.get("hybrid_alpha") or 0.5)
        text_expr = " + ".join(f"bm25({f})" for f in text_fields) or "0"
        return f"""schema {schema} {{
  document {schema} {{
    field id type int {{
      indexing: attribute | summary
    }}
    field doc_id type string {{
      indexing: attribute | summary
    }}
    field title type string {{
      indexing: index | summary
      index: enable-bm25
    }}
    field abstract type string {{
      indexing: index | summary
      index: enable-bm25
    }}
    field doi type string {{
      indexing: attribute | summary
    }}
    field date type string {{
      indexing: attribute | summary
    }}
    field {vector_field} type tensor<float>(x[{dim}]) {{
      indexing: attribute | index
      attribute {{
        distance-metric: {distance}
      }}
    }}
  }}
  fieldset default {{
    fields: {", ".join(text_fields)}
  }}
  rank-profile bm25 inherits default {{
    first-phase {{
      expression: {text_expr}
    }}
  }}
  rank-profile vector inherits default {{
    inputs {{
      query(q) tensor<float>(x[{dim}])
    }}
    first-phase {{
      expression: closeness(field, {vector_field})
    }}
  }}
  rank-profile hybrid inherits default {{
    inputs {{
      query(q) tensor<float>(x[{dim}])
    }}
    first-phase {{
      expression: {text_expr} + {alpha:.6f} * closeness(field, {vector_field})
    }}
  }}
}}
"""

    def _deploy_current_package(self) -> None:
        pkg = self._package_dir()
        (pkg / "schemas").mkdir(parents=True, exist_ok=True)
        if not (pkg / "services.xml").exists():
            self._write_schema_files(self._load_registry())
        archive = self._build_package_zip(pkg)
        try:
            self._request("PUT", self.deploy_target(), "/application/v2/tenant/default")
        except Exception:
            pass
        self._request(
            "POST",
            self.deploy_target(),
            "/application/v2/tenant/default/prepareandactivate",
            params={"applicationName": str(self.cfg.app_name or "bench")},
            payload=archive,
            content_type="application/zip",
        )

    def _build_package_zip(self, pkg: Path) -> bytes:
        buf = BytesIO()
        with ZipFile(buf, "w", compression=ZIP_DEFLATED) as zf:
            for path in sorted(pkg.rglob("*")):
                if path.is_dir():
                    continue
                rel = path.relative_to(pkg)
                zf.writestr(str(rel), path.read_bytes())
        return buf.getvalue()
