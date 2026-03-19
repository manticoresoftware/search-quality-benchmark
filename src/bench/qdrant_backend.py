from __future__ import annotations

from time import perf_counter, sleep
from typing import Any

from .config import QdrantConfig

class QdrantBackend:
    def __init__(self, cfg: QdrantConfig):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qmodels
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qdrant-client is required for Qdrant benchmarking. Install dependencies with `pip install -e .[dev]`."
            ) from exc
        self.cfg = cfg
        self.models = qmodels
        if (cfg.path or "").strip():
            self.client = QdrantClient(path=cfg.path)
        else:
            self.client = QdrantClient(
                host=cfg.host,
                port=cfg.port,
                api_key=cfg.api_key or None,
                https=cfg.https,
                timeout=cfg.timeout_seconds,
            )

    def recreate_dense_collection(self, size: int) -> None:
        quantization_config = None
        if bool(self.cfg.dense_binary_quantization):
            quantization_config = self._build_dense_binary_quantization_config()
        self.client.recreate_collection(
            collection_name=self.cfg.dense_collection,
            vectors_config={
                self.cfg.dense_vector_name: self.models.VectorParams(size=size, distance=self.models.Distance.COSINE),
            },
            quantization_config=quantization_config,
        )

    def recreate_bm25_collection(self) -> None:
        sparse_params = {
            self.cfg.sparse_vector_name: self.models.SparseVectorParams(
                index=self.models.SparseIndexParams(on_disk=False),
                modifier=self.models.Modifier.IDF,
            )
        }
        self.client.recreate_collection(
            collection_name=self.cfg.bm25_collection,
            vectors_config={},
            sparse_vectors_config=sparse_params,
        )

    def upsert_dense_batch(self, batch: list[dict[str, Any]], wait: bool = False) -> int:
        points = [
            self.models.PointStruct(
                id=int(row["id"]),
                vector={self.cfg.dense_vector_name: row["vector"]},
                payload={
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                },
            )
            for row in batch
            if row.get("vector")
        ]
        if not points:
            return 0
        self._upsert_points_with_auto_split(
            collection_name=self.cfg.dense_collection,
            points=points,
            wait=wait,
        )
        return len(points)

    def upsert_bm25_text_batch(self, batch: list[dict[str, Any]]) -> int:
        if not batch:
            return 0
        points: list[Any] = []
        for row in batch:
            text = f"{row.get('title', '')} {row.get('abstract', '')}".strip()
            points.append(
                self.models.PointStruct(
                    id=int(row["id"]),
                    vector={
                        self.cfg.sparse_vector_name: self.models.Document(
                            text=text,
                            model="Qdrant/bm25",
                        )
                    },
                    payload={
                        "doc_id": row["doc_id"],
                        "title": row.get("title", ""),
                        "abstract": row.get("abstract", ""),
                        "doi": row.get("doi", ""),
                        "date": row.get("date", ""),
                    },
                )
            )
        if not points:
            return 0
        self._upsert_points_with_auto_split(
            collection_name=self.cfg.bm25_collection,
            points=points,
            wait=False,
        )
        return len(points)

    @staticmethod
    def _is_payload_too_large_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "payload error" in msg and "larger than allowed" in msg

    def _upsert_points_with_auto_split(self, collection_name: str, points: list[Any], wait: bool) -> None:
        # Handle Qdrant HTTP payload-size limits by progressively splitting oversized batches.
        stack: list[list[Any]] = [points]
        while stack:
            chunk = stack.pop()
            if not chunk:
                continue
            try:
                self.client.upsert(collection_name=collection_name, points=chunk, wait=wait)
            except Exception as exc:
                if self._is_payload_too_large_error(exc) and len(chunk) > 1:
                    mid = len(chunk) // 2
                    # Push right then left so left half is processed first.
                    stack.append(chunk[mid:])
                    stack.append(chunk[:mid])
                    continue
                raise

    def _count_points(self, collection_name: str) -> int:
        response = self.client.count(collection_name=collection_name, exact=True)
        if isinstance(response, dict):
            return int(response.get("count", 0))
        return int(getattr(response, "count", 0))

    def wait_for_points_count(
        self,
        collection_name: str,
        expected_count: int,
        timeout_seconds: int,
    ) -> int:
        if expected_count <= 0:
            return self._count_points(collection_name)
        started = perf_counter()
        current = self._count_points(collection_name)
        while current < expected_count and (perf_counter() - started) < timeout_seconds:
            sleep(0.25)
            current = self._count_points(collection_name)
        return current

    def query_dense(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        query_kwargs: dict[str, Any] = {}
        search_params = self._build_dense_search_params()
        if search_params is not None:
            query_kwargs["search_params"] = search_params
        rows = self.client.query_points(
            collection_name=self.cfg.dense_collection,
            query=vector,
            using=self.cfg.dense_vector_name,
            limit=limit,
            with_payload=True,
            **query_kwargs,
        ).points
        out: list[dict[str, Any]] = []
        for r in rows:
            payload = r.payload or {}
            out.append({"doc_id": str(payload.get("doc_id", "")), "score": float(r.score)})
        return out

    def _build_dense_binary_quantization_config(self) -> Any:
        try:
            BinaryQuantization = getattr(self.models, "BinaryQuantization")
            BinaryQuantizationConfig = getattr(self.models, "BinaryQuantizationConfig")
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError(
                "Installed qdrant-client does not support binary quantization models. "
                "Upgrade qdrant-client to a newer version."
            ) from exc

        binary_kwargs: dict[str, Any] = {"always_ram": bool(self.cfg.dense_binary_always_ram)}
        encoding = (self.cfg.dense_binary_query_encoding or "").strip().lower()
        if encoding and encoding != "default":
            enc_val: Any = encoding
            enc_enum = getattr(self.models, "BinaryQuantizationQueryEncoding", None)
            if enc_enum is not None:
                enc_attr = encoding.upper()
                if hasattr(enc_enum, enc_attr):
                    enc_val = getattr(enc_enum, enc_attr)
            binary_kwargs["query_encoding"] = enc_val

        return BinaryQuantization(binary=BinaryQuantizationConfig(**binary_kwargs))

    def _build_dense_search_params(self) -> Any | None:
        use_quant_params = bool(self.cfg.dense_binary_quantization) or bool(self.cfg.dense_search_quantization_ignore)
        if not use_quant_params:
            return None
        try:
            SearchParams = getattr(self.models, "SearchParams")
            QuantizationSearchParams = getattr(self.models, "QuantizationSearchParams")
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError(
                "Installed qdrant-client does not expose quantization search params. "
                "Upgrade qdrant-client to a newer version."
            ) from exc

        qsp_kwargs: dict[str, Any] = {
            "ignore": bool(self.cfg.dense_search_quantization_ignore),
        }
        if self.cfg.dense_search_quantization_rescore is not None:
            qsp_kwargs["rescore"] = bool(self.cfg.dense_search_quantization_rescore)
        if self.cfg.dense_search_quantization_oversampling is not None:
            qsp_kwargs["oversampling"] = float(self.cfg.dense_search_quantization_oversampling)
        return SearchParams(quantization=QuantizationSearchParams(**qsp_kwargs))

    def query_bm25_text(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        rows = self.client.query_points(
            collection_name=self.cfg.bm25_collection,
            query=self.models.Document(text=query_text, model="Qdrant/bm25"),
            using=self.cfg.sparse_vector_name,
            limit=limit,
            with_payload=True,
        ).points
        out: list[dict[str, Any]] = []
        for r in rows:
            payload = getattr(r, "payload", None) or {}
            out.append({"doc_id": str(payload.get("doc_id", "")), "score": float(getattr(r, "score", 0.0))})
        return out
