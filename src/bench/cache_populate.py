from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import AppConfig
from .embedding_client import resolved_embedding_model
from .io_utils import read_jsonl
from .manticore import ManticoreClient
from .vector_cache import VectorCache, cache_key_from_doc, normalized_vector_text, parse_vector_from_fields


@dataclass
class CachePopulateResult:
    cache_path: str
    source_table: str
    seen_rows: int
    cached_ok: int
    cached_null_from_table: int
    cached_null_from_missing_docs: int


def populate_cache_from_manticore(
    cfg: AppConfig,
    source_table: str | None = None,
    model_name: str | None = None,
    similarity: str | None = None,
    vector_from: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CachePopulateResult:
    table = source_table or cfg.manticore.vector_table
    model = (model_name or resolved_embedding_model(cfg)).strip()
    sim = (similarity or cfg.manticore.vector_similarity).strip().lower()
    fields = parse_vector_from_fields(vector_from or cfg.manticore.vector_from)
    docs = read_jsonl(f"{cfg.dataset.prepared_dir}/docs.jsonl")

    cache = VectorCache(cfg.manticore.vector_cache_path)
    client = ManticoreClient(cfg.manticore)
    seen_keys: set[str] = set()
    seen_rows = 0
    cached_ok = 0
    cached_null_from_table = 0
    cached_null_from_missing_docs = 0
    ok_batch: list[dict[str, Any]] = []
    null_batch: list[dict[str, Any]] = []
    write_batch_size = max(100, int(cfg.manticore.batch_size))
    try:
        for row in client.iter_vectors(table, cfg.manticore.vector_field, batch_size=max(1, int(cfg.manticore.batch_size))):
            seen_rows += 1
            text_value = normalized_vector_text(row, fields)
            key = cache_key_from_doc(row, fields=fields, model=model, similarity=sim)
            seen_keys.add(key)
            vec = row.get("vector") or []
            if vec:
                ok_batch.append(
                    {
                        "cache_key": key,
                        "model": model,
                        "similarity": sim,
                        "text_value": text_value,
                        "vector": [float(x) for x in vec],
                    }
                )
                cached_ok += 1
            else:
                null_batch.append(
                    {
                        "cache_key": key,
                        "model": model,
                        "similarity": sim,
                        "text_value": text_value,
                    }
                )
                cached_null_from_table += 1
            if len(ok_batch) >= write_batch_size:
                cache.upsert_ok_many(ok_batch)
                ok_batch.clear()
            if len(null_batch) >= write_batch_size:
                cache.upsert_null_many(null_batch)
                null_batch.clear()
            if progress_callback is not None and (seen_rows == 1 or seen_rows % 1000 == 0):
                progress_callback(
                    {
                        "stage": "cache_sync",
                        "processed": seen_rows,
                        "total": len(docs),
                        "inserted": cached_ok,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        for d in docs:
            key = cache_key_from_doc(d, fields=fields, model=model, similarity=sim)
            if key in seen_keys:
                continue
            null_batch.append(
                {
                    "cache_key": key,
                    "model": model,
                    "similarity": sim,
                    "text_value": normalized_vector_text(d, fields),
                }
            )
            cached_null_from_missing_docs += 1
            if len(null_batch) >= write_batch_size:
                cache.upsert_null_many(null_batch)
                null_batch.clear()
            if progress_callback is not None and (cached_null_from_missing_docs == 1 or cached_null_from_missing_docs % 1000 == 0):
                progress_callback(
                    {
                        "stage": "cache_sync",
                        "processed": min(seen_rows + cached_null_from_missing_docs, len(docs)),
                        "total": len(docs),
                        "inserted": cached_ok,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        if ok_batch:
            cache.upsert_ok_many(ok_batch)
            ok_batch.clear()
        if null_batch:
            cache.upsert_null_many(null_batch)
            null_batch.clear()
        cache.commit()
    finally:
        client.close()
        cache.close()

    return CachePopulateResult(
        cache_path=cfg.manticore.vector_cache_path,
        source_table=table,
        seen_rows=seen_rows,
        cached_ok=cached_ok,
        cached_null_from_table=cached_null_from_table,
        cached_null_from_missing_docs=cached_null_from_missing_docs,
    )
