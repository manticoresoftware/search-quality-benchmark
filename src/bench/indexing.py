from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .config import AppConfig
from .cache_populate import populate_cache_from_manticore
from .io_utils import read_jsonl
from .manticore import ManticoreClient
from .meilisearch_backend import MeilisearchBackend
from .opensearch_backend import OpenSearchBackend
from .qdrant_backend import QdrantBackend
from .vector_cache import VectorCache, cache_key_from_doc, parse_vector_from_fields


@dataclass
class ModeIndexResult:
    mode: str
    duration_seconds: float
    rows_inserted: int
    docs_per_second: float
    failed_batches: int
    retried_batches: int
    last_error: str = ""
    error_reasons: dict[str, int] = field(default_factory=dict)


def _run_index_manticore(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    docs = read_jsonl(docs_path)

    client = ManticoreClient(cfg.manticore)
    out: list[ModeIndexResult] = []
    try:
        if mode == "fulltext":
            client.drop_table_if_exists(cfg.manticore.fulltext_table)
            client.create_fulltext_table(
                cfg.manticore.fulltext_table,
                fulltext_options=cfg.manticore.fulltext_options,
            )
            started = perf_counter()
            ingest_stats = client.bulk_insert_docs(
                cfg.manticore.fulltext_table,
                docs,
                batch_size=cfg.manticore.batch_size,
                retries=cfg.manticore.insert_retries,
                progress_callback=progress_callback,
                progress_meta={"engine": "manticore", "mode": "fulltext", "stage": "ingest"},
            )
            elapsed = perf_counter() - started
            out.append(
                ModeIndexResult(
                    mode="fulltext",
                    duration_seconds=elapsed,
                    rows_inserted=ingest_stats.rows_inserted,
                    docs_per_second=(ingest_stats.rows_inserted / elapsed) if elapsed > 0 else 0.0,
                    failed_batches=ingest_stats.failed_batches,
                    retried_batches=ingest_stats.retried_batches,
                    last_error=ingest_stats.last_error,
                    error_reasons=dict(ingest_stats.retry_reasons),
                )
            )

        if mode in ("vector", "hybrid"):
            fields = parse_vector_from_fields(cfg.manticore.vector_from)
            cache = VectorCache(cfg.manticore.vector_cache_path)
            docs_cached: list[tuple[dict[str, Any], str]] = []
            docs_for_auto_embed: list[dict[str, Any]] = []
            skipped_null = 0
            try:
                status_map = cache.get_status_map(
                    model=cfg.manticore.vector_model,
                    similarity=cfg.manticore.vector_similarity,
                )
                for i, d in enumerate(docs, start=1):
                    key = cache_key_from_doc(
                        d,
                        fields=fields,
                        model=cfg.manticore.vector_model,
                        similarity=cfg.manticore.vector_similarity,
                    )
                    st = status_map.get(key, "missing")
                    if st == "missing":
                        docs_for_auto_embed.append(d)
                        continue
                    if st == "null":
                        skipped_null += 1
                        # Known non-embeddable; do not retry.
                        continue
                    docs_cached.append((d, key))
                    if progress_callback is not None and (i == 1 or i % 1000 == 0 or i == len(docs)):
                        progress_callback(
                            {
                                "engine": "manticore",
                                "mode": mode,
                                "stage": "cache_lookup",
                                "processed": i,
                                "total": len(docs),
                                "inserted": len(docs_cached),
                                "failed_batches": 0,
                                "retried_batches": 0,
                            }
                        )
            finally:
                cache.close()

            if not docs_cached and not docs_for_auto_embed:
                raise RuntimeError("No vector documents available (all entries are marked null in cache).")

            target_table = cfg.manticore.vector_table
            if mode == "hybrid":
                if cfg.manticore.fulltext_table != cfg.manticore.vector_table:
                    raise ValueError(
                        "For engine=manticore hybrid indexing, fulltext_table and vector_table must point to the same combined table"
                    )
                target_table = cfg.manticore.fulltext_table

            client.drop_table_if_exists(target_table)
            client.create_vector_table(
                target_table,
                cfg.manticore.vector_field,
                model_name=cfg.manticore.vector_model,
                vector_from=cfg.manticore.vector_from,
                hnsw_similarity=cfg.manticore.vector_similarity,
                quantization=cfg.manticore.vector_quantization,
                api_key=cfg.manticore.vector_api_key,
                api_key_file=cfg.manticore.vector_api_key_file,
            )
            started = perf_counter()
            ingest_stats_cached = ModeIndexResult(
                mode=mode,
                duration_seconds=0.0,
                rows_inserted=0,
                docs_per_second=0.0,
                failed_batches=0,
                retried_batches=0,
                error_reasons={},
            )
            if docs_cached:
                cache2 = VectorCache(cfg.manticore.vector_cache_path)
                last_cached_error = ""
                try:
                    total_cached = len(docs_cached)
                    # Hybrid tables must retain text fields for MATCH() on the same table.
                    # Plain vector tables can still skip them when all vectors come from cache.
                    minimal_vector_insert = mode == "vector" and len(docs_for_auto_embed) == 0
                    fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
                    insert_batch_size = max(1, int(cfg.manticore.batch_size))
                    processed_cached = 0
                    for fetch_start in range(0, total_cached, fetch_batch_size):
                        fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                        vecs = cache2.get_vectors_many([k for _, k in fetch_chunk])
                        for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                            ins_chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                            rows: list[dict[str, Any]] = []
                            for doc, key in ins_chunk:
                                vec = vecs.get(key, [])
                                if not vec:
                                    continue
                                r = dict(doc)
                                r["vector"] = vec
                                rows.append(r)
                            s = client.bulk_insert_docs_with_vectors(
                                target_table,
                                cfg.manticore.vector_field,
                                rows,
                                batch_size=max(1, len(rows)),
                                retries=cfg.manticore.insert_retries,
                                minimal_fields=minimal_vector_insert,
                            )
                            if s.last_error:
                                last_cached_error = s.last_error
                                ingest_stats_cached.last_error = s.last_error
                            ingest_stats_cached.rows_inserted += s.rows_inserted
                            ingest_stats_cached.failed_batches += s.failed_batches
                            ingest_stats_cached.retried_batches += s.retried_batches
                            for reason, count in s.retry_reasons.items():
                                ingest_stats_cached.error_reasons[reason] = (
                                    ingest_stats_cached.error_reasons.get(reason, 0) + int(count)
                                )
                            processed_cached += len(ins_chunk)
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "engine": "manticore",
                                        "mode": mode,
                                        "stage": "ingest_cached",
                                        "processed": min(processed_cached, total_cached),
                                        "total": total_cached,
                                        "inserted": ingest_stats_cached.rows_inserted,
                                        "failed_batches": ingest_stats_cached.failed_batches,
                                        "retried_batches": ingest_stats_cached.retried_batches,
                                        "last_error": last_cached_error,
                                        "error_reasons": dict(ingest_stats_cached.error_reasons),
                                    }
                                )
                finally:
                    cache2.close()
            ingest_stats_auto = client.bulk_insert_docs(
                target_table,
                docs_for_auto_embed,
                batch_size=cfg.manticore.batch_size,
                retries=cfg.manticore.insert_retries,
                progress_callback=progress_callback,
                progress_meta={"engine": "manticore", "mode": mode, "stage": "ingest_auto"},
            )
            if ingest_stats_auto.last_error and progress_callback is not None:
                progress_callback(
                    {
                        "engine": "manticore",
                        "mode": mode,
                        "stage": "ingest_auto",
                        "processed": len(docs_for_auto_embed),
                        "total": len(docs_for_auto_embed),
                        "inserted": ingest_stats_auto.rows_inserted,
                        "failed_batches": ingest_stats_auto.failed_batches,
                        "retried_batches": ingest_stats_auto.retried_batches,
                        "last_error": ingest_stats_auto.last_error,
                        "error_reasons": dict(ingest_stats_auto.retry_reasons),
                    }
                )
            ingest_stats = ModeIndexResult(
                mode=mode,
                duration_seconds=0.0,
                rows_inserted=ingest_stats_cached.rows_inserted + ingest_stats_auto.rows_inserted,
                docs_per_second=0.0,
                failed_batches=ingest_stats_cached.failed_batches + ingest_stats_auto.failed_batches,
                retried_batches=ingest_stats_cached.retried_batches + ingest_stats_auto.retried_batches,
                last_error=ingest_stats_auto.last_error or ingest_stats_cached.last_error,
                error_reasons={},
            )
            for reason, count in getattr(ingest_stats_cached, "error_reasons", {}).items():
                ingest_stats.error_reasons[reason] = ingest_stats.error_reasons.get(reason, 0) + int(count)
            for reason, count in getattr(ingest_stats_auto, "retry_reasons", {}).items():
                ingest_stats.error_reasons[reason] = ingest_stats.error_reasons.get(reason, 0) + int(count)

            # Sync cache only when this run actually generated missing vectors.
            if docs_for_auto_embed:
                if progress_callback is not None:
                    progress_callback({"engine": "manticore", "mode": mode, "stage": "cache_sync", "processed": 0, "total": 0, "inserted": ingest_stats.rows_inserted, "failed_batches": 0, "retried_batches": 0})
                populate_cache_from_manticore(
                    cfg,
                    source_table=target_table,
                    model_name=cfg.manticore.vector_model,
                    similarity=cfg.manticore.vector_similarity,
                    vector_from=cfg.manticore.vector_from,
                    progress_callback=(
                        (lambda u: progress_callback({"engine": "manticore", "mode": mode, **u}))
                        if progress_callback is not None
                        else None
                    ),
                )
            elapsed = perf_counter() - started
            out.append(
                ModeIndexResult(
                    mode=mode,
                    duration_seconds=elapsed,
                    rows_inserted=ingest_stats.rows_inserted,
                    docs_per_second=(ingest_stats.rows_inserted / elapsed) if elapsed > 0 else 0.0,
                    failed_batches=ingest_stats.failed_batches,
                    retried_batches=ingest_stats.retried_batches,
                    last_error=ingest_stats.last_error,
                    error_reasons=dict(ingest_stats.error_reasons),
                )
            )
    finally:
        client.close()
    return out


def _run_index_qdrant(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    docs = read_jsonl(docs_path)
    qdrant = QdrantBackend(cfg.qdrant)
    qdrant.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=qdrant, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        qdrant.recreate_bm25_collection()
        started = perf_counter()
        inserted = 0
        processed = 0
        for i in range(0, len(docs), cfg.qdrant.batch_size):
            batch = docs[i : i + cfg.qdrant.batch_size]
            inserted += qdrant.upsert_bm25_text_batch(batch)
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "qdrant",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )
        visible = qdrant.wait_for_points_count(
            qdrant.cfg.bm25_collection,
            expected_count=inserted,
            timeout_seconds=max(cfg.qdrant.timeout_seconds, 600),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "engine": "qdrant",
                    "mode": "fulltext",
                    "stage": "sync",
                    "processed": min(visible, inserted),
                    "total": inserted,
                    "inserted": inserted,
                    "failed_batches": 0,
                    "retried_batches": 0,
                }
            )
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="fulltext",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )

    if mode == "vector":
        fields = parse_vector_from_fields(cfg.manticore.vector_from)
        cache = VectorCache(cfg.manticore.vector_cache_path)
        docs_cached: list[tuple[dict[str, Any], str]] = []
        missing_keys: list[str] = []
        missing_total = 0
        try:
            status_map = cache.get_status_map(
                model=cfg.manticore.vector_model,
                similarity=cfg.manticore.vector_similarity,
            )
            for i, d in enumerate(docs, start=1):
                key = cache_key_from_doc(
                    d,
                    fields=fields,
                    model=cfg.manticore.vector_model,
                    similarity=cfg.manticore.vector_similarity,
                )
                st = status_map.get(key, "missing")
                if st == "missing":
                    missing_total += 1
                    if len(missing_keys) < 5:
                        missing_keys.append(key)
                    continue
                if st == "null":
                    continue
                docs_cached.append((d, key))
                if progress_callback is not None and (i == 1 or i % 1000 == 0 or i == len(docs)):
                    progress_callback(
                        {
                            "engine": "qdrant",
                            "mode": "vector",
                            "stage": "cache_lookup",
                            "processed": i,
                            "total": len(docs),
                            "inserted": len(docs_cached),
                            "failed_batches": 0,
                            "retried_batches": 0,
                        }
                    )
        finally:
            cache.close()
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Qdrant vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding Manticore vector index first so the cache is populated."
            )
        if not docs_cached:
            raise RuntimeError("No vector documents available in cache (all missing/null).")
        cache2 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            first_key = docs_cached[0][1]
            first_vec = (cache2.get_vectors_many([first_key]).get(first_key) or [])
            dim = len(first_vec)
        finally:
            cache2.close()
        if dim <= 0:
            raise RuntimeError("Cached vectors have invalid dimension.")
        qdrant.recreate_dense_collection(size=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.qdrant.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        row = dict(doc)
                        row["vector"] = vec
                        batch.append(row)
                    inserted += qdrant.upsert_dense_batch(batch, wait=False)
                    processed += len(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "qdrant",
                                "mode": "vector",
                                "stage": "ingest",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                            }
                        )
        finally:
            cache3.close()
        visible = qdrant.wait_for_points_count(
            qdrant.cfg.dense_collection,
            expected_count=inserted,
            timeout_seconds=max(cfg.qdrant.timeout_seconds, 600),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "engine": "qdrant",
                    "mode": "vector",
                    "stage": "sync",
                    "processed": min(visible, inserted),
                    "total": inserted,
                    "inserted": inserted,
                    "failed_batches": 0,
                    "retried_batches": 0,
                }
            )
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="vector",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )

    return out


def _run_index_meilisearch(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    docs = read_jsonl(docs_path)
    backend = MeilisearchBackend(cfg.meilisearch)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=meilisearch, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_lexical_index()
        started = perf_counter()
        inserted = 0
        processed = 0
        index_name = backend.lexical_index_name()
        last_task_uid: int | None = None
        for i in range(0, len(docs), cfg.meilisearch.batch_size):
            batch = [
                {
                    "id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.meilisearch.batch_size]
            ]
            last_task_uid = backend.enqueue_documents(index_name, batch)
            inserted += len(batch)
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "meilisearch",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )
        if last_task_uid is not None:
            backend.wait_for_task(last_task_uid, timeout_ms=max(3_600_000, int(cfg.meilisearch.timeout_sec) * 1000))
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="fulltext",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )

    if mode == "vector":
        fields = parse_vector_from_fields(cfg.manticore.vector_from)
        cache = VectorCache(cfg.manticore.vector_cache_path)
        docs_cached: list[tuple[dict[str, Any], str]] = []
        missing_keys: list[str] = []
        missing_total = 0
        try:
            status_map = cache.get_status_map(
                model=cfg.manticore.vector_model,
                similarity=cfg.manticore.vector_similarity,
            )
            for i, d in enumerate(docs, start=1):
                key = cache_key_from_doc(
                    d,
                    fields=fields,
                    model=cfg.manticore.vector_model,
                    similarity=cfg.manticore.vector_similarity,
                )
                st = status_map.get(key, "missing")
                if st == "missing":
                    missing_total += 1
                    if len(missing_keys) < 5:
                        missing_keys.append(key)
                    continue
                if st == "null":
                    continue
                docs_cached.append((d, key))
                if progress_callback is not None and (i == 1 or i % 1000 == 0 or i == len(docs)):
                    progress_callback(
                        {
                            "engine": "meilisearch",
                            "mode": "vector",
                            "stage": "cache_lookup",
                            "processed": i,
                            "total": len(docs),
                            "inserted": len(docs_cached),
                            "failed_batches": 0,
                            "retried_batches": 0,
                        }
                    )
        finally:
            cache.close()
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Meilisearch vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding Manticore vector index first so the cache is populated."
            )
        if not docs_cached:
            raise RuntimeError("No vector documents available in cache (all missing/null).")

        cache2 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            first_key = docs_cached[0][1]
            first_vec = (cache2.get_vectors_many([first_key]).get(first_key) or [])
            dim = len(first_vec)
        finally:
            cache2.close()
        if dim <= 0:
            raise RuntimeError("Cached vectors have invalid dimension.")
        backend.recreate_vector_index(dimension=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        index_name = backend.vector_index_name()
        last_task_uid: int | None = None
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.meilisearch.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": int(doc["id"]),
                                "doc_id": doc["doc_id"],
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "doi": doc.get("doi", ""),
                                "date": doc.get("date", ""),
                                "_vectors": {
                                    cfg.meilisearch.embedder_name: [float(x) for x in vec],
                                },
                            }
                        )
                    last_task_uid = backend.enqueue_documents(index_name, batch)
                    inserted += len(batch)
                    processed += len(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "meilisearch",
                                "mode": "vector",
                                "stage": "ingest",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                            }
                        )
        finally:
            cache3.close()
        if last_task_uid is not None:
            backend.wait_for_task(last_task_uid, timeout_ms=max(3_600_000, int(cfg.meilisearch.timeout_sec) * 1000))
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="vector",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )

    return out


def _run_index_opensearch(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    docs = read_jsonl(docs_path)
    backend = OpenSearchBackend(cfg.opensearch)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=opensearch, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_lexical_index()
        started = perf_counter()
        inserted = 0
        processed = 0
        index_name = backend.lexical_index_name()
        for i in range(0, len(docs), cfg.opensearch.batch_size):
            batch = [
                {
                    "id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.opensearch.batch_size]
            ]
            inserted += backend.bulk_index(index_name, batch)
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "opensearch",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )
        backend.refresh(index_name)
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="fulltext",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )

    if mode == "vector":
        fields = parse_vector_from_fields(cfg.manticore.vector_from)
        cache = VectorCache(cfg.manticore.vector_cache_path)
        docs_cached: list[tuple[dict[str, Any], str]] = []
        missing_keys: list[str] = []
        missing_total = 0
        try:
            status_map = cache.get_status_map(
                model=cfg.manticore.vector_model,
                similarity=cfg.manticore.vector_similarity,
            )
            for i, d in enumerate(docs, start=1):
                key = cache_key_from_doc(
                    d,
                    fields=fields,
                    model=cfg.manticore.vector_model,
                    similarity=cfg.manticore.vector_similarity,
                )
                st = status_map.get(key, "missing")
                if st == "missing":
                    missing_total += 1
                    if len(missing_keys) < 5:
                        missing_keys.append(key)
                    continue
                if st == "null":
                    continue
                docs_cached.append((d, key))
                if progress_callback is not None and (i == 1 or i % 1000 == 0 or i == len(docs)):
                    progress_callback(
                        {
                            "engine": "opensearch",
                            "mode": "vector",
                            "stage": "cache_lookup",
                            "processed": i,
                            "total": len(docs),
                            "inserted": len(docs_cached),
                            "failed_batches": 0,
                            "retried_batches": 0,
                        }
                    )
        finally:
            cache.close()
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during OpenSearch vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding Manticore vector index first so the cache is populated."
            )
        if not docs_cached:
            raise RuntimeError("No vector documents available in cache (all missing/null).")
        cache2 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            first_key = docs_cached[0][1]
            first_vec = (cache2.get_vectors_many([first_key]).get(first_key) or [])
            dim = len(first_vec)
        finally:
            cache2.close()
        if dim <= 0:
            raise RuntimeError("Cached vectors have invalid dimension.")
        backend.recreate_vector_index(dimension=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        index_name = backend.vector_index_name()
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.opensearch.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": int(doc["id"]),
                                "doc_id": doc["doc_id"],
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "doi": doc.get("doi", ""),
                                "date": doc.get("date", ""),
                                cfg.opensearch.vector_field: [float(x) for x in vec],
                            }
                        )
                    inserted += backend.bulk_index(index_name, batch)
                    processed += len(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "opensearch",
                                "mode": "vector",
                                "stage": "ingest",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                            }
                        )
        finally:
            cache3.close()
        backend.refresh(index_name)
        elapsed = perf_counter() - started
        out.append(
            ModeIndexResult(
                mode="vector",
                duration_seconds=elapsed,
                rows_inserted=inserted,
                docs_per_second=(inserted / elapsed) if elapsed > 0 else 0.0,
                failed_batches=0,
                retried_batches=0,
            )
        )
    return out


def run_index(
    cfg: AppConfig,
    engine: str,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    if engine == "manticore":
        return _run_index_manticore(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "qdrant":
        return _run_index_qdrant(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "meilisearch":
        return _run_index_meilisearch(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "opensearch":
        return _run_index_opensearch(cfg, mode=mode, progress_callback=progress_callback)
    raise ValueError(f"Unsupported engine: {engine}")
