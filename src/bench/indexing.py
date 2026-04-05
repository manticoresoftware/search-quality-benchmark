from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .config import AppConfig
from .elasticsearch_backend import ElasticsearchBackend
from .embedding_client import resolved_embedding_model
from .io_utils import read_jsonl
from .manticore import ManticoreClient
from .meilisearch_backend import MeilisearchBackend
from .opensearch_backend import OpenSearchBackend
from .qdrant_backend import QdrantBackend
from .typesense_backend import TypesenseBackend
from .vespa_backend import VespaBackend
from .weaviate_backend import WeaviateBackend
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


def _iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _run_parallel_batches(
    batches: list[list[Any]],
    max_workers: int,
    worker: Callable[[list[Any]], Any],
    on_result: Callable[[list[Any], Any], None],
) -> None:
    if not batches:
        return
    workers = max(1, int(max_workers))
    if workers <= 1:
        for batch in batches:
            on_result(batch, worker(batch))
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[Any], list[Any]] = {}
        batch_iter = iter(batches)
        for _ in range(workers):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            pending[executor.submit(worker, batch)] = batch
        while pending:
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                batch = pending.pop(fut)
                on_result(batch, fut.result())
                try:
                    next_batch = next(batch_iter)
                except StopIteration:
                    continue
                pending[executor.submit(worker, next_batch)] = next_batch


def _collect_cached_docs(
    *,
    docs: list[dict[str, Any]],
    cfg: AppConfig,
    engine: str,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[tuple[dict[str, Any], str]], int, list[str]]:
    fields = parse_vector_from_fields(cfg.manticore.vector_from)
    cache = VectorCache(cfg.manticore.vector_cache_path)
    docs_cached: list[tuple[dict[str, Any], str]] = []
    missing_keys: list[str] = []
    missing_total = 0
    try:
        model = resolved_embedding_model(cfg)
        chunk_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
        total = len(docs)
        for start in range(0, total, chunk_size):
            chunk_docs = docs[start : start + chunk_size]
            keyed = [
                (
                    d,
                    cache_key_from_doc(
                        d,
                        fields=fields,
                        model=model,
                        similarity=cfg.manticore.vector_similarity,
                    ),
                )
                for d in chunk_docs
            ]
            status_map = cache.get_status_many([k for _, k in keyed])
            for d, key in keyed:
                st = status_map.get(key, "missing")
                if st == "missing":
                    missing_total += 1
                    if len(missing_keys) < 5:
                        missing_keys.append(key)
                    continue
                if st == "null":
                    continue
                docs_cached.append((d, key))
            processed = min(start + len(chunk_docs), total)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": engine,
                        "mode": mode,
                        "stage": "cache_lookup",
                        "processed": processed,
                        "total": total,
                        "inserted": len(docs_cached),
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )
    finally:
        cache.close()
    return docs_cached, missing_total, missing_keys


def _run_index_manticore(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "manticore",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
    docs = read_jsonl(docs_path)

    client = ManticoreClient(cfg.manticore)
    out: list[ModeIndexResult] = []
    try:
        if mode == "fulltext":
            client.drop_table_if_exists(cfg.manticore.fulltext_table)
            client.create_fulltext_table(
                cfg.manticore.fulltext_table,
                fulltext_options=cfg.manticore.fulltext_options,
                table_engine=cfg.manticore.table_engine,
            )
            started = perf_counter()
            ingest_stats = ModeIndexResult(
                mode="fulltext",
                duration_seconds=0.0,
                rows_inserted=0,
                docs_per_second=0.0,
                failed_batches=0,
                retried_batches=0,
                error_reasons={},
            )
            processed = 0

            def _worker(batch: list[Any]) -> Any:
                local = ManticoreClient(cfg.manticore)
                try:
                    return local.bulk_insert_docs(
                        cfg.manticore.fulltext_table,
                        batch,
                        batch_size=max(1, len(batch)),
                        retries=cfg.manticore.insert_retries,
                    )
                finally:
                    local.close()

            def _on_result(batch: list[Any], s: Any) -> None:
                nonlocal processed
                ingest_stats.rows_inserted += s.rows_inserted
                ingest_stats.failed_batches += s.failed_batches
                ingest_stats.retried_batches += s.retried_batches
                if s.last_error:
                    ingest_stats.last_error = s.last_error
                for reason, count in s.retry_reasons.items():
                    ingest_stats.error_reasons[reason] = ingest_stats.error_reasons.get(reason, 0) + int(count)
                processed += len(batch)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "manticore",
                            "mode": "fulltext",
                            "stage": "ingest",
                            "processed": processed,
                            "total": len(docs),
                            "inserted": ingest_stats.rows_inserted,
                            "failed_batches": ingest_stats.failed_batches,
                            "retried_batches": ingest_stats.retried_batches,
                            "last_error": ingest_stats.last_error,
                            "error_reasons": dict(ingest_stats.error_reasons),
                        }
                    )

            _run_parallel_batches(
                _iter_batches(docs, max(1, int(cfg.manticore.batch_size))),
                max_workers=max(1, int(cfg.manticore.index_concurrency)),
                worker=_worker,
                on_result=_on_result,
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
                    error_reasons=dict(ingest_stats.error_reasons),
                )
            )

        if mode in ("vector", "hybrid"):
            docs_cached, missing_total, missing_keys = _collect_cached_docs(
                docs=docs,
                cfg=cfg,
                engine="manticore",
                mode=mode,
                progress_callback=progress_callback,
            )

            if (
                mode == "vector"
                and str(cfg.manticore.index_layout or "").strip().lower() == "combined"
                and cfg.manticore.fulltext_table == cfg.manticore.vector_table
            ):
                raise ValueError(
                    "Refusing Manticore vector-only indexing when fulltext_table and vector_table share the same table. "
                    "Use mode=hybrid so the combined table keeps lexical fields."
                )

            if missing_keys:
                raise RuntimeError(
                    "Vector cache miss during Manticore vector/hybrid indexing. "
                    f"Missing keys={missing_total} sample={missing_keys}. "
                    "Run `bench prepare` to build embeddings first."
                )

            if not docs_cached:
                raise RuntimeError("No vector documents available (all entries are marked null in cache).")

            target_table = cfg.manticore.vector_table
            if mode == "hybrid":
                if cfg.manticore.fulltext_table != cfg.manticore.vector_table:
                    raise ValueError(
                        "For engine=manticore hybrid indexing, fulltext_table and vector_table must point to the same combined table"
                    )
                target_table = cfg.manticore.fulltext_table

            client.drop_table_if_exists(target_table)
            cache_dim = VectorCache(cfg.manticore.vector_cache_path)
            try:
                first_key = docs_cached[0][1]
                first_vec = cache_dim.get_vectors_many([first_key]).get(first_key) or []
                dim = len(first_vec)
            finally:
                cache_dim.close()
            if dim <= 0:
                raise RuntimeError("Cached vectors have invalid dimension.")
            client.create_vector_table(
                target_table,
                cfg.manticore.vector_field,
                model_name="",
                hnsw_similarity=cfg.manticore.vector_similarity,
                knn_dims=dim,
                quantization=cfg.manticore.vector_quantization,
                fulltext_options=cfg.manticore.fulltext_options if mode == "hybrid" else "",
                table_engine=cfg.manticore.table_engine,
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
                    minimal_vector_insert = mode == "vector"
                    fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
                    insert_batch_size = max(1, int(cfg.manticore.batch_size))
                    prepare_seconds = 0.0
                    send_seconds = 0.0
                    prepared_cached = 0
                    processed_cached = 0
                    for fetch_start in range(0, total_cached, fetch_batch_size):
                        prep_started = perf_counter()
                        fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                        vecs = cache2.get_vectors_many([k for _, k in fetch_chunk])
                        prepared_batches: list[list[Any]] = []
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
                            prepared_batches.append([rows, len(ins_chunk)])
                        prepare_seconds += perf_counter() - prep_started
                        prepared_cached += len(fetch_chunk)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "engine": "manticore",
                                    "mode": mode,
                                    "stage": "prepare_cached",
                                    "processed": min(prepared_cached, total_cached),
                                    "total": total_cached,
                                    "inserted": ingest_stats_cached.rows_inserted,
                                    "failed_batches": ingest_stats_cached.failed_batches,
                                    "retried_batches": ingest_stats_cached.retried_batches,
                                    "last_error": last_cached_error,
                                    "error_reasons": dict(ingest_stats_cached.error_reasons),
                                    "prepare_seconds": prepare_seconds,
                                    "send_seconds": send_seconds,
                                }
                            )

                        def _worker(item: list[Any]) -> Any:
                            local = ManticoreClient(cfg.manticore)
                            try:
                                return local.bulk_insert_docs_with_vectors(
                                    target_table,
                                    cfg.manticore.vector_field,
                                    item[0],
                                    batch_size=max(1, len(item[0])),
                                    retries=cfg.manticore.insert_retries,
                                    minimal_fields=minimal_vector_insert,
                                )
                            finally:
                                local.close()

                        def _on_result(item: list[Any], s: Any) -> None:
                            nonlocal processed_cached, last_cached_error
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
                            processed_cached += int(item[1])
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "engine": "manticore",
                                        "mode": mode,
                                        "stage": "send_cached",
                                        "processed": min(processed_cached, total_cached),
                                        "total": total_cached,
                                        "inserted": ingest_stats_cached.rows_inserted,
                                        "failed_batches": ingest_stats_cached.failed_batches,
                                        "retried_batches": ingest_stats_cached.retried_batches,
                                        "last_error": last_cached_error,
                                        "error_reasons": dict(ingest_stats_cached.error_reasons),
                                    }
                                )

                        send_started = perf_counter()
                        _run_parallel_batches(
                            prepared_batches,
                            max_workers=max(1, int(cfg.manticore.index_concurrency)),
                            worker=_worker,
                            on_result=_on_result,
                        )
                        send_seconds += perf_counter() - send_started
                finally:
                    cache2.close()
            ingest_stats = ModeIndexResult(
                mode=mode,
                duration_seconds=0.0,
                rows_inserted=ingest_stats_cached.rows_inserted,
                docs_per_second=0.0,
                failed_batches=ingest_stats_cached.failed_batches,
                retried_batches=ingest_stats_cached.retried_batches,
                last_error=ingest_stats_cached.last_error,
                error_reasons={},
            )
            for reason, count in getattr(ingest_stats_cached, "error_reasons", {}).items():
                ingest_stats.error_reasons[reason] = ingest_stats.error_reasons.get(reason, 0) + int(count)
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
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "qdrant",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
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
        def _worker(batch: list[Any]) -> int:
            return qdrant.upsert_bm25_text_batch(batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
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

        _run_parallel_batches(
            _iter_batches(docs, max(1, int(cfg.qdrant.batch_size))),
            max_workers=max(1, int(cfg.qdrant.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="qdrant",
            mode="vector",
            progress_callback=progress_callback,
        )
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
                prepared_batches: list[list[Any]] = []
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
                    prepared_batches.append([batch, len(chunk)])

                def _worker(item: list[Any]) -> int:
                    return qdrant.upsert_dense_batch(item[0], wait=False)

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
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

                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.qdrant.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
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
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "meilisearch",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
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
        task_uids: list[int] = []
        batches = [
            [
                {
                    "id": str(row["doc_id"]),
                    "bench_id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.meilisearch.batch_size]
            ]
            for i in range(0, len(docs), cfg.meilisearch.batch_size)
        ]

        def _worker(batch: list[Any]) -> int | None:
            return backend.enqueue_documents(index_name, batch)

        def _on_result(batch: list[Any], task_uid: int | None) -> None:
            nonlocal inserted, processed
            if task_uid is not None:
                task_uids.append(int(task_uid))
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

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.meilisearch.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
        )
        timeout_ms = max(3_600_000, int(cfg.meilisearch.timeout_sec) * 1000)
        for task_uid in sorted(set(task_uids)):
            backend.wait_for_task(task_uid, timeout_ms=timeout_ms)
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="meilisearch",
            mode="vector",
            progress_callback=progress_callback,
        )
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
        prepare_seconds = 0.0
        send_seconds = 0.0
        index_name = backend.vector_index_name()
        task_uids: list[int] = []
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.meilisearch.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": str(doc["doc_id"]),
                                "bench_id": int(doc["id"]),
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
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "meilisearch",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int | None:
                    return backend.enqueue_documents(index_name, item[0])

                def _on_result(item: list[Any], task_uid: int | None) -> None:
                    nonlocal inserted, processed
                    if task_uid is not None:
                        task_uids.append(int(task_uid))
                    inserted += len(item[0])
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "meilisearch",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.meilisearch.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
        finally:
            cache3.close()
        timeout_ms = max(3_600_000, int(cfg.meilisearch.timeout_sec) * 1000)
        for task_uid in sorted(set(task_uids)):
            backend.wait_for_task(task_uid, timeout_ms=timeout_ms)
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
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "opensearch",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
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
        batches = [
            [
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
            for i in range(0, len(docs), cfg.opensearch.batch_size)
        ]

        def _worker(batch: list[Any]) -> int:
            return backend.bulk_index(index_name, batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
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

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.opensearch.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="opensearch",
            mode="vector",
            progress_callback=progress_callback,
        )
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
        prepare_seconds = 0.0
        send_seconds = 0.0
        index_name = backend.vector_index_name()
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.opensearch.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
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
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "opensearch",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int:
                    return backend.bulk_index(index_name, item[0])

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "opensearch",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.opensearch.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
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


def _run_index_elasticsearch(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "elasticsearch",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
    docs = read_jsonl(docs_path)
    backend = ElasticsearchBackend(cfg.elasticsearch)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=elasticsearch, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_lexical_index()
        started = perf_counter()
        inserted = 0
        processed = 0
        index_name = backend.lexical_index_name()
        batches = [
            [
                {
                    "id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.elasticsearch.batch_size]
            ]
            for i in range(0, len(docs), cfg.elasticsearch.batch_size)
        ]

        def _worker(batch: list[Any]) -> int:
            return backend.bulk_index(index_name, batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "elasticsearch",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.elasticsearch.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="elasticsearch",
            mode="vector",
            progress_callback=progress_callback,
        )
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Elasticsearch vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding embeddings in `bench prepare` first."
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
        prepare_seconds = 0.0
        send_seconds = 0.0
        index_name = backend.vector_index_name()
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.elasticsearch.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
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
                                cfg.elasticsearch.vector_field: [float(x) for x in vec],
                            }
                        )
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "elasticsearch",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int:
                    return backend.bulk_index(index_name, item[0])

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "elasticsearch",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.elasticsearch.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
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


def _run_index_typesense(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "typesense",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
    docs = read_jsonl(docs_path)
    backend = TypesenseBackend(cfg.typesense)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=typesense, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_collection()
        started = perf_counter()
        inserted = 0
        processed = 0
        batches = [
            [
                {
                    "id": str(row["doc_id"]),
                    "bench_id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.typesense.batch_size]
            ]
            for i in range(0, len(docs), cfg.typesense.batch_size)
        ]

        def _worker(batch: list[Any]) -> int:
            return backend.import_documents(batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "typesense",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.typesense.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="typesense",
            mode="vector",
            progress_callback=progress_callback,
        )
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Typesense vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding embeddings in `bench prepare` first."
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
        backend.recreate_collection(dimension=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        prepare_seconds = 0.0
        send_seconds = 0.0
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.typesense.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": str(doc["doc_id"]),
                                "bench_id": int(doc["id"]),
                                "doc_id": doc["doc_id"],
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "doi": doc.get("doi", ""),
                                "date": doc.get("date", ""),
                                cfg.typesense.vector_field: [float(x) for x in vec],
                            }
                        )
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "typesense",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int:
                    return backend.import_documents(item[0])

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "typesense",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.typesense.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
        finally:
            cache3.close()
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


def _run_index_weaviate(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "weaviate",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
    docs = read_jsonl(docs_path)
    backend = WeaviateBackend(cfg.weaviate)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=weaviate, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_collection()
        started = perf_counter()
        inserted = 0
        processed = 0
        batches = [
            [
                {
                    "id": str(row["doc_id"]),
                    "bench_id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.weaviate.batch_size]
            ]
            for i in range(0, len(docs), cfg.weaviate.batch_size)
        ]

        def _worker(batch: list[Any]) -> int:
            return backend.import_documents(batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "weaviate",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.weaviate.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="weaviate",
            mode="vector",
            progress_callback=progress_callback,
        )
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Weaviate vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding embeddings in `bench prepare` first."
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
        backend.recreate_collection(dimension=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        prepare_seconds = 0.0
        send_seconds = 0.0
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.weaviate.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": str(doc["doc_id"]),
                                "bench_id": int(doc["id"]),
                                "doc_id": doc["doc_id"],
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "doi": doc.get("doi", ""),
                                "date": doc.get("date", ""),
                                cfg.weaviate.vector_field: [float(x) for x in vec],
                            }
                        )
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "weaviate",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int:
                    return backend.import_documents(item[0])

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "weaviate",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.weaviate.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
        finally:
            cache3.close()
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


def _run_index_vespa(
    cfg: AppConfig,
    mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ModeIndexResult]:
    docs_path = f"{cfg.dataset.prepared_dir}/docs.jsonl"
    if progress_callback is not None:
        progress_callback(
            {
                "engine": "vespa",
                "mode": mode,
                "stage": "load_docs",
                "processed": 0,
                "total": 0,
                "inserted": 0,
                "failed_batches": 0,
                "retried_batches": 0,
            }
        )
    docs = read_jsonl(docs_path)
    backend = VespaBackend(cfg.vespa)
    backend.ensure_accessible()
    out: list[ModeIndexResult] = []

    if mode not in ("fulltext", "vector"):
        raise ValueError("For engine=vespa, mode must be one of: fulltext, vector")

    if mode == "fulltext":
        backend.recreate_collection()
        started = perf_counter()
        inserted = 0
        processed = 0
        batches = [
            [
                {
                    "id": str(row["doc_id"]),
                    "bench_id": int(row["id"]),
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "doi": row.get("doi", ""),
                    "date": row.get("date", ""),
                }
                for row in docs[i : i + cfg.vespa.batch_size]
            ]
            for i in range(0, len(docs), cfg.vespa.batch_size)
        ]

        def _worker(batch: list[Any]) -> int:
            return backend.import_documents(batch)

        def _on_result(batch: list[Any], count: int) -> None:
            nonlocal inserted, processed
            inserted += count
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "engine": "vespa",
                        "mode": "fulltext",
                        "stage": "ingest",
                        "processed": processed,
                        "total": len(docs),
                        "inserted": inserted,
                        "failed_batches": 0,
                        "retried_batches": 0,
                    }
                )

        _run_parallel_batches(
            batches,
            max_workers=max(1, int(cfg.vespa.index_concurrency)),
            worker=_worker,
            on_result=_on_result,
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
        docs_cached, missing_total, missing_keys = _collect_cached_docs(
            docs=docs,
            cfg=cfg,
            engine="vespa",
            mode="vector",
            progress_callback=progress_callback,
        )
        if missing_keys:
            raise RuntimeError(
                "Vector cache miss during Vespa vector indexing. "
                f"Missing keys={missing_total} sample={missing_keys}. "
                "Build the corresponding embeddings in `bench prepare` first."
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
        backend.recreate_collection(dimension=dim)
        started = perf_counter()
        inserted = 0
        processed = 0
        prepare_seconds = 0.0
        send_seconds = 0.0
        cache3 = VectorCache(cfg.manticore.vector_cache_path)
        try:
            total_cached = len(docs_cached)
            fetch_batch_size = max(1, int(cfg.manticore.vector_cache_fetch_batch_size))
            insert_batch_size = max(1, int(cfg.vespa.batch_size))
            for fetch_start in range(0, total_cached, fetch_batch_size):
                prep_started = perf_counter()
                fetch_chunk = docs_cached[fetch_start : fetch_start + fetch_batch_size]
                vecs = cache3.get_vectors_many([k for _, k in fetch_chunk])
                prepared_batches: list[list[Any]] = []
                for ins_start in range(0, len(fetch_chunk), insert_batch_size):
                    chunk = fetch_chunk[ins_start : ins_start + insert_batch_size]
                    batch: list[dict[str, Any]] = []
                    for doc, key in chunk:
                        vec = vecs.get(key, [])
                        if not vec:
                            continue
                        batch.append(
                            {
                                "id": str(doc["doc_id"]),
                                "bench_id": int(doc["id"]),
                                "doc_id": doc["doc_id"],
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "doi": doc.get("doi", ""),
                                "date": doc.get("date", ""),
                                cfg.vespa.vector_field: [float(x) for x in vec],
                            }
                        )
                    prepared_batches.append([batch, len(chunk)])
                prepare_seconds += perf_counter() - prep_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "engine": "vespa",
                            "mode": "vector",
                            "stage": "prepare_cached",
                            "processed": min(fetch_start + len(fetch_chunk), total_cached),
                            "total": total_cached,
                            "inserted": inserted,
                            "failed_batches": 0,
                            "retried_batches": 0,
                            "prepare_seconds": prepare_seconds,
                            "send_seconds": send_seconds,
                        }
                    )

                def _worker(item: list[Any]) -> int:
                    return backend.import_documents(item[0])

                def _on_result(item: list[Any], count: int) -> None:
                    nonlocal inserted, processed
                    inserted += count
                    processed += int(item[1])
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "engine": "vespa",
                                "mode": "vector",
                                "stage": "send_cached",
                                "processed": processed,
                                "total": total_cached,
                                "inserted": inserted,
                                "failed_batches": 0,
                                "retried_batches": 0,
                                "prepare_seconds": prepare_seconds,
                                "send_seconds": send_seconds,
                            }
                        )

                send_started = perf_counter()
                _run_parallel_batches(
                    prepared_batches,
                    max_workers=max(1, int(cfg.vespa.index_concurrency)),
                    worker=_worker,
                    on_result=_on_result,
                )
                send_seconds += perf_counter() - send_started
        finally:
            cache3.close()
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
    if engine in ("manticore", "columnar"):
        return _run_index_manticore(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "qdrant":
        return _run_index_qdrant(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "meilisearch":
        return _run_index_meilisearch(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "opensearch":
        return _run_index_opensearch(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "elasticsearch":
        return _run_index_elasticsearch(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "typesense":
        return _run_index_typesense(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "weaviate":
        return _run_index_weaviate(cfg, mode=mode, progress_callback=progress_callback)
    if engine == "vespa":
        return _run_index_vespa(cfg, mode=mode, progress_callback=progress_callback)
    raise ValueError(f"Unsupported engine: {engine}")
