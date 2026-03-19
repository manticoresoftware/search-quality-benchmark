from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable

from .config import AppConfig
from .io_utils import ensure_dir, generate_run_id, read_jsonl
from .manticore import ManticoreClient
from .metrics import average_precision_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k
from .qdrant_backend import QdrantBackend
from .vector_cache import VectorCache, cache_key_from_text, normalize_text


@dataclass
class QueryResult:
    mode: str
    topic_id: str
    query: str
    repeat_idx: int
    latency_ms: float
    success: bool
    error: str
    hit_count: int
    doc_ids: list[str]


def query_text_for_mode(topic: dict[str, str], mode: str) -> str:
    short_q = str(topic.get("query_short") or "").strip()
    long_q = str(topic.get("query_long") or "").strip()
    if not short_q:
        short_q = long_q
    if not long_q:
        long_q = short_q
    if not short_q and not long_q:
        raise ValueError(
            "Prepared topics are missing query_short/query_long. "
            "Please rerun: bench prepare"
        )
    # Policy: shorter query for full-text-like modes, longer query for vector modes.
    if mode == "fulltext":
        return short_q
    return long_q


def _load_qrels_map(qrels_path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            topic_id = str(row["topic_id"])
            doc_id = str(row["doc_id"])
            rel = int(row["relevance"])
            out.setdefault(topic_id, {})[doc_id] = rel
    return out


def _run_single_query_manticore(
    cfg: AppConfig,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    client = ManticoreClient(cfg.manticore)
    try:
        started = perf_counter()
        if mode == "fulltext":
            option_clause = _manticore_fulltext_option_clause(cfg)
            qlit = client.conn.escape(str(query_text))
            rows = client.query(
                f"""
                SELECT doc_id, WEIGHT() AS score
                FROM {cfg.manticore.fulltext_table}
                WHERE MATCH({qlit})
                LIMIT {cfg.benchmark.limit}
                OPTION {option_clause}
                """
            )
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Manticore vector search.")
            vector_literal = "(" + ",".join(f"{float(x):.8f}" for x in query_vector) + ")"
            knn_expr = _manticore_knn_predicate(cfg, vector_literal)
            rows = client.query(
                f"""
                SELECT doc_id, knn_dist() AS distance
                FROM {cfg.manticore.vector_table}
                WHERE {knn_expr}
                LIMIT {cfg.benchmark.limit}
                {"OPTION " + _manticore_vector_option_clause(cfg) if _manticore_vector_option_clause(cfg) else ""}
                """
            )
        latency_ms = (perf_counter() - started) * 1000.0
        doc_ids = [str(r["doc_id"]) for r in rows if r.get("doc_id")]
        return QueryResult(
            mode=mode,
            topic_id=topic_id,
            query=query_text,
            repeat_idx=repeat_idx,
            latency_ms=latency_ms,
            success=True,
            error="",
            hit_count=len(doc_ids),
            doc_ids=doc_ids,
        )
    except Exception as exc:
        return QueryResult(
            mode=mode,
            topic_id=topic_id,
            query=query_text,
            repeat_idx=repeat_idx,
            latency_ms=0.0,
            success=False,
            error=str(exc),
            hit_count=0,
            doc_ids=[],
        )
    finally:
        client.close()


def _collect_diagnostics_manticore(cfg: AppConfig, modes: list[str], topics: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        client = ManticoreClient(cfg.manticore)
        try:
            for t in diagnostics_topics:
                qtext = query_text_for_mode(t, mode)
                entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
                try:
                    if mode == "fulltext":
                        option_clause = _manticore_fulltext_option_clause(cfg)
                        qlit = client.conn.escape(str(qtext))
                        client.query(
                            f"""
                            SELECT doc_id
                            FROM {cfg.manticore.fulltext_table}
                            WHERE MATCH({qlit})
                            LIMIT {cfg.benchmark.limit}
                            OPTION {option_clause}
                            """
                        )
                    else:
                        qvec = _get_query_vectors_from_cache_or_embed(cfg, [qtext]).get(qtext, [])
                        if not qvec:
                            raise RuntimeError("Missing query vector for diagnostics.")
                        vector_literal = "(" + ",".join(f"{float(x):.8f}" for x in qvec) + ")"
                        knn_expr = _manticore_knn_predicate(cfg, vector_literal)
                        client.query(
                            f"""
                            SELECT doc_id
                            FROM {cfg.manticore.vector_table}
                            WHERE {knn_expr}
                            LIMIT {cfg.benchmark.limit}
                            {"OPTION " + _manticore_vector_option_clause(cfg) if _manticore_vector_option_clause(cfg) else ""}
                            """
                        )
                    entry["query_profile"] = client.show_profile()
                    entry["plan"] = client.show_plan()
                except Exception as exc:
                    entry["error"] = str(exc)
                diagnostics[mode].append(entry)
        finally:
            client.close()
    return diagnostics


def _normalize_search_options(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    if low.startswith("option "):
        s = s[7:].strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def _manticore_fulltext_option_clause(cfg: AppConfig) -> str:
    parts: list[str] = []
    extra = _normalize_search_options(cfg.manticore.search_options)
    if extra:
        parts.append(extra)
    parts.extend(
        [
            "idf='plain,tfidf_unnormalized'",
            f"max_matches={max(cfg.benchmark.limit * 5, 1000)}",
        ]
    )
    return ", ".join(parts)


def _manticore_vector_option_clause(cfg: AppConfig) -> str:
    return _normalize_search_options(cfg.manticore.search_options)


def _manticore_fulltext_ranker_label(cfg: AppConfig) -> str:
    extras: list[str] = []
    search_opts = _normalize_search_options(cfg.manticore.search_options)
    if search_opts:
        extras.append(f"search:{search_opts}")
    opts = (cfg.manticore.fulltext_options or "").strip()
    if opts:
        extras.append(f"opts:{opts}")
    if extras:
        return f"[{' '.join(extras)}]"
    return "(manticore default)"


def _manticore_vector_knn_options(cfg: AppConfig) -> str:
    opts: list[str] = []
    oversampling = cfg.manticore.vector_search_oversampling
    if oversampling is not None:
        osf = float(oversampling)
        if osf > 0:
            opts.append(f"oversampling={osf}")
    rescore = cfg.manticore.vector_search_rescore
    if rescore is not None:
        opts.append(f"rescore={1 if int(rescore) > 0 else 0}")
    return ", ".join(opts)


def _manticore_knn_predicate(cfg: AppConfig, vector_literal: str) -> str:
    knn_opts = _manticore_vector_knn_options(cfg)
    if knn_opts:
        return (
            f"knn({cfg.manticore.vector_field}, {cfg.benchmark.limit}, "
            f"{vector_literal}, {{{knn_opts}}})"
        )
    return f"knn({cfg.manticore.vector_field}, {cfg.benchmark.limit}, {vector_literal})"


def _manticore_vector_label(cfg: AppConfig) -> str:
    quant = (cfg.manticore.vector_quantization or "").strip().lower()
    extra: list[str] = []
    if quant:
        extra.append(f"q={quant}")
    if cfg.manticore.vector_search_oversampling is not None:
        extra.append(f"os={float(cfg.manticore.vector_search_oversampling)}")
    if cfg.manticore.vector_search_rescore is not None:
        extra.append(f"rs={1 if int(cfg.manticore.vector_search_rescore) > 0 else 0}")
    search_opts = _normalize_search_options(cfg.manticore.search_options)
    if search_opts:
        extra.append(f"search:{search_opts}")
    if extra:
        return f"{cfg.manticore.vector_model} [{','.join(extra)}]"
    return cfg.manticore.vector_model


def _qdrant_vector_label(cfg: AppConfig) -> str:
    if not bool(cfg.qdrant.dense_binary_quantization):
        return cfg.manticore.vector_model
    extra = ["bq"]
    if cfg.qdrant.dense_search_quantization_oversampling is not None:
        extra.append(f"os={float(cfg.qdrant.dense_search_quantization_oversampling)}")
    if cfg.qdrant.dense_search_quantization_rescore is not None:
        extra.append(f"rs={1 if bool(cfg.qdrant.dense_search_quantization_rescore) else 0}")
    return f"{cfg.manticore.vector_model} [{','.join(extra)}]"


def _get_query_vectors_from_cache_or_embed(cfg: AppConfig, query_texts: list[str]) -> dict[str, list[float]]:
    unique_queries = [q for q in dict.fromkeys(query_texts) if str(q or "").strip()]
    if not unique_queries:
        return {}
    model = cfg.manticore.vector_model
    similarity = cfg.manticore.vector_similarity
    cache = VectorCache(cfg.manticore.vector_cache_path)
    out: dict[str, list[float]] = {}
    missing_queries: list[str] = []
    try:
        for q in unique_queries:
            norm = normalize_text(q)
            key = cache_key_from_text(norm, model=model, similarity=similarity)
            entry = cache.get(key)
            if entry.status == "ok" and entry.vector:
                out[q] = entry.vector
                continue
            if entry.status == "null":
                continue
            missing_queries.append(q)

        if missing_queries:
            manticore = ManticoreClient(cfg.manticore)
            try:
                embedded = manticore.embed_texts(
                    missing_queries,
                    vector_field=cfg.manticore.vector_field,
                    model_name=cfg.manticore.vector_model,
                    vector_from=cfg.manticore.vector_from,
                    api_key=cfg.manticore.vector_api_key,
                    api_key_file=cfg.manticore.vector_api_key_file,
                )
            finally:
                manticore.close()

            for q, vec in zip(missing_queries, embedded):
                norm = normalize_text(q)
                key = cache_key_from_text(norm, model=model, similarity=similarity)
                if vec:
                    vv = [float(x) for x in vec]
                    cache.upsert_ok(
                        cache_key=key,
                        model=model,
                        similarity=similarity,
                        text_value=norm,
                        vector=vv,
                    )
                    out[q] = vv
                else:
                    cache.upsert_null(
                        cache_key=key,
                        model=model,
                        similarity=similarity,
                        text_value=norm,
                    )
            cache.commit()
    finally:
        cache.close()
    return out


def _run_single_query_qdrant(
    cfg: AppConfig,
    backend: QdrantBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.query_bm25_text(query_text, limit=cfg.benchmark.limit)
        else:
            rows = backend.query_dense(query_vector or [], limit=cfg.benchmark.limit)
        latency_ms = (perf_counter() - started) * 1000.0
        doc_ids = [str(r["doc_id"]) for r in rows if r.get("doc_id")]
        return QueryResult(
            mode=mode,
            topic_id=topic_id,
            query=query_text,
            repeat_idx=repeat_idx,
            latency_ms=latency_ms,
            success=True,
            error="",
            hit_count=len(doc_ids),
            doc_ids=doc_ids,
        )
    except Exception as exc:
        return QueryResult(
            mode=mode,
            topic_id=topic_id,
            query=query_text,
            repeat_idx=repeat_idx,
            latency_ms=0.0,
            success=False,
            error=str(exc),
            hit_count=0,
            doc_ids=[],
        )


def _mask_value(path: str, value: Any) -> Any:
    p = path.lower()
    if any(k in p for k in ("password", "api_key")) and str(value or "").strip():
        return "***"
    return value


def _fmt_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _flatten_cfg(prefix: str, obj: Any, out: list[str]) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_cfg(key, obj[k], out)
        return
    out.append(f"{prefix}: {_fmt_value(_mask_value(prefix, obj))}")


def _build_resolved_config_lines(cfg: AppConfig, engine: str, mode: str, profile_name: str | None) -> list[str]:
    lines = [
        f"profile: {profile_name or '-'}",
        f"engine: {engine}",
        f"mode: {'full-text' if mode == 'fulltext' else mode}",
    ]
    visible_sections = {"dataset", "benchmark", "metrics", engine}
    cfg_obj = asdict(cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    flattened: list[str] = []
    _flatten_cfg("", filtered_cfg, flattened)
    lines.extend(flattened)
    return lines


def _build_live_state_lines(cfg: AppConfig, engine: str, modes: list[str]) -> list[str]:
    lines: list[str] = []
    if engine == "manticore":
        client = ManticoreClient(cfg.manticore)
        try:
            for m in modes:
                table = cfg.manticore.fulltext_table if m == "fulltext" else cfg.manticore.vector_table
                try:
                    cnt = client.count_rows(table)
                    lines.append(f"rows ({table}): {int(cnt):,}")
                except Exception as exc:
                    lines.append(f"rows ({table}): n/a ({exc})")
        finally:
            client.close()
        return lines

    backend = QdrantBackend(cfg.qdrant)
    for m in modes:
        collection = cfg.qdrant.bm25_collection if m == "fulltext" else cfg.qdrant.dense_collection
        try:
            cnt = int(backend._count_points(collection))
            lines.append(f"rows ({collection}): {cnt:,}")
        except Exception as exc:
            lines.append(f"rows ({collection}): n/a ({exc})")
    return lines


def run_benchmark(
    cfg: AppConfig,
    engine: str,
    mode: str,
    profile_name: str | None = None,
    concurrency_override: int | None = None,
    fail_fast: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    prep_dir = Path(cfg.dataset.prepared_dir)
    topics = read_jsonl(prep_dir / "topics.jsonl")
    qrels = _load_qrels_map(prep_dir / "qrels.tsv")
    run_id = generate_run_id()
    run_at_unix = int(time())

    if engine == "manticore":
        if mode not in ("fulltext", "vector", "both"):
            raise ValueError("For engine=manticore, mode must be one of: fulltext, vector, both")
        modes = [mode] if mode in ("fulltext", "vector") else ["fulltext", "vector"]
    elif engine == "qdrant":
        if mode not in ("fulltext", "vector", "both"):
            raise ValueError("For engine=qdrant, mode must be one of: fulltext, vector, both")
        modes = [mode] if mode in ("fulltext", "vector") else ["fulltext", "vector"]
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    tasks: list[tuple[str, str, str, int]] = []
    for m in modes:
        for r in range(cfg.benchmark.repeats):
            for t in topics:
                tasks.append((m, t["topic_id"], query_text_for_mode(t, m), r))

    # warmup, sequential and not recorded
    warmup_topics = topics[: cfg.benchmark.warmup_queries]
    query_vectors: dict[str, list[float]] = {}
    qdrant_backend: QdrantBackend | None = None
    if "vector" in modes:
        vector_queries = [query_text_for_mode(t, "vector") for t in topics]
        query_vectors = _get_query_vectors_from_cache_or_embed(cfg, vector_queries)
    if engine == "qdrant":
        qdrant_backend = QdrantBackend(cfg.qdrant)

    for m in modes:
        for t in warmup_topics:
            qtext = query_text_for_mode(t, m)
            if engine == "manticore":
                _run_single_query_manticore(
                    cfg,
                    m,
                    t["topic_id"],
                    qtext,
                    repeat_idx=-1,
                    query_vector=query_vectors.get(qtext, []) if m == "vector" else None,
                )
            else:
                _run_single_query_qdrant(
                    cfg,
                    qdrant_backend,  # type: ignore[arg-type]
                    m,
                    t["topic_id"],
                    qtext,
                    repeat_idx=-1,
                    query_vector=query_vectors.get(qtext, []),
                )

    concurrency = concurrency_override or cfg.benchmark.concurrency
    started = perf_counter()
    query_results: list[QueryResult] = []
    total_tasks = len(tasks)
    completed_tasks = 0
    success_count = 0
    error_count = 0
    completed_by_mode: dict[str, int] = {m: 0 for m in modes}
    report_step = max(1, total_tasks // 20)
    last_reported_completed = 0
    last_report_at = started
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if engine == "manticore":
            futures = [
                pool.submit(
                    _run_single_query_manticore,
                    cfg,
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m == "vector" else None,
                )
                for (m, tid, q, r) in tasks
            ]
        else:
            futures = [
                pool.submit(
                    _run_single_query_qdrant,
                    cfg,
                    qdrant_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []),
                )
                for (m, tid, q, r) in tasks
            ]
        for f in as_completed(futures):
            result = f.result()
            query_results.append(result)
            completed_tasks += 1
            completed_by_mode[result.mode] = completed_by_mode.get(result.mode, 0) + 1
            if result.success:
                success_count += 1
            else:
                error_count += 1
                if fail_fast:
                    for pending in futures:
                        pending.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        "FATAL query failure: "
                        f"engine={engine} mode={result.mode} topic_id={result.topic_id} "
                        f"repeat={result.repeat_idx} error={result.error}"
                    )

            now = perf_counter()
            should_report = (
                completed_tasks == 1
                or completed_tasks == total_tasks
                or (completed_tasks - last_reported_completed) >= report_step
                or (now - last_report_at) >= 1.0
            )
            if progress_callback is not None and should_report:
                progress_callback(
                    {
                        "completed": completed_tasks,
                        "total": total_tasks,
                        "success": success_count,
                        "errors": error_count,
                        "elapsed_seconds": now - started,
                        "by_mode": dict(completed_by_mode),
                    }
                )
                last_reported_completed = completed_tasks
                last_report_at = now
    total_runtime = perf_counter() - started

    first_pass_results: dict[str, dict[str, list[str]]] = {m: {} for m in modes}
    per_query_rows: list[dict[str, Any]] = []
    latencies_by_mode: dict[str, list[float]] = {m: [] for m in modes}
    errors_by_mode: dict[str, int] = {m: 0 for m in modes}
    successes_by_mode: dict[str, int] = {m: 0 for m in modes}

    for r in query_results:
        per_query_rows.append(
            {
                "mode": r.mode,
                "topic_id": r.topic_id,
                "repeat_idx": r.repeat_idx,
                "query": r.query,
                "latency_ms": round(r.latency_ms, 3),
                "success": int(r.success),
                "error": r.error,
                "hit_count": r.hit_count,
                "top_doc_ids": "|".join(r.doc_ids),
            }
        )
        if r.success:
            successes_by_mode[r.mode] += 1
            latencies_by_mode[r.mode].append(r.latency_ms)
            if r.repeat_idx == 0 and r.topic_id not in first_pass_results[r.mode]:
                first_pass_results[r.mode][r.topic_id] = r.doc_ids
        else:
            errors_by_mode[r.mode] += 1

    relevance_metrics: dict[str, dict[str, float]] = {}
    ndcg_ks = sorted(set(int(k) for k in (cfg.metrics.ndcg_ks or [10]) if int(k) > 0))
    if not ndcg_ks:
        ndcg_ks = [10]
    for m in modes:
        ndcgs_by_k: dict[int, list[float]] = {k: [] for k in ndcg_ks}
        maps: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        for topic in topics:
            topic_id = topic["topic_id"]
            rels = qrels.get(topic_id, {})
            if not rels:
                continue
            retrieved = first_pass_results[m].get(topic_id, [])
            for k in ndcg_ks:
                ndcgs_by_k[k].append(ndcg_at_k(retrieved, rels, k))
            maps.append(average_precision_at_k(retrieved, rels, cfg.metrics.map_k))
            precisions.append(precision_at_k(retrieved, rels, cfg.metrics.precision_k))
            recalls.append(recall_at_k(retrieved, rels, cfg.metrics.recall_k))
        mode_rel: dict[str, float] = {
            f"map@{cfg.metrics.map_k}": sum(maps) / len(maps) if maps else 0.0,
            f"p@{cfg.metrics.precision_k}": sum(precisions) / len(precisions) if precisions else 0.0,
            f"recall@{cfg.metrics.recall_k}": sum(recalls) / len(recalls) if recalls else 0.0,
        }
        for k in ndcg_ks:
            xs = ndcgs_by_k[k]
            mode_rel[f"ndcg@{k}"] = sum(xs) / len(xs) if xs else 0.0
        relevance_metrics[m] = mode_rel

    perf_metrics: dict[str, dict[str, float]] = {}
    for m in modes:
        lat = latencies_by_mode[m]
        total = successes_by_mode[m] + errors_by_mode[m]
        perf_metrics[m] = {
            "latency_p50_ms": percentile(lat, 0.50),
            "latency_p95_ms": percentile(lat, 0.95),
            "latency_p99_ms": percentile(lat, 0.99),
            "qps": (successes_by_mode[m] / total_runtime) if total_runtime > 0 else 0.0,
            "error_rate": (errors_by_mode[m] / total) if total > 0 else 0.0,
            "queries_total": float(total),
        }

    diagnostics: dict[str, list[dict[str, Any]]]
    if engine == "manticore":
        diagnostics = _collect_diagnostics_manticore(cfg, modes, topics)
    else:
        diagnostics = {m: [] for m in modes}

    if mode == "vector":
        model_label = _manticore_vector_label(cfg) if engine == "manticore" else _qdrant_vector_label(cfg)
    elif mode == "fulltext":
        model_label = _manticore_fulltext_ranker_label(cfg) if engine == "manticore" else "Qdrant/bm25"
    else:
        if "vector" in modes:
            model_label = _manticore_vector_label(cfg) if engine == "manticore" else _qdrant_vector_label(cfg)
        else:
            model_label = "n/a"

    metrics = {
        "run_id": run_id,
        "run_at_unix": run_at_unix,
        "profile": profile_name or "",
        "engine": engine,
        "mode": mode,
        "model": model_label,
        "benchmark_settings": {
            "concurrency": concurrency,
            "limit": cfg.benchmark.limit,
            "warmup_queries": cfg.benchmark.warmup_queries,
            "repeats": cfg.benchmark.repeats,
        },
        "relevance": relevance_metrics,
        "performance": perf_metrics,
        "runtime_seconds": total_runtime,
    }
    metrics["resolved_config_lines"] = _build_resolved_config_lines(cfg, engine=engine, mode=mode, profile_name=profile_name)
    metrics["live_state_lines"] = _build_live_state_lines(cfg, engine=engine, modes=modes)
    metrics["diagnostics"] = diagnostics
    return run_id, metrics, per_query_rows


def write_run_artifacts(run_id: str, metrics: dict[str, Any], per_query_rows: list[dict[str, Any]]) -> Path:
    from .io_utils import write_csv, write_json
    from .reporting import build_summary_markdown

    out_dir = ensure_dir(Path("artifacts") / run_id)
    write_json(out_dir / "metrics.json", metrics)
    write_csv(
        out_dir / "per_query.csv",
        per_query_rows,
        fieldnames=[
            "mode",
            "topic_id",
            "repeat_idx",
            "query",
            "latency_ms",
            "success",
            "error",
            "hit_count",
            "top_doc_ids",
        ],
    )
    (out_dir / "summary.md").write_text(build_summary_markdown(metrics), encoding="utf-8")
    write_json(out_dir / "diagnostics.json", metrics.get("diagnostics", {}))
    return out_dir
