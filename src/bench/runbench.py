from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter, sleep, time
from typing import Any, Callable

from .config import AppConfig, display_engine_for_profile, is_manticore_like_type
from .elasticsearch_backend import ElasticsearchBackend
from .embedding_client import EmbeddingClient, resolved_embedding_model, resolved_embedding_similarity
from .engine_metadata import resolve_engine_artifact_metadata
from .io_utils import ensure_dir, generate_run_id, read_jsonl
from .manticore import ManticoreClient
from .mariadb_backend import MariaDBBackend
from .meilisearch_backend import MeilisearchBackend
from .milvus_backend import MilvusBackend
from .opensearch_backend import OpenSearchBackend
from .metrics import average_precision_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k, reciprocal_rank_at_k
from .qdrant_backend import QdrantBackend
from .search_options import normalize_search_options
from .serenedb_backend import SereneDBBackend
from .typesense_backend import TypesenseBackend
from .vespa_backend import VespaBackend
from .weaviate_backend import WeaviateBackend
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


def query_text_for_mode(
    topic: dict[str, str],
    mode: str,
    *,
    fulltext_query_variant: str = "short",
) -> str:
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
    # Policy: shorter query for full-text-like modes by default, longer query for
    # vector modes. Specific profiles may override the full-text variant.
    if mode == "fulltext":
        return long_q if str(fulltext_query_variant or "short").strip().lower() == "long" else short_q
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
            payload = _manticore_search_request(cfg, mode, query_text, query_vector)
            response = client.search_json(payload)
            hits = response.get("hits", {}).get("hits", [])
            doc_ids = [str((h.get("_source") or {}).get("doc_id", "")) for h in hits if (h.get("_source") or {}).get("doc_id")]
        else:
            sql, params = _manticore_search_sql_request(cfg, mode, query_text, query_vector)
            rows = client.query(sql, params)
            doc_ids = [str(row.get("doc_id", "")) for row in rows if row.get("doc_id")]
        latency_ms = (perf_counter() - started) * 1000.0
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
                qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
                entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
                try:
                    qvec: list[float] | None = None
                    if mode == "hybrid":
                        qvec = _get_query_vectors_from_cache_or_embed(
                            cfg,
                            [query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant)],
                        ).get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        )
                    elif mode == "vector":
                        qvec = _get_query_vectors_from_cache_or_embed(cfg, [qtext]).get(qtext, [])
                    if mode in ("vector", "hybrid") and not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    if mode == "fulltext":
                        payload = _manticore_search_request(cfg, mode, qtext, qvec)
                        entry["request"] = payload
                        entry["response"] = client.search_json(payload)
                    else:
                        sql, params = _manticore_search_sql_request(cfg, mode, qtext, qvec)
                        entry["request"] = {"sql": sql, "params": list(params)}
                        rows = client.query(sql, params)
                        entry["response"] = {
                            "hits": {
                                "hits": [{"_source": {"doc_id": str(row.get("doc_id", ""))}} for row in rows if row.get("doc_id")]
                            }
                        }
                except Exception as exc:
                    entry["error"] = str(exc)
                diagnostics[mode].append(entry)
        finally:
            client.close()
    return diagnostics


def _sanitize_manticore_fulltext_query(query_text: str) -> str:
    q = str(query_text or "").strip()
    if not q:
        return q
    # MATCH() treats punctuation as query syntax operators in extended mode.
    # For benchmark NL queries, normalize punctuation to spaces to avoid parser errors
    # (e.g. MSMARCO query fragments like "5/5 of 60").
    q = re.sub(r"[^\w\s]+", " ", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q, flags=re.UNICODE).strip()
    return q or str(query_text or "").strip()


def _split_manticore_option_list(raw: str) -> list[str]:
    s = normalize_search_options(raw)
    if not s:
        return []
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_manticore_json_options(raw: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in _split_manticore_option_list(raw):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        k = key.strip()
        v = value.strip()
        if not k:
            continue
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            out[k] = v[1:-1]
            continue
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except Exception:
            pass
        try:
            out[k] = float(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def _manticore_match_query_json(cfg: AppConfig, query_text: str) -> dict[str, Any]:
    raw = str(query_text or "").strip()
    mode = str(cfg.manticore.fulltext_query_mode or "and").strip().lower()
    if mode == "quorum":
        sanitized = _sanitize_manticore_fulltext_query(raw)
        terms = re.findall(r"[^\W_]+", sanitized, flags=re.UNICODE)
        if not terms:
            return {"query_string": raw}
        quorum = float(cfg.manticore.fulltext_query_quorum or 0.5)
        return {"query_string": f"\"{' '.join(terms)}\"/{quorum}"}
    if mode == "and":
        return {"match": {"*": {"query": raw, "operator": "and"}}}
    return {"match": {"*": raw}}


def _manticore_match_query_sql(cfg: AppConfig, query_text: str) -> str:
    raw = str(query_text or "").strip()
    mode = str(cfg.manticore.fulltext_query_mode or "and").strip().lower()
    sanitized = _sanitize_manticore_fulltext_query(raw)
    terms = re.findall(r"[^\W_]+", sanitized, flags=re.UNICODE)
    if mode == "quorum":
        if not terms:
            return raw
        quorum = float(cfg.manticore.fulltext_query_quorum or 0.5)
        return f"\"{' '.join(terms)}\"/{quorum}"
    if mode == "or":
        if terms:
            return " | ".join(terms)
        return sanitized or raw
    if terms:
        return " ".join(terms)
    return sanitized or raw


def _manticore_fulltext_option_clause(cfg: AppConfig) -> str:
    parts: list[str] = []
    extra = normalize_search_options(cfg.manticore.search_options)
    if extra:
        parts.append(extra)
    parts.append(f"max_matches={max(cfg.benchmark.limit * 5, 1000)}")
    return ", ".join(parts)


def _manticore_vector_option_clause(cfg: AppConfig) -> str:
    return normalize_search_options(cfg.manticore.search_options)


def _manticore_hybrid_option_clause(cfg: AppConfig) -> str:
    parts = [f"max_matches={max(cfg.benchmark.limit * 5, 1000)}", "fusion_method='rrf'"]
    vector_opts = _manticore_vector_option_clause(cfg)
    if vector_opts:
        parts.append(vector_opts)
    return ", ".join(parts)


def _manticore_knn_json(cfg: AppConfig, query_vector: list[float]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "field": cfg.manticore.vector_field,
        "query": [float(x) for x in query_vector],
    }
    oversampling = cfg.manticore.vector_search_oversampling
    if oversampling is not None and float(oversampling) > 0:
        payload["oversampling"] = float(oversampling)
    rescore = cfg.manticore.vector_search_rescore
    if rescore is not None:
        payload["rescore"] = bool(int(rescore) > 0)
    return payload


def _manticore_search_request(
    cfg: AppConfig,
    mode: str,
    query_text: str,
    query_vector: list[float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "_source": ["doc_id"],
        "limit": cfg.benchmark.limit,
    }
    if mode == "fulltext":
        payload["table"] = cfg.manticore.fulltext_table
        payload["query"] = _manticore_match_query_json(cfg, query_text)
        payload["options"] = _parse_manticore_json_options(_manticore_fulltext_option_clause(cfg))
        return payload
    if not query_vector:
        raise RuntimeError("Missing query vector for Manticore vector/hybrid search.")
    payload["table"] = cfg.manticore.vector_table
    payload["knn"] = _manticore_knn_json(cfg, query_vector)
    if mode == "hybrid":
        payload["query"] = _manticore_match_query_json(cfg, query_text)
        payload["options"] = _parse_manticore_json_options(_manticore_hybrid_option_clause(cfg))
    else:
        vector_opts = _parse_manticore_json_options(_manticore_vector_option_clause(cfg))
        if vector_opts:
            payload["options"] = vector_opts
    return payload


def _target_name_for_run(cfg: AppConfig, engine: str, mode: str) -> str:
    if is_manticore_like_type(engine):
        return cfg.manticore.fulltext_table if mode == "fulltext" else cfg.manticore.vector_table
    if engine == "qdrant":
        if mode == "fulltext":
            return cfg.qdrant.bm25_collection
        if mode == "hybrid":
            return f"{cfg.qdrant.bm25_collection} + {cfg.qdrant.dense_collection}"
        return cfg.qdrant.dense_collection
    if engine == "opensearch":
        base = str(cfg.opensearch.index_name or "").strip()
        return base
    if engine == "elasticsearch":
        return str(cfg.elasticsearch.index_name or "").strip()
    if engine == "typesense":
        return str(cfg.typesense.collection_name or "").strip()
    if engine == "milvus":
        return str(cfg.milvus.collection_name or "").strip()
    if engine == "mariadb":
        return str(cfg.mariadb.table_name or "").strip()
    if engine == "serenedb":
        return str(cfg.serenedb.table_name or "").strip()
    if engine == "weaviate":
        return str(cfg.weaviate.collection_name or "").strip()
    if engine == "vespa":
        return str(cfg.vespa.collection_name or "").strip()
    base = str(cfg.meilisearch.index_name or "").strip()
    return base


def _manticore_fulltext_ranker_label(cfg: AppConfig) -> str:
    extras: list[str] = []
    profile = str(cfg.manticore.table_profile or "").strip()
    if profile:
        extras.append(f"profile:{profile}")
    search_opts = normalize_search_options(cfg.manticore.search_options)
    if search_opts:
        extras.append(f"search:{search_opts}")
    qmode = str(cfg.manticore.fulltext_query_mode or "and").strip().lower()
    if qmode == "quorum":
        extras.append(f"query:quorum/{float(cfg.manticore.fulltext_query_quorum or 0.5)}")
    elif qmode != "and":
        extras.append(f"query:{qmode}")
    opts = (cfg.manticore.fulltext_options or "").strip()
    if opts:
        extras.append(f"opts:{opts}")
    if extras:
        return f"[{' '.join(extras)}]"
    return "(manticore default)"


def _manticore_table_options_label(cfg: AppConfig) -> str:
    extras: list[str] = []
    profile = str(cfg.manticore.table_profile or "").strip()
    if profile:
        extras.append(f"profile:{profile}")
    opts = (cfg.manticore.fulltext_options or "").strip()
    if opts:
        extras.append(f"opts:{opts}")
    return " ".join(extras)


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
        return f"knn({cfg.manticore.vector_field}, {vector_literal}, {{{knn_opts}}})"
    return f"knn({cfg.manticore.vector_field}, {vector_literal})"


def _manticore_vector_literal(query_vector: list[float]) -> str:
    return "(" + ",".join(format(float(x), ".8g") for x in query_vector) + ")"


def _manticore_search_sql_request(
    cfg: AppConfig,
    mode: str,
    query_text: str,
    query_vector: list[float] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    table_name = cfg.manticore.fulltext_table if mode == "fulltext" else cfg.manticore.vector_table
    where_parts: list[str] = []
    params: list[Any] = []
    if mode in ("fulltext", "hybrid"):
        where_parts.append("MATCH(%s)")
        params.append(_manticore_match_query_sql(cfg, query_text))
    if mode in ("vector", "hybrid"):
        if not query_vector:
            raise RuntimeError("Missing query vector for Manticore vector/hybrid search.")
        where_parts.append(_manticore_knn_predicate(cfg, _manticore_vector_literal(query_vector)))
    if not where_parts:
        raise RuntimeError(f"Unsupported Manticore mode: {mode}")
    option_clause = ""
    if mode == "fulltext":
        option_clause = _manticore_fulltext_option_clause(cfg)
    elif mode == "hybrid":
        option_clause = _manticore_hybrid_option_clause(cfg)
    else:
        option_clause = _manticore_vector_option_clause(cfg)
    sql = f"SELECT doc_id FROM {table_name} WHERE " + " AND ".join(where_parts)
    sql += f" LIMIT {int(cfg.benchmark.limit)}"
    if option_clause:
        sql += f" OPTION {option_clause}"
    return sql, tuple(params)


def _manticore_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    quant = (cfg.manticore.vector_quantization or "").strip().lower()
    extra: list[str] = []
    if quant:
        extra.append(f"q={quant}")
    if cfg.manticore.vector_search_oversampling is not None:
        extra.append(f"os={float(cfg.manticore.vector_search_oversampling)}")
    if cfg.manticore.vector_search_rescore is not None:
        extra.append(f"rs={1 if int(cfg.manticore.vector_search_rescore) > 0 else 0}")
    search_opts = normalize_search_options(cfg.manticore.search_options)
    if search_opts:
        extra.append(f"search:{search_opts}")
    if extra:
        return f"{model} [{','.join(extra)}]"
    return model


def _qdrant_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    if not bool(cfg.qdrant.dense_binary_quantization):
        return model
    extra = ["bq"]
    if cfg.qdrant.dense_search_quantization_oversampling is not None:
        extra.append(f"os={float(cfg.qdrant.dense_search_quantization_oversampling)}")
    if cfg.qdrant.dense_search_quantization_rescore is not None:
        extra.append(f"rs={1 if bool(cfg.qdrant.dense_search_quantization_rescore) else 0}")
    return f"{model} [{','.join(extra)}]"


def _qdrant_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extra = ["fusion=rrf", "sparse=Qdrant/bm25"]
    if bool(cfg.qdrant.dense_binary_quantization):
        extra.append("bq")
        if cfg.qdrant.dense_search_quantization_oversampling is not None:
            extra.append(f"os={float(cfg.qdrant.dense_search_quantization_oversampling)}")
        if cfg.qdrant.dense_search_quantization_rescore is not None:
            extra.append(f"rs={1 if bool(cfg.qdrant.dense_search_quantization_rescore) else 0}")
    return f"{model} [{' '.join(extra)}]"


def _meilisearch_fulltext_label(cfg: AppConfig) -> str:
    extras: list[str] = []
    if cfg.meilisearch.searchable_attributes:
        extras.append(f"searchable:{','.join(cfg.meilisearch.searchable_attributes)}")
    if cfg.meilisearch.ranking_rules:
        extras.append(f"rules:{','.join(cfg.meilisearch.ranking_rules)}")
    if cfg.meilisearch.stop_words:
        extras.append(f"stopwords:{len(cfg.meilisearch.stop_words)}")
    if not bool(cfg.meilisearch.typo_tolerance):
        extras.append("typo:off")
    return f"[{' '.join(extras)}]" if extras else "(meilisearch lexical)"


def _meilisearch_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"embedder:{cfg.meilisearch.embedder_name}"]
    if bool(cfg.meilisearch.binary_quantized):
        extras.append("q=1bit")
    return f"{model} [{' '.join(extras)}]"


def _meilisearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        f"embedder:{cfg.meilisearch.embedder_name}",
        f"semanticRatio={float(cfg.meilisearch.hybrid_semantic_ratio):.2f}",
    ]
    if bool(cfg.meilisearch.binary_quantized):
        extras.append("q=1bit")
    return f"{model} [{' '.join(extras)}]"


def _opensearch_fulltext_label(cfg: AppConfig) -> str:
    extras = [f"fields:{','.join(cfg.opensearch.text_fields or ['title', 'abstract'])}"]
    return f"[{' '.join(extras)}]"


def _opensearch_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"field:{cfg.opensearch.vector_field}"]
    return f"{model} [{' '.join(extras)}]"


def _opensearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        f"textBoost={float(cfg.opensearch.hybrid_text_boost):.2f}",
        f"vectorBoost={float(cfg.opensearch.hybrid_vector_boost):.2f}",
    ]
    return f"{model} [{' '.join(extras)}]"


def _elasticsearch_fulltext_label(cfg: AppConfig) -> str:
    extras = [f"fields:{','.join(cfg.elasticsearch.text_fields or ['title', 'abstract'])}"]
    return f"[{' '.join(extras)}]"


def _elasticsearch_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"field:{cfg.elasticsearch.vector_field}"]
    return f"{model} [{' '.join(extras)}]"


def _elasticsearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        f"textBoost={float(cfg.elasticsearch.hybrid_text_boost):.2f}",
        f"vectorBoost={float(cfg.elasticsearch.hybrid_vector_boost):.2f}",
    ]
    return f"{model} [{' '.join(extras)}]"


def _typesense_fulltext_label(cfg: AppConfig) -> str:
    extras = [f"fields:{','.join(cfg.typesense.text_fields or ['title', 'abstract'])}"]
    return f"[{' '.join(extras)}]"


def _typesense_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"field:{cfg.typesense.vector_field}"]
    return f"{model} [{' '.join(extras)}]"


def _typesense_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"alpha={float(cfg.typesense.hybrid_alpha):.2f}"]
    return f"{model} [{' '.join(extras)}]"


def _milvus_fulltext_label(cfg: AppConfig) -> str:
    fields = ",".join(cfg.milvus.text_fields or ["title", "abstract"])
    return (
        f"[BM25 algo:{cfg.milvus.sparse_index_algorithm} "
        f"k1:{float(cfg.milvus.bm25_k1):.2f} b:{float(cfg.milvus.bm25_b):.2f} "
        f"fields:{fields}]"
    )


def _milvus_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return (
        f"{model} [field:{cfg.milvus.dense_vector_field} "
        f"index:{cfg.milvus.dense_index_type} metric:{cfg.milvus.dense_metric_type}]"
    )


def _milvus_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return (
        f"{model} [fusion=rrf k={int(cfg.milvus.hybrid_rrf_k)} "
        f"candidates={int(cfg.milvus.hybrid_candidates)} sparse=BM25]"
    )


def _mariadb_fulltext_label(cfg: AppConfig) -> str:
    fields = ",".join(cfg.mariadb.text_fields or ["title", "abstract"])
    return f"[MATCH/AGAINST mode:{cfg.mariadb.fulltext_search_mode} fields:{fields}]"


def _mariadb_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        f"field:{cfg.mariadb.vector_field}",
        f"distance:{cfg.mariadb.vector_distance}",
        f"M:{int(cfg.mariadb.vector_m)}",
    ]
    return f"{model} [{' '.join(extras)}]"


def _mariadb_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        "fusion=rrf",
        f"k={int(cfg.mariadb.hybrid_rrf_k)}",
        f"alpha={float(cfg.mariadb.hybrid_alpha):.2f}",
    ]
    return f"{model} [{' '.join(extras)}]"


def _serenedb_fulltext_label(cfg: AppConfig) -> str:
    fields = ",".join(cfg.serenedb.text_fields or ["title", "abstract"])
    return f"[BM25 fields:{fields}]"


def _serenedb_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    query_mode = str(cfg.serenedb.vector_query_mode or "exact").strip().lower()
    extras = [
        f"field:{cfg.serenedb.vector_field}",
        f"search:{query_mode}",
        f"metric:{cfg.serenedb.vector_metric}",
    ]
    if query_mode == "index":
        extras.extend(
            [
                f"index:{cfg.serenedb.vector_index_type}",
                f"quant:{cfg.serenedb.vector_quantization or 'none'}",
                f"nprobe:{int(cfg.serenedb.vector_search_nprobe)}",
                f"rerank:{int(cfg.serenedb.vector_rerank_factor)}",
            ]
        )
    return f"{model} [{' '.join(extras)}]"


def _serenedb_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [
        "fusion=rrf",
        f"vector-search:{cfg.serenedb.vector_query_mode}",
        f"k={int(cfg.serenedb.hybrid_rrf_k)}",
        f"alpha={float(cfg.serenedb.hybrid_alpha):.2f}",
    ]
    return f"{model} [{' '.join(extras)}]"


def _weaviate_fulltext_label(cfg: AppConfig) -> str:
    extras = [f"fields:{','.join(cfg.weaviate.text_fields or ['title', 'abstract'])}"]
    return f"[{' '.join(extras)}]"


def _weaviate_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"field:{cfg.weaviate.vector_field}"]
    return f"{model} [{' '.join(extras)}]"


def _weaviate_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"alpha={float(cfg.weaviate.hybrid_alpha):.2f}"]
    return f"{model} [{' '.join(extras)}]"


def _vespa_fulltext_label(cfg: AppConfig) -> str:
    extras = [f"fields:{','.join(cfg.vespa.text_fields or ['title', 'abstract'])}"]
    return f"[{' '.join(extras)}]"


def _vespa_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"field:{cfg.vespa.vector_field}"]
    return f"{model} [{' '.join(extras)}]"


def _vespa_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extras = [f"alpha={float(cfg.vespa.hybrid_alpha):.2f}"]
    return f"{model} [{' '.join(extras)}]"


def _get_query_vectors_from_cache_or_embed(cfg: AppConfig, query_texts: list[str]) -> dict[str, list[float]]:
    unique_queries = [q for q in dict.fromkeys(query_texts) if str(q or "").strip()]
    if not unique_queries:
        return {}
    model = resolved_embedding_model(cfg)
    similarity = resolved_embedding_similarity(cfg)
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
            # Query embedding failures are rare and may be transient/provider-specific.
            # Unlike corpus docs, always retry query-side null cache entries.
            missing_queries.append(q)

        if missing_queries:
            retry_delays = [0.0, 0.5, 1.0, 2.0, 3.0]
            unresolved = list(missing_queries)
            for attempt_idx, delay in enumerate(retry_delays, start=1):
                if not unresolved:
                    break
                if delay > 0:
                    sleep(delay)
                embedded = EmbeddingClient(cfg).embed_texts(unresolved).vectors

                next_unresolved: list[str] = []
                for q, vec in zip(unresolved, embedded):
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
                        next_unresolved.append(q)
                unresolved = next_unresolved

            cache.commit()
    finally:
        cache.close()
    return out


def _collect_diagnostics_meilisearch(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = MeilisearchBackend(cfg.meilisearch)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.lexical_index_name(),
                        "query": qtext,
                        "limit": min(cfg.benchmark.limit, 5),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "vector_dim": len(qvec),
                        "limit": min(cfg.benchmark.limit, 5),
                    }
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "semanticRatio": float(cfg.meilisearch.hybrid_semantic_ratio),
                        "limit": min(cfg.benchmark.limit, 5),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


def _collect_diagnostics_opensearch(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = OpenSearchBackend(cfg.opensearch)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.lexical_index_name(),
                        "query": qtext,
                        "fields": list(cfg.opensearch.text_fields or ["title", "abstract"]),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "vector_dim": len(qvec),
                    }
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "textBoost": float(cfg.opensearch.hybrid_text_boost),
                        "vectorBoost": float(cfg.opensearch.hybrid_vector_boost),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


def _collect_diagnostics_typesense(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = TypesenseBackend(cfg.typesense)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "fields": list(cfg.typesense.text_fields or ["title", "abstract"]),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "vector_dim": len(qvec),
                    }
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "alpha": float(cfg.typesense.hybrid_alpha),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


def _collect_diagnostics_milvus(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = MilvusBackend(cfg.milvus)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    try:
        for mode in modes:
            diagnostics[mode] = []
            for topic in diagnostics_topics:
                qtext = query_text_for_mode(
                    topic,
                    mode,
                    fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                )
                entry: dict[str, Any] = {"topic_id": topic["topic_id"], "query": qtext}
                try:
                    if mode == "fulltext":
                        rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    else:
                        vector_text = query_text_for_mode(
                            topic,
                            "vector",
                            fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                        )
                        query_vector = query_vectors.get(vector_text, [])
                        if not query_vector:
                            raise RuntimeError("Missing query vector for Milvus diagnostics.")
                        if mode == "vector":
                            rows = backend.search_vector(
                                query_vector,
                                limit=min(cfg.benchmark.limit, 5),
                            )
                        else:
                            rows = backend.search_hybrid(
                                qtext,
                                query_vector,
                                limit=min(cfg.benchmark.limit, 5),
                            )
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "mode": mode,
                        "query": qtext,
                        "vector_dim": len(query_vector) if mode != "fulltext" else 0,
                    }
                    entry["top_hits"] = rows[:5]
                except Exception as exc:
                    entry["error"] = str(exc)
                diagnostics[mode].append(entry)
    finally:
        backend.close()
    return diagnostics


def _collect_diagnostics_mariadb(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = MariaDBBackend(cfg.mariadb)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    try:
        for mode in modes:
            diagnostics[mode] = []
            for topic in diagnostics_topics:
                qtext = query_text_for_mode(
                    topic,
                    mode,
                    fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                )
                entry: dict[str, Any] = {"topic_id": topic["topic_id"], "query": qtext}
                try:
                    if mode == "fulltext":
                        rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    else:
                        vector_text = query_text_for_mode(
                            topic,
                            "vector",
                            fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                        )
                        query_vector = query_vectors.get(vector_text, [])
                        if not query_vector:
                            raise RuntimeError("Missing query vector for MariaDB diagnostics.")
                        if mode == "vector":
                            rows = backend.search_vector(query_vector, limit=min(cfg.benchmark.limit, 5))
                        else:
                            rows = backend.search_hybrid(qtext, query_vector, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "table": backend.table_name(),
                        "mode": mode,
                        "query": qtext,
                        "vector_dim": len(query_vector) if mode != "fulltext" else 0,
                    }
                    entry["top_hits"] = rows[:5]
                except Exception as exc:
                    entry["error"] = str(exc)
                diagnostics[mode].append(entry)
    finally:
        backend.close()
    return diagnostics


def _collect_diagnostics_serenedb(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = SereneDBBackend(cfg.serenedb)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    try:
        for mode in modes:
            diagnostics[mode] = []
            for topic in diagnostics_topics:
                qtext = query_text_for_mode(
                    topic,
                    mode,
                    fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                )
                entry: dict[str, Any] = {"topic_id": topic["topic_id"], "query": qtext}
                try:
                    query_vector: list[float] = []
                    if mode == "fulltext":
                        rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    else:
                        vector_text = query_text_for_mode(
                            topic,
                            "vector",
                            fulltext_query_variant=cfg.manticore.fulltext_query_variant,
                        )
                        query_vector = query_vectors.get(vector_text, [])
                        if not query_vector:
                            raise RuntimeError("Missing query vector for SereneDB diagnostics.")
                        if mode == "vector":
                            rows = backend.search_vector(
                                query_vector, limit=min(cfg.benchmark.limit, 5)
                            )
                        else:
                            rows = backend.search_hybrid(
                                qtext, query_vector, limit=min(cfg.benchmark.limit, 5)
                            )
                    entry["request"] = {
                        "table": backend.table_name(),
                        "index": backend.index_name(),
                        "mode": mode,
                        "query": qtext,
                        "vector_dim": len(query_vector),
                    }
                    if mode != "fulltext":
                        vector_query_mode = str(
                            cfg.serenedb.vector_query_mode or "exact"
                        ).strip().lower()
                        entry["request"]["vector_query_mode"] = vector_query_mode
                        entry["request"]["vector_source"] = (
                            backend.table_name()
                            if vector_query_mode == "exact"
                            else backend.index_name()
                        )
                    entry["top_hits"] = rows[:5]
                except Exception as exc:
                    entry["error"] = str(exc)
                diagnostics[mode].append(entry)
    finally:
        backend.close()
    return diagnostics


def _collect_diagnostics_elasticsearch(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = ElasticsearchBackend(cfg.elasticsearch)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.lexical_index_name(),
                        "query": qtext,
                        "fields": list(cfg.elasticsearch.text_fields or ["title", "abstract"]),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "vector_dim": len(qvec),
                    }
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "index": backend.vector_index_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "textBoost": float(cfg.elasticsearch.hybrid_text_boost),
                        "vectorBoost": float(cfg.elasticsearch.hybrid_vector_boost),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


def _collect_diagnostics_weaviate(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = WeaviateBackend(cfg.weaviate)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "fields": list(cfg.weaviate.text_fields or ["title", "abstract"]),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "vector_dim": len(qvec),
                    }
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "alpha": float(cfg.weaviate.hybrid_alpha),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


def _collect_diagnostics_vespa(
    cfg: AppConfig,
    modes: list[str],
    topics: list[dict[str, str]],
    *,
    query_vectors: dict[str, list[float]],
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    backend = VespaBackend(cfg.vespa)
    diagnostics_topics = topics[: cfg.benchmark.diagnostics_queries]
    for mode in modes:
        diagnostics[mode] = []
        for t in diagnostics_topics:
            qtext = query_text_for_mode(t, mode, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            entry: dict[str, Any] = {"topic_id": t["topic_id"], "query": qtext}
            try:
                if mode == "fulltext":
                    rows = backend.search_lexical(qtext, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "fields": list(cfg.vespa.text_fields or ["title", "abstract"]),
                    }
                elif mode == "vector":
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_vector(qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {"collection": backend.collection_name(), "vector_dim": len(qvec)}
                else:
                    qvec = query_vectors.get(
                        query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                    )
                    if not qvec:
                        raise RuntimeError("Missing query vector for diagnostics.")
                    rows = backend.search_hybrid(qtext, qvec, limit=min(cfg.benchmark.limit, 5))
                    entry["request"] = {
                        "collection": backend.collection_name(),
                        "query": qtext,
                        "vector_dim": len(qvec),
                        "alpha": float(cfg.vespa.hybrid_alpha),
                    }
                entry["top_hits"] = rows[:5]
            except Exception as exc:
                entry["error"] = str(exc)
            diagnostics[mode].append(entry)
    return diagnostics


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
        elif mode == "hybrid":
            if not query_vector:
                raise RuntimeError("Missing query vector for Qdrant hybrid search.")
            rows = backend.query_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_meilisearch(
    cfg: AppConfig,
    backend: MeilisearchBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Meilisearch vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Meilisearch hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_opensearch(
    cfg: AppConfig,
    backend: OpenSearchBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for OpenSearch vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for OpenSearch hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_typesense(
    cfg: AppConfig,
    backend: TypesenseBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Typesense vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Typesense hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_milvus(
    cfg: AppConfig,
    backend: MilvusBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Milvus vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Milvus hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
        latency_ms = (perf_counter() - started) * 1000.0
        doc_ids = [str(row["doc_id"]) for row in rows if row.get("doc_id")]
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


def _run_single_query_mariadb(
    cfg: AppConfig,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    backend = MariaDBBackend(cfg.mariadb)
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for MariaDB vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for MariaDB hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
        latency_ms = (perf_counter() - started) * 1000.0
        doc_ids = [str(row["doc_id"]) for row in rows if row.get("doc_id")]
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
        backend.close()


def _run_single_query_serenedb(
    cfg: AppConfig,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    backend = SereneDBBackend(cfg.serenedb)
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for SereneDB vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for SereneDB hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
        latency_ms = (perf_counter() - started) * 1000.0
        doc_ids = [str(row["doc_id"]) for row in rows if row.get("doc_id")]
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
        backend.close()


def _run_single_query_elasticsearch(
    cfg: AppConfig,
    backend: ElasticsearchBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Elasticsearch vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Elasticsearch hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_weaviate(
    cfg: AppConfig,
    backend: WeaviateBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Weaviate vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Weaviate hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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


def _run_single_query_vespa(
    cfg: AppConfig,
    backend: VespaBackend,
    mode: str,
    topic_id: str,
    query_text: str,
    repeat_idx: int,
    query_vector: list[float] | None = None,
) -> QueryResult:
    try:
        started = perf_counter()
        if mode == "fulltext":
            rows = backend.search_lexical(query_text, limit=cfg.benchmark.limit)
        elif mode == "vector":
            if not query_vector:
                raise RuntimeError("Missing query vector for Vespa vector search.")
            rows = backend.search_vector(query_vector, limit=cfg.benchmark.limit)
        else:
            if not query_vector:
                raise RuntimeError("Missing query vector for Vespa hybrid search.")
            rows = backend.search_hybrid(query_text, query_vector, limit=cfg.benchmark.limit)
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
    if any(k in p for k in ("password", "api_key", "token")) and str(value or "").strip():
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
    visible_engine = "manticore" if is_manticore_like_type(engine) else engine
    visible_sections = {"dataset", "benchmark", "metrics", visible_engine}
    cfg_obj = asdict(cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    flattened: list[str] = []
    _flatten_cfg("", filtered_cfg, flattened)
    lines.extend(flattened)
    return lines


def _build_live_state_lines(cfg: AppConfig, engine: str, modes: list[str]) -> list[str]:
    lines: list[str] = []
    if is_manticore_like_type(engine):
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

    if engine == "meilisearch":
        backend = MeilisearchBackend(cfg.meilisearch)
        for m in modes:
            index_name = backend.lexical_index_name() if m == "fulltext" else backend.vector_index_name()
            try:
                cnt = backend.get_document_count(index_name)
                lines.append(f"rows ({index_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({index_name}): n/a ({exc})")
        return lines

    if engine == "opensearch":
        backend = OpenSearchBackend(cfg.opensearch)
        for m in modes:
            index_name = backend.lexical_index_name() if m == "fulltext" else backend.vector_index_name()
            try:
                cnt = backend.get_document_count(index_name)
                lines.append(f"rows ({index_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({index_name}): n/a ({exc})")
        return lines

    if engine == "elasticsearch":
        backend = ElasticsearchBackend(cfg.elasticsearch)
        for m in modes:
            index_name = backend.lexical_index_name() if m == "fulltext" else backend.vector_index_name()
            try:
                cnt = backend.get_document_count(index_name)
                lines.append(f"rows ({index_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({index_name}): n/a ({exc})")
        return lines

    if engine == "typesense":
        backend = TypesenseBackend(cfg.typesense)
        for m in modes:
            collection_name = backend.collection_name()
            try:
                cnt = backend.get_document_count(collection_name)
                lines.append(f"rows ({collection_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({collection_name}): n/a ({exc})")
        return lines

    if engine == "milvus":
        backend = MilvusBackend(cfg.milvus)
        try:
            collection_name = backend.collection_name()
            try:
                count = backend.get_document_count(collection_name)
                lines.append(f"rows ({collection_name}): {int(count):,}")
            except Exception as exc:
                lines.append(f"rows ({collection_name}): n/a ({exc})")
        finally:
            backend.close()
        return lines

    if engine == "mariadb":
        backend = MariaDBBackend(cfg.mariadb)
        try:
            table_name = backend.table_name()
            try:
                count = backend.get_document_count(table_name)
                lines.append(f"rows ({table_name}): {int(count):,}")
            except Exception as exc:
                lines.append(f"rows ({table_name}): n/a ({exc})")
        finally:
            backend.close()
        return lines

    if engine == "serenedb":
        backend = SereneDBBackend(cfg.serenedb)
        try:
            table_name = backend.table_name()
            try:
                count = backend.get_document_count(table_name)
                lines.append(f"rows ({table_name}): {int(count):,}")
            except Exception as exc:
                lines.append(f"rows ({table_name}): n/a ({exc})")
        finally:
            backend.close()
        return lines

    if engine == "weaviate":
        backend = WeaviateBackend(cfg.weaviate)
        for m in modes:
            collection_name = backend.collection_name()
            try:
                cnt = backend.get_document_count(collection_name)
                lines.append(f"rows ({collection_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({collection_name}): n/a ({exc})")
        return lines

    if engine == "vespa":
        backend = VespaBackend(cfg.vespa)
        for m in modes:
            collection_name = backend.collection_name()
            try:
                cnt = backend.get_document_count(collection_name)
                lines.append(f"rows ({collection_name}): {int(cnt):,}")
            except Exception as exc:
                lines.append(f"rows ({collection_name}): n/a ({exc})")
        return lines

    backend = QdrantBackend(cfg.qdrant)
    for m in modes:
        if m == "hybrid":
            for collection in (cfg.qdrant.bm25_collection, cfg.qdrant.dense_collection):
                try:
                    cnt = int(backend._count_points(collection))
                    lines.append(f"rows ({collection}): {cnt:,}")
                except Exception as exc:
                    lines.append(f"rows ({collection}): n/a ({exc})")
            continue
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

    if is_manticore_like_type(engine):
        if mode not in ("fulltext", "vector", "hybrid", "both"):
            raise ValueError("For engine=manticore, mode must be one of: fulltext, vector, hybrid, both")
        modes = [mode] if mode in ("fulltext", "vector", "hybrid") else ["fulltext", "vector"]
    elif engine == "qdrant":
        if mode not in ("fulltext", "vector", "hybrid", "both"):
            raise ValueError("For engine=qdrant, mode must be one of: fulltext, vector, hybrid, both")
        modes = [mode] if mode in ("fulltext", "vector", "hybrid") else ["fulltext", "vector"]
    elif engine == "meilisearch":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=meilisearch, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "opensearch":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=opensearch, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "elasticsearch":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=elasticsearch, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "typesense":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=typesense, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "milvus":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=milvus, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "mariadb":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=mariadb, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "serenedb":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=serenedb, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "weaviate":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=weaviate, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    elif engine == "vespa":
        if mode not in ("fulltext", "vector", "hybrid"):
            raise ValueError("For engine=vespa, mode must be one of: fulltext, vector, hybrid")
        modes = [mode]
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    tasks: list[tuple[str, str, str, int]] = []
    for m in modes:
        for r in range(cfg.benchmark.repeats):
            for t in topics:
                tasks.append(
                    (
                        m,
                        t["topic_id"],
                        query_text_for_mode(t, m, fulltext_query_variant=cfg.manticore.fulltext_query_variant),
                        r,
                    )
                )

    # warmup, sequential and not recorded
    warmup_topics = topics[: cfg.benchmark.warmup_queries]
    query_vectors: dict[str, list[float]] = {}
    vector_query_by_topic_id = {
        str(t["topic_id"]): query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant)
        for t in topics
    }
    qdrant_backend: QdrantBackend | None = None
    meili_backend: MeilisearchBackend | None = None
    opensearch_backend: OpenSearchBackend | None = None
    elasticsearch_backend: ElasticsearchBackend | None = None
    typesense_backend: TypesenseBackend | None = None
    milvus_backend: MilvusBackend | None = None
    weaviate_backend: WeaviateBackend | None = None
    vespa_backend: VespaBackend | None = None
    if "vector" in modes or "hybrid" in modes:
        vector_queries = [
            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant) for t in topics
        ]
        query_vectors = _get_query_vectors_from_cache_or_embed(cfg, vector_queries)
    if engine == "qdrant":
        qdrant_backend = QdrantBackend(cfg.qdrant)
    elif engine == "meilisearch":
        meili_backend = MeilisearchBackend(cfg.meilisearch)
    elif engine == "opensearch":
        opensearch_backend = OpenSearchBackend(cfg.opensearch)
    elif engine == "elasticsearch":
        elasticsearch_backend = ElasticsearchBackend(cfg.elasticsearch)
    elif engine == "typesense":
        typesense_backend = TypesenseBackend(cfg.typesense)
    elif engine == "milvus":
        milvus_backend = MilvusBackend(cfg.milvus)
    elif engine == "weaviate":
        weaviate_backend = WeaviateBackend(cfg.weaviate)
    elif engine == "vespa":
        vespa_backend = VespaBackend(cfg.vespa)

    for m in modes:
        for t in warmup_topics:
            qtext = query_text_for_mode(t, m, fulltext_query_variant=cfg.manticore.fulltext_query_variant)
            if is_manticore_like_type(engine):
                _run_single_query_manticore(
                    cfg,
                    m,
                    t["topic_id"],
                    qtext,
                    repeat_idx=-1,
                    query_vector=(
                        query_vectors.get(qtext, [])
                        if m == "vector"
                        else (
                            query_vectors.get(vector_query_by_topic_id.get(str(t["topic_id"]), qtext), [])
                            if m == "hybrid"
                            else None
                        )
                    ),
                )
            else:
                if engine == "meilisearch":
                    _run_single_query_meilisearch(
                        cfg,
                        meili_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "opensearch":
                    _run_single_query_opensearch(
                        cfg,
                        opensearch_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "typesense":
                    _run_single_query_typesense(
                        cfg,
                        typesense_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "milvus":
                    _run_single_query_milvus(
                        cfg,
                        milvus_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "mariadb":
                    _run_single_query_mariadb(
                        cfg,
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "serenedb":
                    _run_single_query_serenedb(
                        cfg,
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "weaviate":
                    _run_single_query_weaviate(
                        cfg,
                        weaviate_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "vespa":
                    _run_single_query_vespa(
                        cfg,
                        vespa_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                if engine == "elasticsearch":
                    _run_single_query_elasticsearch(
                        cfg,
                        elasticsearch_backend,  # type: ignore[arg-type]
                        m,
                        t["topic_id"],
                        qtext,
                        repeat_idx=-1,
                        query_vector=query_vectors.get(
                            query_text_for_mode(t, "vector", fulltext_query_variant=cfg.manticore.fulltext_query_variant), []
                        ),
                    )
                    continue
                _run_single_query_qdrant(
                    cfg,
                    qdrant_backend,  # type: ignore[arg-type]
                    m,
                    t["topic_id"],
                    qtext,
                    repeat_idx=-1,
                    query_vector=query_vectors.get(vector_query_by_topic_id.get(str(t["topic_id"]), qtext), []),
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
        if is_manticore_like_type(engine):
            futures = [
                pool.submit(
                    _run_single_query_manticore,
                    cfg,
                    m,
                    tid,
                    q,
                    r,
                    (
                        query_vectors.get(q, [])
                        if m == "vector"
                        else (
                            query_vectors.get(vector_query_by_topic_id.get(str(tid), q), [])
                            if m == "hybrid"
                            else None
                        )
                    ),
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "qdrant":
            futures = [
                pool.submit(
                    _run_single_query_qdrant,
                    cfg,
                    qdrant_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m == "vector" else query_vectors.get(vector_query_by_topic_id.get(str(tid), q), []),
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "typesense":
            futures = [
                pool.submit(
                    _run_single_query_typesense,
                    cfg,
                    typesense_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "milvus":
            futures = [
                pool.submit(
                    _run_single_query_milvus,
                    cfg,
                    milvus_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "mariadb":
            futures = [
                pool.submit(
                    _run_single_query_mariadb,
                    cfg,
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "serenedb":
            futures = [
                pool.submit(
                    _run_single_query_serenedb,
                    cfg,
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "weaviate":
            futures = [
                pool.submit(
                    _run_single_query_weaviate,
                    cfg,
                    weaviate_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "vespa":
            futures = [
                pool.submit(
                    _run_single_query_vespa,
                    cfg,
                    vespa_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        elif engine == "elasticsearch":
            futures = [
                pool.submit(
                    _run_single_query_elasticsearch,
                    cfg,
                    elasticsearch_backend,  # type: ignore[arg-type]
                    m,
                    tid,
                    q,
                    r,
                    query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                )
                for (m, tid, q, r) in tasks
            ]
        else:
            if engine == "meilisearch":
                futures = [
                    pool.submit(
                        _run_single_query_meilisearch,
                        cfg,
                        meili_backend,  # type: ignore[arg-type]
                        m,
                        tid,
                        q,
                        r,
                        query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
                    )
                    for (m, tid, q, r) in tasks
                ]
            else:
                futures = [
                    pool.submit(
                        _run_single_query_opensearch,
                        cfg,
                        opensearch_backend,  # type: ignore[arg-type]
                        m,
                        tid,
                        q,
                        r,
                        query_vectors.get(q, []) if m in ("vector", "hybrid") else None,
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
        mrrs: list[float] = []
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
            mrrs.append(reciprocal_rank_at_k(retrieved, rels, cfg.metrics.mrr_k))
            precisions.append(precision_at_k(retrieved, rels, cfg.metrics.precision_k))
            recalls.append(recall_at_k(retrieved, rels, cfg.metrics.recall_k))
        mode_rel: dict[str, float] = {
            f"map@{cfg.metrics.map_k}": sum(maps) / len(maps) if maps else 0.0,
            f"mrr@{cfg.metrics.mrr_k}": sum(mrrs) / len(mrrs) if mrrs else 0.0,
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
    if is_manticore_like_type(engine):
        diagnostics = _collect_diagnostics_manticore(cfg, modes, topics)
    elif engine == "meilisearch":
        diagnostics = _collect_diagnostics_meilisearch(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "opensearch":
        diagnostics = _collect_diagnostics_opensearch(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "elasticsearch":
        diagnostics = _collect_diagnostics_elasticsearch(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "typesense":
        diagnostics = _collect_diagnostics_typesense(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "milvus":
        diagnostics = _collect_diagnostics_milvus(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "mariadb":
        diagnostics = _collect_diagnostics_mariadb(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "serenedb":
        diagnostics = _collect_diagnostics_serenedb(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "weaviate":
        diagnostics = _collect_diagnostics_weaviate(cfg, modes, topics, query_vectors=query_vectors)
    elif engine == "vespa":
        diagnostics = _collect_diagnostics_vespa(cfg, modes, topics, query_vectors=query_vectors)
    else:
        diagnostics = {m: [] for m in modes}

    if mode == "vector":
        if is_manticore_like_type(engine):
            model_label = _manticore_vector_label(cfg)
        elif engine == "qdrant":
            model_label = _qdrant_vector_label(cfg)
        elif engine == "opensearch":
            model_label = _opensearch_vector_label(cfg)
        elif engine == "elasticsearch":
            model_label = _elasticsearch_vector_label(cfg)
        elif engine == "typesense":
            model_label = _typesense_vector_label(cfg)
        elif engine == "milvus":
            model_label = _milvus_vector_label(cfg)
        elif engine == "mariadb":
            model_label = _mariadb_vector_label(cfg)
        elif engine == "serenedb":
            model_label = _serenedb_vector_label(cfg)
        elif engine == "weaviate":
            model_label = _weaviate_vector_label(cfg)
        elif engine == "vespa":
            model_label = _vespa_vector_label(cfg)
        else:
            model_label = _meilisearch_vector_label(cfg)
    elif mode == "hybrid":
        if is_manticore_like_type(engine):
            model = resolved_embedding_model(cfg)
            hybrid_extra = ["fusion=rrf"]
            table_opts = _manticore_table_options_label(cfg)
            if table_opts:
                hybrid_extra.append(table_opts)
            model_label = _manticore_vector_label(cfg).replace(model, f"{model} [{' '.join(hybrid_extra)}]", 1)
        elif engine == "qdrant":
            model_label = _qdrant_hybrid_label(cfg)
        elif engine == "opensearch":
            model_label = _opensearch_hybrid_label(cfg)
        elif engine == "elasticsearch":
            model_label = _elasticsearch_hybrid_label(cfg)
        elif engine == "typesense":
            model_label = _typesense_hybrid_label(cfg)
        elif engine == "milvus":
            model_label = _milvus_hybrid_label(cfg)
        elif engine == "mariadb":
            model_label = _mariadb_hybrid_label(cfg)
        elif engine == "serenedb":
            model_label = _serenedb_hybrid_label(cfg)
        elif engine == "weaviate":
            model_label = _weaviate_hybrid_label(cfg)
        elif engine == "vespa":
            model_label = _vespa_hybrid_label(cfg)
        else:
            model_label = _meilisearch_hybrid_label(cfg)
    elif mode == "fulltext":
        if is_manticore_like_type(engine):
            model_label = _manticore_fulltext_ranker_label(cfg)
        elif engine == "qdrant":
            model_label = "Qdrant/bm25"
        elif engine == "opensearch":
            model_label = _opensearch_fulltext_label(cfg)
        elif engine == "elasticsearch":
            model_label = _elasticsearch_fulltext_label(cfg)
        elif engine == "typesense":
            model_label = _typesense_fulltext_label(cfg)
        elif engine == "milvus":
            model_label = _milvus_fulltext_label(cfg)
        elif engine == "mariadb":
            model_label = _mariadb_fulltext_label(cfg)
        elif engine == "serenedb":
            model_label = _serenedb_fulltext_label(cfg)
        elif engine == "weaviate":
            model_label = _weaviate_fulltext_label(cfg)
        elif engine == "vespa":
            model_label = _vespa_fulltext_label(cfg)
        else:
            model_label = _meilisearch_fulltext_label(cfg)
    else:
        if "vector" in modes:
            if is_manticore_like_type(engine):
                model_label = _manticore_vector_label(cfg)
            elif engine == "qdrant":
                model_label = _qdrant_vector_label(cfg)
            elif engine == "typesense":
                model_label = _typesense_vector_label(cfg)
            elif engine == "milvus":
                model_label = _milvus_vector_label(cfg)
            elif engine == "mariadb":
                model_label = _mariadb_vector_label(cfg)
            elif engine == "serenedb":
                model_label = _serenedb_vector_label(cfg)
            elif engine == "weaviate":
                model_label = _weaviate_vector_label(cfg)
            elif engine == "vespa":
                model_label = _vespa_vector_label(cfg)
            elif engine == "elasticsearch":
                model_label = _elasticsearch_vector_label(cfg)
            else:
                model_label = _opensearch_vector_label(cfg)
        else:
            model_label = "n/a"

    report_engine = display_engine_for_profile(engine, profile_name)
    metrics = {
        "run_id": run_id,
        "run_at_unix": run_at_unix,
        "profile": profile_name or "",
        "dataset_source": cfg.dataset.source,
        "engine": report_engine,
        "mode": mode,
        "model": model_label,
        "target": _target_name_for_run(cfg, engine=engine, mode=mode),
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
    metrics["resolved_config_lines"] = _build_resolved_config_lines(cfg, engine=report_engine, mode=mode, profile_name=profile_name)
    metrics["live_state_lines"] = _build_live_state_lines(cfg, engine=engine, modes=modes)
    metrics["diagnostics"] = diagnostics
    if milvus_backend is not None:
        milvus_backend.close()
    return run_id, metrics, per_query_rows


def write_run_artifacts(run_id: str, metrics: dict[str, Any], per_query_rows: list[dict[str, Any]]) -> Path:
    from .io_utils import write_csv, write_json
    from .reporting import build_summary_markdown

    engine_metadata = resolve_engine_artifact_metadata(str(metrics.get("engine") or ""))
    for key, value in engine_metadata.items():
        metrics.setdefault(key, value)

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
