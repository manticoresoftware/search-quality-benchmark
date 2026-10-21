from __future__ import annotations

import curses
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from .config import AppConfig, _raw_config, display_engine_for_profile, is_manticore_like_type, load_engine_targets
from .discovery import discover_config_paths
from .elasticsearch_backend import ElasticsearchBackend
from .embedding_client import resolved_embedding_model
from .manticore import ManticoreClient
from .mariadb_backend import MariaDBBackend
from .meilisearch_backend import MeilisearchBackend
from .milvus_backend import MilvusBackend, suppress_pymilvus_probe_logs
from .opensearch_backend import OpenSearchBackend
from .qdrant_backend import QdrantBackend
from .runbench import run_benchmark, write_run_artifacts
from .search_options import normalize_search_options
from .serenedb_backend import SereneDBBackend
from .typesense_backend import TypesenseBackend
from .vespa_backend import VespaBackend
from .weaviate_backend import WeaviateBackend
from .table_filters import (
    draw_header_with_filters,
    filter_block_height,
    filter_rows,
    fit_text,
    prompt_filter_value,
)
from .table_layout import dynamic_row_lines, fit_column_widths, wrap_cell, wrap_text_lines

_fit = fit_text
_DATASET_KEYS = {"source", "prepared_dir", "max_docs", "keep_qrels_docs"}
_ENGINE_TARGET_KEYS = {
    "manticore": {"fulltext_table", "vector_table"},
    "qdrant": {"bm25_collection", "dense_collection"},
    "meilisearch": {"index_name"},
    "opensearch": {"index_name"},
    "elasticsearch": {"index_name"},
    "typesense": {"collection_name"},
    "milvus": {"collection_name"},
    "mariadb": {"table_name"},
    "serenedb": {"table_name"},
    "vespa": {"collection_name"},
    "weaviate": {"collection_name"},
}


def _visible_profile_names(path: Path, targets: list[Any] | None = None) -> set[str]:
    raw = _raw_config(path)
    entry_defs: dict[str, dict[str, Any]] = {k: v for k, v in raw.items() if isinstance(v, dict)}
    referenced_bases = {
        str(v.get("base", "")).strip()
        for v in entry_defs.values()
        if isinstance(v, dict) and str(v.get("base", "")).strip()
    }
    resolved_mode: dict[str, str] = {}
    for t in targets or []:
        resolved_mode[str(t.name)] = str(t.mode_filter or "").strip().lower()
    visible: set[str] = set()
    for name, ent in entry_defs.items():
        mode = resolved_mode.get(name) or str(ent.get("type", "")).strip().lower()
        if mode not in {"fulltext", "vector", "hybrid"}:
            continue
        target_keys = _ENGINE_TARGET_KEYS.get(
            "manticore"
            if "manticore" in name or "columnar" in name
            else "qdrant"
            if "qdrant" in name
            else "meilisearch"
            if "meilisearch" in name
            else "opensearch"
            if "opensearch" in name
            else "elasticsearch"
            if "elasticsearch" in name
            else "typesense"
            if "typesense" in name
            else "milvus"
            if "milvus" in name
            else "mariadb"
            if "mariadb" in name
            else "serenedb"
            if "serenedb" in name
            else "vespa"
            if "vespa" in name
            else "weaviate"
            if "weaviate" in name
            else "",
            set(),
        )
        defines_dataset = any(k in ent for k in _DATASET_KEYS)
        defines_target = any(k in ent for k in target_keys)
        if name not in referenced_bases or defines_dataset or defines_target:
            visible.add(name)
    return visible


@dataclass
class RunVariant:
    config_path: str
    entry: str
    engine: str
    mode: str
    model: str
    default_concurrency: int
    cfg: AppConfig
    base_entry: str = ""


@dataclass
class RunExecutionResult:
    variant: RunVariant
    run_id: str
    artifacts_dir: str


@dataclass(frozen=True)
class RunHistoryStatus:
    has_previous_run: bool
    run_count: int


@dataclass(frozen=True)
class RunTargetStatus:
    available: bool
    runnable: bool
    error: str | None = None


def _legacy_history_key(
    profile: str,
    engine: str,
    mode: str,
    model: str,
) -> tuple[str, str, str, str]:
    return (profile, engine.lower(), mode.lower(), model)


def _display_mode(mode: str) -> str:
    m = mode.lower()
    if m == "fulltext":
        return "full-text"
    if m == "vector":
        return "vector"
    if m == "hybrid":
        return "hybrid"
    return mode


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


def _manticore_table_options_label(cfg: AppConfig) -> str:
    parts: list[str] = []
    profile = str(cfg.manticore.table_profile or "").strip()
    if profile:
        parts.append(f"profile:{profile}")
    opts = (cfg.manticore.fulltext_options or "").strip()
    if opts:
        parts.append(f"opts:{opts}")
    return " ".join(parts)


def _qdrant_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    if not bool(cfg.qdrant.dense_binary_quantization):
        return model
    extra = ["bq"]
    if cfg.qdrant.dense_search_quantization_oversampling is not None:
        extra.append(f"os={float(cfg.qdrant.dense_search_quantization_oversampling)}")
    if cfg.qdrant.dense_search_quantization_rescore is not None:
        extra.append(f"rs={int(bool(cfg.qdrant.dense_search_quantization_rescore))}")
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


def _meilisearch_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extra = [f"embedder:{cfg.meilisearch.embedder_name}"]
    if bool(cfg.meilisearch.binary_quantized):
        extra.append("q=1bit")
    return f"{model} [{' '.join(extra)}]"


def _meilisearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extra = [
        f"embedder:{cfg.meilisearch.embedder_name}",
        f"semanticRatio={float(cfg.meilisearch.hybrid_semantic_ratio):.2f}",
    ]
    if bool(cfg.meilisearch.binary_quantized):
        extra.append("q=1bit")
    return f"{model} [{' '.join(extra)}]"


def _opensearch_vector_label(cfg: AppConfig) -> str:
    return f"{resolved_embedding_model(cfg)} [field:{cfg.opensearch.vector_field}]"


def _opensearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extra = [
        f"textBoost={float(cfg.opensearch.hybrid_text_boost):.2f}",
        f"vectorBoost={float(cfg.opensearch.hybrid_vector_boost):.2f}",
    ]
    return f"{model} [{' '.join(extra)}]"


def _elasticsearch_vector_label(cfg: AppConfig) -> str:
    return f"{resolved_embedding_model(cfg)} [field:{cfg.elasticsearch.vector_field}]"


def _elasticsearch_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    extra = [
        f"textBoost={float(cfg.elasticsearch.hybrid_text_boost):.2f}",
        f"vectorBoost={float(cfg.elasticsearch.hybrid_vector_boost):.2f}",
    ]
    return f"{model} [{' '.join(extra)}]"


def _typesense_vector_label(cfg: AppConfig) -> str:
    return f"{resolved_embedding_model(cfg)} [field:{cfg.typesense.vector_field}]"


def _typesense_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return f"{model} [alpha={float(cfg.typesense.hybrid_alpha):.2f}]"


def _milvus_vector_label(cfg: AppConfig) -> str:
    return (
        f"{resolved_embedding_model(cfg)} [field:{cfg.milvus.dense_vector_field} "
        f"index:{cfg.milvus.dense_index_type} metric:{cfg.milvus.dense_metric_type}]"
    )


def _milvus_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return (
        f"{model} [fusion=rrf k={int(cfg.milvus.hybrid_rrf_k)} "
        f"candidates={int(cfg.milvus.hybrid_candidates)} sparse=BM25]"
    )


def _mariadb_vector_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return (
        f"{model} [field:{cfg.mariadb.vector_field} "
        f"distance:{cfg.mariadb.vector_distance} M:{int(cfg.mariadb.vector_m)}]"
    )


def _mariadb_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return (
        f"{model} [fusion=rrf k={int(cfg.mariadb.hybrid_rrf_k)} "
        f"alpha={float(cfg.mariadb.hybrid_alpha):.2f}]"
    )


def _serenedb_vector_label(cfg: AppConfig) -> str:
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
            ]
        )
    return f"{resolved_embedding_model(cfg)} [{' '.join(extras)}]"


def _serenedb_hybrid_label(cfg: AppConfig) -> str:
    return (
        f"{resolved_embedding_model(cfg)} [fusion=rrf k={int(cfg.serenedb.hybrid_rrf_k)} "
        f"alpha={float(cfg.serenedb.hybrid_alpha):.2f} "
        f"lexical-min-match:{float(cfg.serenedb.fulltext_min_match_ratio) * 100.0:g}% "
        f"vector-search:{cfg.serenedb.vector_query_mode}]"
    )


def _weaviate_vector_label(cfg: AppConfig) -> str:
    return f"{resolved_embedding_model(cfg)} [field:{cfg.weaviate.vector_field}]"


def _weaviate_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return f"{model} [alpha={float(cfg.weaviate.hybrid_alpha):.2f}]"


def _vespa_vector_label(cfg: AppConfig) -> str:
    return f"{resolved_embedding_model(cfg)} [field:{cfg.vespa.vector_field}]"


def _vespa_hybrid_label(cfg: AppConfig) -> str:
    model = resolved_embedding_model(cfg)
    return f"{model} [alpha={float(cfg.vespa.hybrid_alpha):.2f}]"


def build_variants_from_config_paths(paths: list[Path]) -> list[RunVariant]:
    variants: list[RunVariant] = []
    for p in paths:
        try:
            targets = load_engine_targets(p)
            visible_names = _visible_profile_names(p, targets)
        except Exception:
            continue
        for target in targets:
            if target.name not in visible_names:
                continue
            cfg = target.cfg
            engine = display_engine_for_profile(target.type, target.name)
            modes = [target.mode_filter] if target.mode_filter in ("fulltext", "vector", "hybrid") else (["fulltext", "vector", "hybrid"] if engine in ("meilisearch", "opensearch", "elasticsearch", "typesense", "milvus", "mariadb", "serenedb", "weaviate", "vespa") else ["fulltext", "vector"])
            for mode in modes:
                if mode == "vector":
                    if is_manticore_like_type(engine):
                        model_or_ranker = _manticore_vector_label(cfg)
                    elif engine == "qdrant":
                        model_or_ranker = _qdrant_vector_label(cfg)
                    elif engine == "opensearch":
                        model_or_ranker = _opensearch_vector_label(cfg)
                    elif engine == "elasticsearch":
                        model_or_ranker = _elasticsearch_vector_label(cfg)
                    elif engine == "typesense":
                        model_or_ranker = _typesense_vector_label(cfg)
                    elif engine == "milvus":
                        model_or_ranker = _milvus_vector_label(cfg)
                    elif engine == "mariadb":
                        model_or_ranker = _mariadb_vector_label(cfg)
                    elif engine == "serenedb":
                        model_or_ranker = _serenedb_vector_label(cfg)
                    elif engine == "weaviate":
                        model_or_ranker = _weaviate_vector_label(cfg)
                    elif engine == "vespa":
                        model_or_ranker = _vespa_vector_label(cfg)
                    else:
                        model_or_ranker = _meilisearch_vector_label(cfg)
                elif mode == "hybrid":
                    if is_manticore_like_type(engine):
                        model = resolved_embedding_model(cfg)
                        hybrid_extra = ["fusion=rrf"]
                        table_opts = _manticore_table_options_label(cfg)
                        if table_opts:
                            hybrid_extra.append(table_opts)
                        model_or_ranker = _manticore_vector_label(cfg).replace(model, f"{model} [{' '.join(hybrid_extra)}]", 1)
                    elif engine == "qdrant":
                        model_or_ranker = _qdrant_hybrid_label(cfg)
                    elif engine == "typesense":
                        model_or_ranker = _typesense_hybrid_label(cfg)
                    elif engine == "milvus":
                        model_or_ranker = _milvus_hybrid_label(cfg)
                    elif engine == "mariadb":
                        model_or_ranker = _mariadb_hybrid_label(cfg)
                    elif engine == "serenedb":
                        model_or_ranker = _serenedb_hybrid_label(cfg)
                    elif engine == "weaviate":
                        model_or_ranker = _weaviate_hybrid_label(cfg)
                    elif engine == "vespa":
                        model_or_ranker = _vespa_hybrid_label(cfg)
                    elif engine == "elasticsearch":
                        model_or_ranker = _elasticsearch_hybrid_label(cfg)
                    else:
                        model_or_ranker = _opensearch_hybrid_label(cfg) if engine == "opensearch" else _meilisearch_hybrid_label(cfg)
                elif is_manticore_like_type(engine):
                    extras: list[str] = []
                    table_opts = _manticore_table_options_label(cfg)
                    if table_opts:
                        extras.append(table_opts)
                    search_opts = normalize_search_options(cfg.manticore.search_options)
                    if search_opts:
                        extras.append(f"search:{search_opts}")
                    qmode = str(cfg.manticore.fulltext_query_mode or "and").strip().lower()
                    if qmode == "quorum":
                        extras.append(f"query:quorum/{float(cfg.manticore.fulltext_query_quorum or 0.5)}")
                    elif qmode != "and":
                        extras.append(f"query:{qmode}")
                    model_or_ranker = f"[{' '.join(extras)}]" if extras else "(manticore default)"
                elif engine == "qdrant":
                    model_or_ranker = "Qdrant/bm25"
                elif engine == "opensearch":
                    model_or_ranker = f"[fields:{','.join(cfg.opensearch.text_fields or ['title', 'abstract'])}]"
                elif engine == "elasticsearch":
                    model_or_ranker = f"[fields:{','.join(cfg.elasticsearch.text_fields or ['title', 'abstract'])}]"
                elif engine == "typesense":
                    model_or_ranker = f"[fields:{','.join(cfg.typesense.text_fields or ['title', 'abstract'])}]"
                elif engine == "milvus":
                    model_or_ranker = f"[BM25 field:{cfg.milvus.content_field} source:{'+'.join(cfg.milvus.text_fields or ['title', 'abstract'])}]"
                elif engine == "mariadb":
                    model_or_ranker = f"[MATCH/AGAINST mode:{cfg.mariadb.fulltext_search_mode} fields:{','.join(cfg.mariadb.text_fields or ['title', 'abstract'])}]"
                elif engine == "serenedb":
                    model_or_ranker = (
                        f"[BM25 fields:{','.join(cfg.serenedb.text_fields or ['title', 'abstract'])} "
                        f"min-match:{float(cfg.serenedb.fulltext_min_match_ratio) * 100.0:g}%]"
                    )
                elif engine == "weaviate":
                    model_or_ranker = f"[fields:{','.join(cfg.weaviate.text_fields or ['title', 'abstract'])}]"
                elif engine == "vespa":
                    model_or_ranker = f"[fields:{','.join(cfg.vespa.text_fields or ['title', 'abstract'])}]"
                else:
                    extras: list[str] = []
                    if cfg.meilisearch.searchable_attributes:
                        extras.append(f"searchable:{','.join(cfg.meilisearch.searchable_attributes)}")
                    if cfg.meilisearch.ranking_rules:
                        extras.append(f"rules:{','.join(cfg.meilisearch.ranking_rules)}")
                    if not bool(cfg.meilisearch.typo_tolerance):
                        extras.append("typo:off")
                    model_or_ranker = f"[{' '.join(extras)}]" if extras else "(meilisearch lexical)"
                variants.append(
                    RunVariant(
                        config_path=str(p),
                        entry=target.name,
                        base_entry=target.base_name,
                        engine=engine,
                        mode=mode,
                        model=model_or_ranker,
                        default_concurrency=int(cfg.benchmark.concurrency),
                        cfg=cfg,
                    )
                )
    dedup: dict[tuple[str, str, str, str, str, int], RunVariant] = {}
    for v in variants:
        key = (
            v.engine,
            v.mode,
            v.model,
            _variant_target_name(v),
            str(v.cfg.dataset.source or ""),
            int(v.cfg.dataset.max_docs or 0),
        )
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = v
            continue
        if prev.entry == v.base_entry:
            dedup[key] = v
            continue
        if v.entry == prev.base_entry:
            continue

    mode_order = {"fulltext": 0, "vector": 1, "hybrid": 2}
    return sorted(
        list(dedup.values()),
        key=lambda v: (
            v.entry.lower(),
            mode_order.get(v.mode, 99),
            v.model.lower(),
            v.engine.lower(),
            v.config_path,
        ),
    )


def _history_key(
    profile: str,
    engine: str,
    mode: str,
    model: str,
    target: str,
) -> tuple[str, str, str, str, str]:
    return (profile, engine.lower(), mode.lower(), model, target)


def _variant_history_key(v: RunVariant) -> tuple[str, str, str, str, str]:
    return _history_key(v.entry, v.engine, v.mode, v.model, _variant_target_name(v))


def _load_run_history_status() -> tuple[
    dict[tuple[str, str, str, str, str], RunHistoryStatus],
    dict[tuple[str, str, str, str], RunHistoryStatus],
]:
    out_exact: dict[tuple[str, str, str, str, str], RunHistoryStatus] = {}
    out_legacy: dict[tuple[str, str, str, str], RunHistoryStatus] = {}
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return out_exact, out_legacy
    for metrics_path in artifacts_dir.glob("*/metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        profile = str(payload.get("profile", "") or "").strip()
        engine = str(payload.get("engine", "") or "").strip().lower()
        mode = str(payload.get("mode", "") or "").strip().lower()
        model = str(payload.get("model", "") or "").strip()
        target = str(payload.get("target", "") or "").strip()
        if not profile or not engine or not mode or not model:
            continue
        legacy_key = _legacy_history_key(profile, engine, mode, model)
        legacy_prev = out_legacy.get(legacy_key)
        out_legacy[legacy_key] = RunHistoryStatus(True, (legacy_prev.run_count if legacy_prev else 0) + 1)
        if target:
            key = _history_key(profile, engine, mode, model, target)
            prev = out_exact.get(key)
            out_exact[key] = RunHistoryStatus(True, (prev.run_count if prev else 0) + 1)
    return out_exact, out_legacy


def _draw_row(
    stdscr: Any,
    y: int,
    v: RunVariant,
    idx: int,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    model_lines_max: int,
    history_status: RunHistoryStatus,
    target_status: RunTargetStatus,
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    color_entry = 0
    color_engine = 0
    color_mode = 0
    color_target = 0
    if curses.has_colors():
        if not history_status.has_previous_run:
            color_entry = curses.color_pair(8)
        if v.engine == "manticore":
            color_engine = curses.color_pair(1)
        elif v.engine == "manticore_avgdl":
            color_engine = curses.color_pair(6)
        elif v.engine == "manticore_default_ranker":
            color_engine = curses.color_pair(3)
        elif v.engine == "manticore_all_default":
            color_engine = curses.color_pair(4)
        elif v.engine == "manticore_full_default":
            color_engine = curses.color_pair(12)
        elif v.engine == "columnar":
            color_engine = curses.color_pair(15)
        elif v.engine == "qdrant":
            color_engine = curses.color_pair(2)
        elif v.engine == "opensearch":
            color_engine = curses.color_pair(9)
        elif v.engine == "elasticsearch":
            color_engine = curses.color_pair(12)
        elif v.engine == "typesense":
            color_engine = curses.color_pair(11)
        elif v.engine == "milvus":
            color_engine = curses.color_pair(6)
        elif v.engine == "mariadb":
            color_engine = curses.color_pair(7)
        elif v.engine == "serenedb":
            color_engine = curses.color_pair(7)
        elif v.engine == "weaviate":
            color_engine = curses.color_pair(13)
        elif v.engine == "vespa":
            color_engine = curses.color_pair(14)
        elif v.engine == "meilisearch":
            color_engine = curses.color_pair(5)

        if v.mode == "fulltext":
            color_mode = curses.color_pair(3)
        elif v.mode == "vector":
            color_mode = curses.color_pair(4)
        elif v.mode == "hybrid":
            color_mode = curses.color_pair(6)
        if not target_status.runnable:
            color_target = curses.color_pair(10)
    columns: list[tuple[str, int, int]] = [
        ("[x]" if selected else "[ ]", attrs, widths["sel"]),
        (v.entry, attrs | color_entry, widths["entry"]),
        (v.engine, attrs | color_engine, widths["engine"]),
        (_display_mode(v.mode), attrs | color_mode, widths["mode"]),
        (_variant_target_name(v), attrs | color_target, widths["target"]),
        (v.model, attrs, widths["model"]),
    ]
    x = 0
    for text, cell_attrs, width in columns:
        cell_lines = wrap_cell(text, width, max_lines=model_lines_max)
        for i in range(model_lines_max):
            stdscr.addstr(y + i, x, cell_lines[i], cell_attrs)
        x += width + 1


def _variant_target_name(v: RunVariant) -> str:
    if is_manticore_like_type(v.engine):
        return v.cfg.manticore.fulltext_table if v.mode == "fulltext" else v.cfg.manticore.vector_table
    if v.engine == "qdrant":
        if v.mode == "fulltext":
            return v.cfg.qdrant.bm25_collection
        if v.mode == "hybrid":
            return f"{v.cfg.qdrant.bm25_collection} + {v.cfg.qdrant.dense_collection}"
        return v.cfg.qdrant.dense_collection
    if v.engine == "opensearch":
        base = str(v.cfg.opensearch.index_name or "").strip()
        return base
    if v.engine == "elasticsearch":
        return str(v.cfg.elasticsearch.index_name or "").strip()
    if v.engine == "typesense":
        return str(v.cfg.typesense.collection_name or "").strip()
    if v.engine == "milvus":
        return str(v.cfg.milvus.collection_name or "").strip()
    if v.engine == "mariadb":
        return str(v.cfg.mariadb.table_name or "").strip()
    if v.engine == "serenedb":
        return str(v.cfg.serenedb.table_name or "").strip()
    if v.engine == "weaviate":
        return str(v.cfg.weaviate.collection_name or "").strip()
    if v.engine == "vespa":
        return str(v.cfg.vespa.collection_name or "").strip()
    base = str(v.cfg.meilisearch.index_name or "").strip()
    return base


def is_variant_runnable(v: RunVariant) -> bool:
    try:
        target = _variant_target_name(v)
        if is_manticore_like_type(v.engine):
            client = ManticoreClient(v.cfg.manticore)
            try:
                return bool(client.table_exists(target))
            finally:
                client.close()
        if v.engine == "qdrant":
            backend = QdrantBackend(v.cfg.qdrant)
            collection_exists = getattr(backend.client, "collection_exists", None)
            if v.mode == "hybrid":
                names = [v.cfg.qdrant.bm25_collection, v.cfg.qdrant.dense_collection]
                if callable(collection_exists):
                    return all(bool(collection_exists(name)) for name in names)
                for name in names:
                    backend._count_points(name)
                return True
            if callable(collection_exists):
                return bool(collection_exists(target))
            backend._count_points(target)
            return True
        if v.engine == "opensearch":
            backend = OpenSearchBackend(v.cfg.opensearch)
            backend.ensure_accessible()
            return backend.index_exists(target)
        if v.engine == "elasticsearch":
            backend = ElasticsearchBackend(v.cfg.elasticsearch)
            backend.ensure_accessible()
            return backend.index_exists(target)
        if v.engine == "typesense":
            backend = TypesenseBackend(v.cfg.typesense)
            backend.ensure_accessible()
            return backend.collection_exists(target)
        if v.engine == "milvus":
            backend = MilvusBackend(v.cfg.milvus)
            try:
                backend.ensure_accessible()
                return backend.collection_supports(v.mode)
            finally:
                backend.close()
        if v.engine == "mariadb":
            backend = MariaDBBackend(v.cfg.mariadb)
            try:
                backend.ensure_accessible()
                if not backend.table_exists(target):
                    return False
                if v.mode in ("fulltext", "hybrid") and not backend.table_has_fulltext(target):
                    return False
                return v.mode == "fulltext" or backend.table_has_vector(target)
            finally:
                backend.close()
        if v.engine == "serenedb":
            backend = SereneDBBackend(v.cfg.serenedb)
            try:
                backend.ensure_accessible()
                if not backend.table_exists(target) or not backend.table_has_fulltext(target):
                    return False
                return v.mode == "fulltext" or backend.table_has_vector(target)
            finally:
                backend.close()
        if v.engine == "weaviate":
            backend = WeaviateBackend(v.cfg.weaviate)
            backend.ensure_accessible()
            return backend.collection_exists(target)
        if v.engine == "vespa":
            backend = VespaBackend(v.cfg.vespa)
            backend.ensure_accessible()
            return backend.collection_exists(target)
        backend = MeilisearchBackend(v.cfg.meilisearch)
        backend.ensure_accessible()
        return backend.index_exists(target)
    except Exception:
        return False


def filter_runnable_variants(variants: list[RunVariant]) -> list[RunVariant]:
    return [v for v in variants if is_variant_runnable(v)]


def _resolve_target_status(v: RunVariant) -> RunTargetStatus:
    try:
        return RunTargetStatus(available=True, runnable=is_variant_runnable(v), error=None)
    except Exception as exc:
        return RunTargetStatus(available=False, runnable=False, error=str(exc))


def _resolve_row_count(v: RunVariant) -> tuple[int | None, str | None]:
    try:
        if is_manticore_like_type(v.engine):
            client = ManticoreClient(v.cfg.manticore)
            try:
                table = _variant_target_name(v)
                return client.count_rows(table), None
            finally:
                client.close()
        if v.engine == "qdrant":
            backend = QdrantBackend(v.cfg.qdrant)
            if v.mode == "hybrid":
                bm25_cnt = int(backend._count_points(v.cfg.qdrant.bm25_collection))
                dense_cnt = int(backend._count_points(v.cfg.qdrant.dense_collection))
                return min(bm25_cnt, dense_cnt), None
            collection = _variant_target_name(v)
            return int(backend._count_points(collection)), None
        if v.engine == "opensearch":
            backend = OpenSearchBackend(v.cfg.opensearch)
            return int(backend.get_document_count(_variant_target_name(v))), None
        if v.engine == "elasticsearch":
            backend = ElasticsearchBackend(v.cfg.elasticsearch)
            return int(backend.get_document_count(_variant_target_name(v))), None
        if v.engine == "typesense":
            backend = TypesenseBackend(v.cfg.typesense)
            return int(backend.get_document_count(_variant_target_name(v))), None
        if v.engine == "milvus":
            backend = MilvusBackend(v.cfg.milvus)
            try:
                return int(backend.get_document_count(_variant_target_name(v))), None
            finally:
                backend.close()
        if v.engine == "mariadb":
            backend = MariaDBBackend(v.cfg.mariadb)
            try:
                return int(backend.get_document_count(_variant_target_name(v))), None
            finally:
                backend.close()
        if v.engine == "serenedb":
            backend = SereneDBBackend(v.cfg.serenedb)
            try:
                return int(backend.get_document_count(_variant_target_name(v))), None
            finally:
                backend.close()
        if v.engine == "weaviate":
            backend = WeaviateBackend(v.cfg.weaviate)
            return int(backend.get_document_count(_variant_target_name(v))), None
        if v.engine == "vespa":
            backend = VespaBackend(v.cfg.vespa)
            return int(backend.get_document_count(_variant_target_name(v))), None
        backend = MeilisearchBackend(v.cfg.meilisearch)
        return int(backend.get_document_count(_variant_target_name(v))), None
    except Exception as exc:
        return None, str(exc)


def _mask_detail_value(path: str, value: Any) -> Any:
    p = path.lower()
    if any(k in p for k in ("password", "api_key", "token")) and str(value or "").strip():
        return "***"
    return value


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _flatten_detail_values(prefix: str, obj: Any, out: dict[str, str]) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_detail_values(key, obj[k], out)
        return
    out[prefix] = _format_detail_value(_mask_detail_value(prefix, obj))


def _variant_detail_values(v: RunVariant) -> dict[str, str]:
    visible_sections = {"dataset", "benchmark", "metrics", v.engine}
    cfg_obj = asdict(v.cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    flattened: dict[str, str] = {}
    _flatten_detail_values("", filtered_cfg, flattened)
    return flattened


def _comparison_variants(variants: list[RunVariant], current: RunVariant) -> list[RunVariant]:
    same_engine = [v for v in variants if v.engine == current.engine]
    return same_engine or [current]


def _common_detail_keys(variants: list[RunVariant], current: RunVariant) -> set[str]:
    variants = _comparison_variants(variants, current)
    if len(variants) <= 1:
        return set()
    flattened = [_variant_detail_values(v) for v in variants]
    common_keys = set(flattened[0].keys())
    for values in flattened[1:]:
        common_keys &= set(values.keys())
    out: set[str] = set()
    for key in common_keys:
        first = flattened[0][key]
        if all(values.get(key) == first for values in flattened[1:]):
            out.add(key)
    return out


def _details_lines(
    v: RunVariant,
    rows_line: str | None = None,
    *,
    hidden_common_keys: set[str] | None = None,
    comparison_scope: str | None = None,
    history_status: RunHistoryStatus | None = None,
    target_status: RunTargetStatus | None = None,
) -> list[str]:
    hidden_common_keys = hidden_common_keys or set()

    lines: list[str] = ["Resolved Config"]
    meta = [
        f"profile: {v.entry}",
        f"engine: {v.engine}",
        f"mode: {_display_mode(v.mode)}",
        f"target: {_variant_target_name(v)}",
    ]
    if history_status is not None:
        meta.append(f"previous runs: {history_status.run_count}")
    if target_status is not None:
        meta.append(f"target status: {'ready' if target_status.runnable else 'missing'}")
        if target_status.error:
            meta.append(f"target error: {target_status.error}")
    lines.extend(meta)
    if hidden_common_keys:
        scope = comparison_scope or "visible rows"
        lines.append(f"common across {scope}: {len(hidden_common_keys)} hidden")
    flattened = _variant_detail_values(v)
    detail_lines = [
        f"{key}: {value}"
        for key, value in flattened.items()
        if key not in hidden_common_keys
    ]
    lines.extend(detail_lines)
    if rows_line:
        lines.extend(["", "Live State", rows_line])
    return lines


def _draw_details_panel(
    stdscr: Any,
    v: RunVariant,
    variants: list[RunVariant],
    widths: dict[str, int],
    w: int,
    h: int,
    row_count_cache: dict[tuple[str, str, str], str],
    history_status: RunHistoryStatus,
    target_status: RunTargetStatus,
) -> None:
    table_width = sum(widths.values()) + (len(widths) - 1)
    panel_x = table_width + 3
    panel_w = w - panel_x - 1
    if panel_w < 24:
        return
    ck = (v.entry, v.engine, v.mode)
    rows_line = row_count_cache.get(ck)
    if rows_line is None:
        cnt, err = _resolve_row_count(v)
        target = _variant_target_name(v)
        if err:
            rows_line = f"rows ({target}): n/a ({err})"
        else:
            rows_line = f"rows ({target}): {int(cnt or 0):,}"
        row_count_cache[ck] = rows_line
    comparison_variants = _comparison_variants(variants, v)
    hidden_common_keys = _common_detail_keys(variants, v)
    lines = _details_lines(
        v,
        rows_line=rows_line,
        hidden_common_keys=hidden_common_keys,
        comparison_scope=f"visible {v.engine} rows ({len(comparison_variants)})",
        history_status=history_status,
        target_status=target_status,
    )
    max_rows = max(0, h - 3)
    y = 1
    for line in lines:
        if y > max_rows:
            break
        attrs = curses.A_BOLD if line in ("Resolved Config", "Live State") else curses.A_DIM
        wrapped = wrap_text_lines(line, panel_w)
        for j, part in enumerate(wrapped):
            if y > max_rows:
                break
            part_attrs = attrs if j == 0 else curses.A_DIM
            stdscr.addstr(y, panel_x, _fit(part, panel_w), part_attrs)
            y += 1


def _select_variants_curses(variants: list[RunVariant]) -> list[RunVariant]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    row_count_cache: dict[tuple[str, str, str], str] = {}
    history_cache, legacy_history_cache = _load_run_history_status()
    target_status_cache: dict[tuple[str, str, str, str], RunTargetStatus] = {}
    all_variants = list(variants)

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset, variants, selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)     # engine: manticore
            curses.init_pair(15, curses.COLOR_WHITE, -1)    # engine: columnar
            curses.init_pair(2, curses.COLOR_YELLOW, -1)    # engine: qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)      # mode: full-text
            curses.init_pair(4, curses.COLOR_WHITE, -1)     # mode: vector
            curses.init_pair(5, curses.COLOR_CYAN, -1)      # engine: meilisearch
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # mode: hybrid
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # profile: no previous run
            curses.init_pair(9, curses.COLOR_MAGENTA, -1)   # engine: opensearch
            curses.init_pair(12, curses.COLOR_BLUE, -1)     # engine: elasticsearch
            curses.init_pair(10, curses.COLOR_WHITE, curses.COLOR_RED)    # target: missing
            curses.init_pair(11, curses.COLOR_RED, -1)      # engine: typesense
            curses.init_pair(7, curses.COLOR_YELLOW, -1)   # engine: mariadb
            curses.init_pair(13, curses.COLOR_CYAN, -1)     # engine: weaviate
            curses.init_pair(14, curses.COLOR_GREEN, -1)    # engine: vespa
        preferred_widths = {
            "sel": 3,
            "entry": 18,
            "engine": 10,
            "mode": 9,
            "target": 24,
            "model": 45,
        }
        minimum_widths = {
            "sel": 3,
            "entry": 12,
            "engine": 8,
            "mode": 7,
            "target": 12,
            "model": 12,
        }
        sort_cols = ["entry", "engine", "mode", "target", "model"]
        sort_idx = 0
        sort_key = sort_cols[sort_idx]
        sort_desc = False
        filters: dict[str, str] = {}

        def _variant_id(v: RunVariant) -> tuple[str, str, str, str]:
            return (v.entry, v.engine, v.mode, v.model)

        def _history_status(v: RunVariant) -> RunHistoryStatus:
            exact = history_cache.get(_variant_history_key(v))
            if exact is not None:
                return exact
            return legacy_history_cache.get(
                _legacy_history_key(v.entry, v.engine, v.mode, v.model),
                RunHistoryStatus(False, 0),
            )

        def _target_status(v: RunVariant) -> RunTargetStatus:
            key = _variant_id(v)
            if key not in target_status_cache:
                target_status_cache[key] = _resolve_target_status(v)
            return target_status_cache[key]

        def _mode_ord(mode: str) -> int:
            return 0 if mode == "fulltext" else (1 if mode == "vector" else (2 if mode == "hybrid" else 99))

        def _sort_value(v: RunVariant, col: str) -> Any:
            if col == "entry":
                return v.entry.lower()
            if col == "engine":
                return v.engine.lower()
            if col == "mode":
                return _mode_ord(v.mode)
            if col == "target":
                return _variant_target_name(v).lower()
            if col == "model":
                return v.model.lower()
            return v.entry.lower()

        def _filter_value(v: RunVariant, col: str) -> Any:
            if col == "entry":
                return v.entry
            if col == "engine":
                return v.engine
            if col == "mode":
                return _display_mode(v.mode)
            if col == "target":
                return _variant_target_name(v)
            if col == "model":
                return v.model
            return ""

        def _apply_sort(*, preserve_current_item: bool = True) -> None:
            nonlocal variants, selected_idx, current, offset
            selected_ids = {_variant_id(variants[i]) for i in selected_idx if 0 <= i < len(variants)}
            current_id = _variant_id(variants[current]) if preserve_current_item and 0 <= current < len(variants) else None
            variants = [
                v
                for v in filter_rows(all_variants, filters, _filter_value)
                if _target_status(v).available
            ]
            variants.sort(key=lambda v: _sort_value(v, sort_key), reverse=sort_desc)
            selected_idx = {i for i, v in enumerate(variants) if _variant_id(v) in selected_ids}
            if current_id is not None:
                for i, v in enumerate(variants):
                    if _variant_id(v) == current_id:
                        current = i
                        break
            current = min(max(0, current), max(0, len(variants) - 1))
            offset = 0

        _apply_sort()
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            widths = fit_column_widths(preferred_widths, minimum_widths, max(1, w - 1))
            row_h = dynamic_row_lines(len(variants), h)
            stdscr.addstr(0, 0, fit_text("↑/↓ move  ←/→ sort col  SPACE toggle  f filter  ENTER run selected  ESC/q cancel", max(1, w - 1)), curses.A_BOLD)
            filter_lines = filter_block_height(filters, widths)
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[
                    ("SEL", "sel"),
                    ("PROFILE", "entry"),
                    ("ENGINE", "engine"),
                    ("MODE", "mode"),
                    ("TARGET", "target"),
                    ("MODEL/OPTS", "model"),
                ],
                widths=widths,
                max_width=max(1, w - 1),
                active_col=sort_key,
                sort_desc=sort_desc,
                filters=filters,
                filter_lines=filter_lines,
            )
            rows_y = 2 + filter_lines
            visible = max(1, (h - rows_y - 2) // row_h)
            if current < offset:
                offset = current
            if current >= offset + visible:
                offset = current - visible + 1
            for i in range(visible):
                ridx = offset + i
                if ridx >= len(variants):
                    break
                _draw_row(
                    stdscr,
                    y=rows_y + i * row_h,
                    v=variants[ridx],
                    idx=ridx + 1,
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    model_lines_max=row_h,
                    history_status=_history_status(variants[ridx]),
                    target_status=_target_status(variants[ridx]),
                )
            if not variants:
                message = (
                    "No run targets match the active filters."
                    if filters
                    else "No available run targets. Start the required engine first."
                )
                stdscr.addstr(rows_y, 0, fit_text(message, max(1, w - 1)), curses.A_DIM)
            if variants:
                _draw_details_panel(
                    stdscr,
                    variants[current],
                    variants,
                    widths=widths,
                    w=w,
                    h=h,
                    row_count_cache=row_count_cache,
                    history_status=_history_status(variants[current]),
                    target_status=_target_status(variants[current]),
                )
            stdscr.addstr(h - 1, 0, fit_text(f"Selected: {len(selected_idx)} / {len(variants)}", max(1, w - 1)), curses.A_DIM)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                current = max(0, current - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                if variants:
                    current = min(len(variants) - 1, current + 1)
            elif ch == curses.KEY_LEFT:
                sort_idx = (sort_idx - 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_sort(preserve_current_item=False)
            elif ch == curses.KEY_RIGHT:
                sort_idx = (sort_idx + 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_sort(preserve_current_item=False)
            elif ch in (ord("f"), ord("F")):
                val = prompt_filter_value(
                    stdscr,
                    prompt=f"Filter {sort_key} (empty clears): ",
                    initial=str(filters.get(sort_key, "") or ""),
                )
                if val is not None:
                    val = val.strip()
                    if val:
                        filters[sort_key] = val
                    else:
                        filters.pop(sort_key, None)
                    _apply_sort()
            elif ch == ord(" "):
                if variants:
                    if current in selected_idx:
                        selected_idx.remove(current)
                    else:
                        selected_idx.add(current)
            elif ch in (10, 13, curses.KEY_ENTER):
                return sorted(selected_idx) if variants else []
            elif ch == 27 or ch in (ord("q"),):
                return []

    # PyMilvus logs handled RPC failures, including full tracebacks, to stderr.
    # Keep all live target probes quiet while curses owns the terminal; their
    # failures are already represented by the target status and details panel.
    with suppress_pymilvus_probe_logs():
        picked = curses.wrapper(_inner)
    return [variants[i] for i in picked]


def select_variants_interactive(variants: list[RunVariant]) -> list[RunVariant]:
    if not variants:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_variants_curses(variants)
        except Exception:
            pass
    # Text fallback
    print("Available runs:")
    for i, v in enumerate(variants, start=1):
        print(
            f"{i:>2}. {v.entry}\t{v.engine}\t{_display_mode(v.mode)}\t"
            f"target={_variant_target_name(v)}\t"
            f"model={v.model}"
        )
    raw = input("Select variant numbers (e.g. 1,3-5): ").strip()
    idxs: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                if 1 <= x <= len(variants):
                    idxs.add(x - 1)
        else:
            x = int(part)
            if 1 <= x <= len(variants):
                idxs.add(x - 1)
    return [variants[i] for i in sorted(idxs)]


def run_selected_variants(
    variants: list[RunVariant],
    concurrency_override: int | None = None,
) -> list[RunExecutionResult]:
    out: list[RunExecutionResult] = []
    use_color = sys.stdout.isatty()
    c_label = "\033[1;35m" if use_color else ""
    c_cmd = "\033[36m" if use_color else ""
    c_reset = "\033[0m" if use_color else ""
    for v in variants:
        conc_part = f" --concurrency {concurrency_override}" if concurrency_override is not None else ""
        cmd = (
            f"bench --config {v.config_path} run --engine {v.engine} "
            f"--mode {v.mode} --entry {v.entry}{conc_part}"
        )
        print(
            f"{c_label}Replay command:{c_reset} {c_cmd}{cmd}{c_reset}",
            flush=True,
        )
        print(
            f"Running: profile={v.entry} engine={v.engine} mode={_display_mode(v.mode)} model={v.model}",
            flush=True,
        )

        def _progress(update: dict[str, Any]) -> None:
            completed = int(update.get("completed", 0))
            total = int(update.get("total", 0))
            success = int(update.get("success", 0))
            errors = int(update.get("errors", 0))
            elapsed = float(update.get("elapsed_seconds", 0.0))
            pct = (completed / total * 100.0) if total else 0.0
            print(
                f"  progress {completed}/{total} ({pct:5.1f}%) ok={success} err={errors} elapsed={elapsed:6.1f}s",
                flush=True,
            )

        run_id, metrics, per_query = run_benchmark(
            cfg=v.cfg,
            engine=v.engine,
            mode=v.mode,
            profile_name=v.entry,
            concurrency_override=concurrency_override,
            progress_callback=_progress,
        )
        artifacts_dir = write_run_artifacts(run_id, metrics, per_query)
        out.append(RunExecutionResult(variant=v, run_id=run_id, artifacts_dir=str(artifacts_dir)))
    return out
