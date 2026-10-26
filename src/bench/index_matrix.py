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
class IndexVariant:
    config_path: str
    entry: str
    engine: str
    mode: str
    model: str
    batch_size: int
    target: str
    cfg: AppConfig
    base_entry: str = ""
    target_kind: str = ""
    used_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetStatus:
    state: str
    actual_count: int | None
    expected_count: int | None
    error: str | None = None


def _display_mode(mode: str) -> str:
    if mode == "fulltext":
        return "full-text"
    if mode == "vector":
        return "vector"
    if mode == "hybrid":
        return "hybrid"
    if mode == "both":
        return "both"
    return mode


def _display_target_kind(kind: str) -> str:
    if kind == "combined":
        return "combined"
    if kind == "fulltext":
        return "full-text"
    if kind == "vector":
        return "vector"
    return kind


def _is_vector_capable_target_kind(kind: str) -> bool:
    return kind in ("vector", "hybrid", "combined")


def _manticore_table_options_label(cfg: AppConfig) -> str:
    parts: list[str] = []
    profile = str(cfg.manticore.table_profile or "").strip()
    if profile:
        parts.append(f"profile:{profile}")
    opts = (cfg.manticore.fulltext_options or "").strip()
    if opts:
        parts.append(f"opts:{opts}")
    return " ".join(parts)


def _model_opts_label(cfg: AppConfig, engine: str, mode: str) -> str:
    model = resolved_embedding_model(cfg)
    if mode == "vector":
        if engine == "meilisearch":
            extra = [f"embedder:{cfg.meilisearch.embedder_name}"]
            if bool(cfg.meilisearch.binary_quantized):
                extra.append("q=1bit")
            return f"{model} [{' '.join(extra)}]"
        if engine == "opensearch":
            return f"{model} [field:{cfg.opensearch.vector_field}]"
        if engine == "elasticsearch":
            return f"{model} [field:{cfg.elasticsearch.vector_field}]"
        if engine == "typesense":
            return f"{model} [field:{cfg.typesense.vector_field}]"
        if engine == "milvus":
            return f"{model} [field:{cfg.milvus.dense_vector_field} index:{cfg.milvus.dense_index_type} metric:{cfg.milvus.dense_metric_type}]"
        if engine == "mariadb":
            return f"{model} [field:{cfg.mariadb.vector_field} distance:{cfg.mariadb.vector_distance} M:{int(cfg.mariadb.vector_m)}]"
        if engine == "serenedb":
            return f"{model} [field:{cfg.serenedb.vector_field} index:{cfg.serenedb.vector_index_type} metric:{cfg.serenedb.vector_metric} quant:{cfg.serenedb.vector_quantization or 'none'}]"
        if engine == "weaviate":
            return f"{model} [field:{cfg.weaviate.vector_field}]"
        if engine == "vespa":
            return f"{model} [field:{cfg.vespa.vector_field}]"
        return model
    if mode == "hybrid":
        if is_manticore_like_type(engine):
            return _manticore_table_options_label(cfg)
        if engine == "meilisearch":
            extra = [f"embedder:{cfg.meilisearch.embedder_name}", f"semanticRatio={float(cfg.meilisearch.hybrid_semantic_ratio):.2f}"]
            if bool(cfg.meilisearch.binary_quantized):
                extra.append("q=1bit")
            return f"{model} [{' '.join(extra)}]"
        if engine == "opensearch":
            return (
                f"{model} [native-rrf k={int(cfg.opensearch.hybrid_rrf_k)} "
                f"weights={float(cfg.opensearch.hybrid_text_boost):g}:"
                f"{float(cfg.opensearch.hybrid_vector_boost):g} "
                f"candidates={int(cfg.opensearch.hybrid_candidates)}]"
            )
        if engine == "elasticsearch":
            return f"{model} [textBoost:{float(cfg.elasticsearch.hybrid_text_boost):.2f} vectorBoost:{float(cfg.elasticsearch.hybrid_vector_boost):.2f}]"
        if engine == "typesense":
            return f"{model} [alpha:{float(cfg.typesense.hybrid_alpha):.2f}]"
        if engine == "milvus":
            return f"{model} [fusion=rrf k={int(cfg.milvus.hybrid_rrf_k)} candidates={int(cfg.milvus.hybrid_candidates)} sparse=BM25]"
        if engine == "mariadb":
            return f"{model} [rrf:k={int(cfg.mariadb.hybrid_rrf_k)} alpha:{float(cfg.mariadb.hybrid_alpha):.2f}]"
        if engine == "serenedb":
            return f"{model} [rrf:k={int(cfg.serenedb.hybrid_rrf_k)} alpha:{float(cfg.serenedb.hybrid_alpha):.2f}]"
        if engine == "weaviate":
            return f"{model} [alpha:{float(cfg.weaviate.hybrid_alpha):.2f}]"
        if engine == "vespa":
            return f"{model} [alpha:{float(cfg.vespa.hybrid_alpha):.2f}]"
        if engine == "qdrant":
            return (
                f"{model} [native-rrf k={int(cfg.qdrant.hybrid_rrf_k)} "
                f"weights={float(cfg.qdrant.hybrid_sparse_weight):g}:"
                f"{float(cfg.qdrant.hybrid_dense_weight):g} "
                f"candidates={int(cfg.qdrant.hybrid_candidates)}]"
            )
        return f"{model} [fusion=rrf]"
    if is_manticore_like_type(engine):
        return _manticore_table_options_label(cfg) or "(none)"
    if engine == "meilisearch":
        extras: list[str] = []
        if cfg.meilisearch.searchable_attributes:
            extras.append(f"searchable:{','.join(cfg.meilisearch.searchable_attributes)}")
        if cfg.meilisearch.ranking_rules:
            extras.append(f"rules:{','.join(cfg.meilisearch.ranking_rules)}")
        if not bool(cfg.meilisearch.typo_tolerance):
            extras.append("typo:off")
        return " ".join(extras) or "(meilisearch lexical)"
    if engine == "opensearch":
        return f"fields:{','.join(cfg.opensearch.text_fields or ['title', 'abstract'])}"
    if engine == "elasticsearch":
        return f"fields:{','.join(cfg.elasticsearch.text_fields or ['title', 'abstract'])}"
    if engine == "typesense":
        return f"fields:{','.join(cfg.typesense.text_fields or ['title', 'abstract'])}"
    if engine == "milvus":
        return f"BM25 field:{cfg.milvus.content_field} source:{'+'.join(cfg.milvus.text_fields or ['title', 'abstract'])}"
    if engine == "mariadb":
        return f"MATCH/AGAINST mode:{cfg.mariadb.fulltext_search_mode} fields:{','.join(cfg.mariadb.text_fields or ['title', 'abstract'])}"
    if engine == "serenedb":
        return f"BM25 fields:{','.join(cfg.serenedb.text_fields or ['title', 'abstract'])}"
    if engine == "weaviate":
        return f"fields:{','.join(cfg.weaviate.text_fields or ['title', 'abstract'])}"
    if engine == "vespa":
        return f"fields:{','.join(cfg.vespa.text_fields or ['title', 'abstract'])}"
    return "Qdrant/bm25"


def _index_target_name(cfg: AppConfig, engine: str, mode: str) -> str:
    if is_manticore_like_type(engine):
        return cfg.manticore.fulltext_table if mode in ("fulltext", "hybrid") else cfg.manticore.vector_table
    if engine == "meilisearch":
        base = str(cfg.meilisearch.index_name or "").strip()
        return base
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
    return cfg.qdrant.bm25_collection if mode == "fulltext" else cfg.qdrant.dense_collection


def _show_hybrid_index_target(cfg: AppConfig, engine: str) -> bool:
    if is_manticore_like_type(engine):
        # Native Manticore hybrid indexing only makes sense for explicit profiles that
        # point both full-text and vector settings at the same combined table.
        # The plain base Manticore config normally has different full-text/vector tables,
        # which should not synthesize a fake hybrid row in `bench index`.
        return False
    if engine == "qdrant":
        return (
            str(cfg.qdrant.index_layout or "").strip().lower() == "combined"
            and str(cfg.qdrant.bm25_collection) == str(cfg.qdrant.dense_collection)
        )
    hybrid_target = _index_target_name(cfg, engine, "hybrid")
    vector_target = _index_target_name(cfg, engine, "vector")
    return bool(hybrid_target) and hybrid_target != vector_target


def _preferred_layout_mode(cfg: AppConfig, engine: str) -> str | None:
    if is_manticore_like_type(engine) and str(cfg.manticore.index_layout or "").strip().lower() == "combined":
        return "hybrid"
    if (
        engine == "qdrant"
        and str(cfg.qdrant.index_layout or "").strip().lower() == "combined"
        and str(cfg.qdrant.bm25_collection) == str(cfg.qdrant.dense_collection)
    ):
        return "hybrid"
    if engine == "meilisearch" and str(cfg.meilisearch.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "opensearch" and str(cfg.opensearch.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "elasticsearch" and str(cfg.elasticsearch.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "typesense" and str(cfg.typesense.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "milvus" and str(cfg.milvus.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "mariadb" and str(cfg.mariadb.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "serenedb" and str(cfg.serenedb.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "weaviate" and str(cfg.weaviate.index_layout or "").strip().lower() == "combined":
        return "vector"
    if engine == "vespa" and str(cfg.vespa.index_layout or "").strip().lower() == "combined":
        return "vector"
    return None


def _shares_fulltext_and_vector_target(cfg: AppConfig, engine: str) -> bool:
    if _preferred_layout_mode(cfg, engine) is None:
        return False
    fulltext_target = _index_target_name(cfg, engine, "fulltext")
    vector_target = _index_target_name(cfg, engine, "vector")
    return bool(fulltext_target) and fulltext_target == vector_target


def build_index_variants_from_config_paths(paths: list[Path]) -> list[IndexVariant]:
    raw_rows: list[IndexVariant] = []
    for path in paths:
        try:
            targets = load_engine_targets(path)
            visible_names = _visible_profile_names(path, targets)
        except Exception:
            continue
        for target_cfg in targets:
            if target_cfg.name not in visible_names:
                continue
            cfg = target_cfg.cfg
            engine = display_engine_for_profile(target_cfg.type, target_cfg.name)
            show_hybrid = _show_hybrid_index_target(cfg, engine)
            if (
                target_cfg.mode_filter == "hybrid"
                and is_manticore_like_type(engine)
                and str(cfg.manticore.fulltext_table) == str(cfg.manticore.vector_table)
            ):
                show_hybrid = True
            if target_cfg.mode_filter in ("fulltext", "vector", "hybrid"):
                if target_cfg.mode_filter == "hybrid" and not show_hybrid:
                    modes = ["vector"]
                else:
                    modes = [target_cfg.mode_filter]
            else:
                modes = ["fulltext", "vector"]
                if show_hybrid:
                    modes.append("hybrid")
            for mode in modes:
                target = _index_target_name(cfg, engine, mode)
                raw_rows.append(
                    IndexVariant(
                        config_path=str(path),
                        entry=target_cfg.name,
                        base_entry=target_cfg.base_name,
                        engine=engine,
                        mode=mode,
                        model=_model_opts_label(cfg, engine=engine, mode=mode),
                        batch_size=(
                            cfg.manticore.batch_size
                            if is_manticore_like_type(engine)
                            else (
                                cfg.meilisearch.batch_size
                                if engine == "meilisearch"
                                else (
                                    cfg.opensearch.batch_size
                                    if engine == "opensearch"
                                    else (
                                        cfg.elasticsearch.batch_size
                                        if engine == "elasticsearch"
                                        else (
                                            cfg.typesense.batch_size
                                            if engine == "typesense"
                                            else (
                                                cfg.milvus.batch_size
                                                if engine == "milvus"
                                                else (
                                                    cfg.mariadb.batch_size
                                                    if engine == "mariadb"
                                                    else (cfg.serenedb.batch_size if engine == "serenedb" else (cfg.weaviate.batch_size if engine == "weaviate" else (cfg.vespa.batch_size if engine == "vespa" else cfg.qdrant.batch_size)))
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        target=target,
                        cfg=cfg,
                    )
                )
    grouped: dict[tuple[str, str, str], list[IndexVariant]] = {}
    for v in raw_rows:
        build_kind = "fulltext" if v.mode == "fulltext" else "vector"
        grouped.setdefault((v.engine, v.target, build_kind), []).append(v)

    def _preferred_mode(rows: list[IndexVariant]) -> str:
        modes = {r.mode for r in rows}
        layout_mode = _preferred_layout_mode(rows[0].cfg, rows[0].engine)
        if layout_mode is not None and layout_mode in modes:
            return layout_mode
        if "hybrid" in modes and is_manticore_like_type(rows[0].engine):
            return "hybrid"
        if "vector" in modes:
            return "vector"
        return "fulltext"

    def _pick_repr(rows: list[IndexVariant], preferred_mode: str) -> IndexVariant:
        exact = [r for r in rows if r.mode == preferred_mode]
        pool = exact or rows
        pool.sort(key=lambda r: (r.entry != f"{r.target}_{preferred_mode}", r.entry))
        return pool[0]

    out: list[IndexVariant] = []
    for (_, _, build_kind), rows in grouped.items():
        preferred_mode = _preferred_mode(rows)
        rep = _pick_repr(rows, preferred_mode)
        used_by = tuple(sorted(r.entry for r in rows))
        target_kind = (
            "hybrid"
            if build_kind == "vector"
            and any(_shares_fulltext_and_vector_target(r.cfg, r.engine) for r in rows)
            else build_kind
        )
        out.append(
            IndexVariant(
                config_path=rep.config_path,
                entry=rep.entry,
                base_entry=rep.base_entry,
                engine=rep.engine,
                mode=preferred_mode,
                model=rep.model,
                batch_size=rep.batch_size,
                target=rep.target,
                cfg=rep.cfg,
                target_kind=target_kind,
                used_by=used_by,
            )
        )

    engine_order = {"manticore": 0, "manticore_avgdl": 1, "manticore_default_ranker": 2, "manticore_all_default": 3, "manticore_full_default": 4, "columnar": 5, "qdrant": 6, "meilisearch": 7, "opensearch": 8, "elasticsearch": 9, "typesense": 10, "milvus": 11, "mariadb": 12, "serenedb": 13, "weaviate": 14, "vespa": 15}
    kind_order = {"fulltext": 0, "hybrid": 1, "vector": 2, "combined": 3}
    return sorted(
        out,
        key=lambda v: (engine_order.get(v.engine, 99), kind_order.get(v.target_kind or "", 99), v.target, v.entry),
    )


def _expected_doc_count(v: IndexVariant) -> int | None:
    manifest_path = Path(v.cfg.dataset.prepared_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    count = payload.get("doc_count")
    try:
        return int(count) if count is not None else None
    except Exception:
        return None


def _resolve_target_status(v: IndexVariant) -> TargetStatus:
    expected = _expected_doc_count(v)
    try:
        if is_manticore_like_type(v.engine):
            client = ManticoreClient(v.cfg.manticore)
            try:
                if not client.table_exists(v.target):
                    return TargetStatus("pending", None, expected)
                actual = client.count_rows(v.target)
            finally:
                client.close()
        elif v.engine == "qdrant":
            backend = QdrantBackend(v.cfg.qdrant)
            required_mode = (
                "fulltext"
                if v.mode == "fulltext"
                else "hybrid"
                if v.mode == "hybrid"
                else "vector"
            )
            if not backend.collection_supports(v.target, required_mode):
                return TargetStatus("pending", None, expected)
            actual = int(backend._count_points(v.target))
        elif v.engine == "opensearch":
            backend = OpenSearchBackend(v.cfg.opensearch)
            backend.ensure_accessible()
            if not backend.index_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
        elif v.engine == "elasticsearch":
            backend = ElasticsearchBackend(v.cfg.elasticsearch)
            backend.ensure_accessible()
            if not backend.index_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
        elif v.engine == "typesense":
            backend = TypesenseBackend(v.cfg.typesense)
            backend.ensure_accessible()
            if not backend.collection_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
        elif v.engine == "milvus":
            backend = MilvusBackend(v.cfg.milvus)
            try:
                backend.ensure_accessible()
                required_mode = "vector" if _is_vector_capable_target_kind(v.target_kind) else "fulltext"
                if not backend.collection_supports(required_mode):
                    return TargetStatus("pending", None, expected)
                actual = int(backend.get_document_count(v.target))
            finally:
                backend.close()
        elif v.engine == "mariadb":
            backend = MariaDBBackend(v.cfg.mariadb)
            try:
                backend.ensure_accessible()
                if not backend.table_exists(v.target):
                    return TargetStatus("pending", None, expected)
                if not backend.table_has_fulltext(v.target):
                    return TargetStatus("pending", None, expected)
                if _is_vector_capable_target_kind(v.target_kind) and not backend.table_has_vector(v.target):
                    return TargetStatus("pending", None, expected)
                actual = int(backend.get_document_count(v.target))
            finally:
                backend.close()
        elif v.engine == "serenedb":
            backend = SereneDBBackend(v.cfg.serenedb)
            try:
                backend.ensure_accessible()
                if not backend.table_exists(v.target) or not backend.table_has_fulltext(v.target):
                    return TargetStatus("pending", None, expected)
                if _is_vector_capable_target_kind(v.target_kind) and not backend.table_has_vector(v.target):
                    return TargetStatus("pending", None, expected)
                actual = int(backend.get_document_count(v.target))
            finally:
                backend.close()
        elif v.engine == "weaviate":
            backend = WeaviateBackend(v.cfg.weaviate)
            backend.ensure_accessible()
            if not backend.collection_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
        elif v.engine == "vespa":
            backend = VespaBackend(v.cfg.vespa)
            backend.ensure_accessible()
            if not backend.collection_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
        else:
            backend = MeilisearchBackend(v.cfg.meilisearch)
            backend.ensure_accessible()
            if not backend.index_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
    except Exception as exc:
        return TargetStatus("unavailable", None, expected, str(exc))
    if expected is not None and actual != expected:
        return TargetStatus("unexpected", actual, expected)
    return TargetStatus("ok", actual, expected)


def _draw_row(
    stdscr: Any,
    y: int,
    v: IndexVariant,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    row_lines: int,
    target_status: TargetStatus,
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    color_engine = 0
    color_mode = 0
    color_target = 0
    if curses.has_colors():
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
            color_engine = curses.color_pair(10)
        elif v.engine == "serenedb":
            color_engine = curses.color_pair(10)
        elif v.engine == "weaviate":
            color_engine = curses.color_pair(13)
        elif v.engine == "vespa":
            color_engine = curses.color_pair(14)
        elif v.engine == "meilisearch":
            color_engine = curses.color_pair(5)
        displayed_kind = v.target_kind or v.mode
        if displayed_kind == "fulltext":
            color_mode = curses.color_pair(3)
        elif displayed_kind == "vector":
            color_mode = curses.color_pair(4)
        elif displayed_kind in ("hybrid", "combined", "both"):
            color_mode = curses.color_pair(5)
        if target_status.state == "pending":
            color_target = curses.color_pair(7)
        elif target_status.state == "unexpected":
            color_target = curses.color_pair(8)

    columns: list[tuple[str, int, int]] = [
        ("[x]" if selected else "[ ]", 0, widths["sel"]),
        (v.engine, attrs | color_engine, widths["engine"]),
        (v.target, attrs | color_target, widths["target"]),
        (_display_target_kind(v.target_kind or v.mode), attrs | color_mode, widths["kind"]),
        (", ".join(v.used_by or (v.entry,)), attrs, widths["used_by"]),
        (v.model, attrs, widths["model"]),
        (str(v.batch_size), attrs, widths["batch"]),
    ]
    x = 0
    for text, cell_attrs, width in columns:
        cell_lines = wrap_cell(text, width, max_lines=row_lines)
        for i in range(row_lines):
            stdscr.addstr(y + i, x, cell_lines[i], cell_attrs)
        x += width + 1


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


def _variant_detail_values(v: IndexVariant) -> dict[str, str]:
    visible_sections = {"dataset", "benchmark", "metrics", v.engine}
    cfg_obj = asdict(v.cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    flattened: dict[str, str] = {}
    _flatten_detail_values("", filtered_cfg, flattened)
    return flattened


def _comparison_variants(variants: list[IndexVariant], current: IndexVariant) -> list[IndexVariant]:
    same_engine = [v for v in variants if v.engine == current.engine]
    return same_engine or [current]


def _common_detail_keys(variants: list[IndexVariant], current: IndexVariant) -> set[str]:
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
    v: IndexVariant,
    *,
    hidden_common_keys: set[str] | None = None,
    comparison_scope: str | None = None,
    target_status: TargetStatus | None = None,
) -> list[str]:
    hidden_common_keys = hidden_common_keys or set()

    lines: list[str] = [
        "Resolved Config",
        f"index action entry: {v.entry}",
        f"engine: {v.engine}",
        f"index mode: {_display_mode(v.mode)}",
        f"target: {v.target}",
        f"target kind: {_display_target_kind(v.target_kind or v.mode)}",
        f"used by: {', '.join(v.used_by or (v.entry,))}",
    ]
    if target_status is not None:
        state_label = {
            "ok": "indexed",
            "pending": "pending indexing",
            "unexpected": "row count mismatch",
        }.get(target_status.state, target_status.state)
        if target_status.actual_count is not None and target_status.expected_count is not None:
            lines.append(f"target status: {state_label} ({target_status.actual_count:,}/{target_status.expected_count:,})")
        elif target_status.actual_count is not None:
            lines.append(f"target status: {state_label} ({target_status.actual_count:,})")
        elif target_status.expected_count is not None:
            lines.append(f"target status: {state_label} (expected {target_status.expected_count:,})")
        else:
            lines.append(f"target status: {state_label}")
        if target_status.error:
            lines.append(f"target error: {target_status.error}")
    if hidden_common_keys:
        scope = comparison_scope or "visible rows"
        lines.append(f"common across {scope}: {len(hidden_common_keys)} hidden")
    flattened = _variant_detail_values(v)
    lines.extend(
        f"{key}: {value}"
        for key, value in flattened.items()
        if key not in hidden_common_keys
    )
    return lines


def _draw_details_panel(
    stdscr: Any,
    v: IndexVariant,
    variants: list[IndexVariant],
    widths: dict[str, int],
    w: int,
    h: int,
    target_status: TargetStatus,
) -> None:
    table_width = sum(widths.values()) + (len(widths) - 1)
    panel_x = table_width + 3
    panel_w = w - panel_x - 1
    if panel_w < 24:
        return
    comparison_variants = _comparison_variants(variants, v)
    lines = _details_lines(
        v,
        hidden_common_keys=_common_detail_keys(variants, v),
        comparison_scope=f"visible {v.engine} rows ({len(comparison_variants)})",
        target_status=target_status,
    )
    max_rows = max(0, h - 3)
    y = 1
    for i, line in enumerate(lines):
        if y > max_rows:
            break
        attrs = curses.A_BOLD if i == 0 else curses.A_DIM
        wrapped = wrap_text_lines(line, panel_w)
        for j, part in enumerate(wrapped):
            if y > max_rows:
                break
            part_attrs = attrs if j == 0 else curses.A_DIM
            stdscr.addstr(y, panel_x, _fit(part, panel_w), part_attrs)
            y += 1


def _toggle_index_selection(
    variants: list[IndexVariant], selected_idx: set[int], current: int
) -> None:
    if current in selected_idx:
        selected_idx.remove(current)
        return

    candidate = variants[current]
    candidate_kind = candidate.target_kind or candidate.mode
    if candidate_kind in ("fulltext", "hybrid"):
        opposite_kind = "hybrid" if candidate_kind == "fulltext" else "fulltext"
        conflicts = {
            i
            for i in selected_idx
            if variants[i].engine == candidate.engine
            and variants[i].target == candidate.target
            and (variants[i].target_kind or variants[i].mode) == opposite_kind
        }
        selected_idx.difference_update(conflicts)
    selected_idx.add(current)


def _select_index_variants_curses(variants: list[IndexVariant]) -> list[IndexVariant]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    all_variants = list(variants)

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset, variants, selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # manticore
            curses.init_pair(15, curses.COLOR_WHITE, -1)  # columnar
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)    # full-text
            curses.init_pair(4, curses.COLOR_WHITE, -1)   # vector
            curses.init_pair(5, curses.COLOR_CYAN, -1)    # meilisearch
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # hybrid / milvus / manticore avgdl
            curses.init_pair(9, curses.COLOR_MAGENTA, -1) # opensearch
            curses.init_pair(12, curses.COLOR_BLUE, -1)   # elasticsearch
            curses.init_pair(11, curses.COLOR_RED, -1)    # typesense
            curses.init_pair(10, curses.COLOR_YELLOW, -1) # mariadb
            curses.init_pair(13, curses.COLOR_CYAN, -1)   # weaviate
            curses.init_pair(14, curses.COLOR_GREEN, -1)  # vespa
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_RED)     # target pending
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # target unexpected

        preferred_widths = {
            "sel": 3,
            "engine": 10,
            "target": 22,
            "kind": 10,
            "used_by": 38,
            "model": 38,
            "batch": 8,
        }
        minimum_widths = {
            "sel": 3,
            "engine": 8,
            "target": 12,
            "kind": 7,
            "used_by": 12,
            "model": 12,
            "batch": 5,
        }
        sort_cols = ["engine", "target", "kind", "used_by", "model", "batch"]
        sort_idx = 0
        sort_key = sort_cols[sort_idx]
        sort_desc = False
        filters: dict[str, str] = {}
        status_cache: dict[tuple[str, str, str, str, str, int], TargetStatus] = {}

        def _variant_id(v: IndexVariant) -> tuple[str, str, str, str, str, int]:
            return (v.entry, v.engine, v.mode, v.target, v.model, int(v.batch_size))

        def _get_status(v: IndexVariant) -> TargetStatus:
            key = _variant_id(v)
            if key not in status_cache:
                status_cache[key] = _resolve_target_status(v)
            return status_cache[key]

        def _mode_ord(mode: str) -> int:
            return 0 if mode == "fulltext" else (1 if mode == "vector" else 99)

        def _sort_value(v: IndexVariant, col: str) -> Any:
            if col == "engine":
                return v.engine.lower()
            if col == "target":
                return v.target.lower()
            if col == "kind":
                return {
                    "fulltext": 0,
                    "hybrid": 1,
                    "vector": 2,
                    "combined": 3,
                }.get(v.target_kind or v.mode, 99)
            if col == "used_by":
                return ",".join(v.used_by).lower()
            if col == "model":
                return v.model.lower()
            if col == "batch":
                return int(v.batch_size)
            return v.target.lower()

        def _filter_value(v: IndexVariant, col: str) -> Any:
            if col == "engine":
                return v.engine
            if col == "target":
                return v.target
            if col == "kind":
                return _display_target_kind(v.target_kind or v.mode)
            if col == "used_by":
                return ", ".join(v.used_by or (v.entry,))
            if col == "model":
                return v.model
            if col == "batch":
                return v.batch_size
            return ""

        def _apply_sort(*, preserve_current_item: bool = True) -> None:
            nonlocal variants, selected_idx, current, offset
            selected_ids = {_variant_id(variants[i]) for i in selected_idx if 0 <= i < len(variants)}
            current_id = _variant_id(variants[current]) if preserve_current_item and 0 <= current < len(variants) else None
            variants = [
                v
                for v in filter_rows(all_variants, filters, _filter_value)
                if _get_status(v).state != "unavailable"
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
            stdscr.addstr(
                0,
                0,
                fit_text("↑/↓ move  ←/→ sort col  SPACE toggle  f filter  ENTER index selected  ESC/q back", max(1, w - 1)),
                curses.A_BOLD,
            )
            filter_lines = filter_block_height(filters, widths)
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[
                    ("SEL", "sel"),
                    ("ENGINE", "engine"),
                    ("TARGET", "target"),
                    ("KIND", "kind"),
                    ("USED BY", "used_by"),
                    ("MODEL/OPTS", "model"),
                    ("BATCH", "batch"),
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
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    row_lines=row_h,
                    target_status=_get_status(variants[ridx]),
                )
            if not variants:
                message = (
                    "No index targets match the active filters."
                    if filters
                    else "No available index targets. Start an engine from `bench engines`."
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
                    target_status=_get_status(variants[current]),
                )
            legend_y = h - 2
            if legend_y >= rows_y + visible * row_h:
                if curses.has_colors():
                    x = 0
                    stdscr.addstr(legend_y, x, "TARGET:", curses.A_BOLD)
                    x += len("TARGET: ")
                    stdscr.addstr(legend_y, x, "pending", curses.color_pair(7))
                    x += len("pending")
                    stdscr.addstr(legend_y, x, " = missing/not built, ", curses.A_DIM)
                    x += len(" = missing/not built, ")
                    stdscr.addstr(legend_y, x, "unexpected", curses.color_pair(8))
                    x += len("unexpected")
                    stdscr.addstr(legend_y, x, " = count mismatch", curses.A_DIM)
                else:
                    stdscr.addstr(
                        legend_y,
                        0,
                        fit_text("TARGET: pending = missing/not built, unexpected = count mismatch", max(1, w - 1)),
                        curses.A_DIM,
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
                    _toggle_index_selection(variants, selected_idx, current)
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


def select_index_variants_interactive(variants: list[IndexVariant]) -> list[IndexVariant]:
    if not variants:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_index_variants_curses(variants)
        except Exception:
            pass
    print("Available index runs:")
    for i, v in enumerate(variants, start=1):
        print(
            f"{i:>2}. {v.entry}\t{v.engine}\t{_display_mode(v.mode)}\t"
            f"target={v.target}\tmodel={v.model}\tbatch={v.batch_size}"
        )
    raw = input("Select numbers (e.g. 1,3-5): ").strip()
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
