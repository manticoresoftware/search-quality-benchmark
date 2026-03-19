from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    source: str = "beir/trec-covid"
    prepared_dir: str = "data/prepared"


@dataclass
class ManticoreConfig:
    host: str = "127.0.0.1"
    sql_port: int = 9306
    http_port: int = 9308
    user: str = ""
    password: str = ""
    fulltext_table: str = "trec_covid_ft"
    vector_table: str = "trec_covid_vec"
    vector_field: str = "embedding"
    vector_model: str = "openai/text-embedding-3-large"
    vector_from: str = "title,abstract"
    vector_similarity: str = "cosine"
    vector_quantization: str = ""
    vector_search_oversampling: float | None = None
    vector_search_rescore: int | None = None
    vector_cache_path: str = "vector_cache/embeddings.sqlite"
    vector_cache_fetch_batch_size: int = 1000
    vector_api_key: str = ""
    vector_api_key_file: str = ""
    search_options: str = "ranker=expr('1000*(bm25a(1.2,0.75)+sum(lcs*user_weight))')"
    fulltext_options: str = "morphology='stem_en' index_field_lengths='1'"
    batch_size: int = 500
    insert_retries: int = 3
    connect_timeout_seconds: int = 10
    query_timeout_seconds: int = 60


@dataclass
class BenchmarkConfig:
    limit: int = 100
    warmup_queries: int = 5
    repeats: int = 1
    concurrency: int = 4
    timeout_seconds: int = 60
    diagnostics_queries: int = 3


@dataclass
class QdrantConfig:
    path: str = ""
    host: str = "127.0.0.1"
    port: int = 6333
    api_key: str = ""
    https: bool = False
    timeout_seconds: int = 60
    batch_size: int = 2000
    dense_collection: str = "trec_covid_qdrant_dense"
    bm25_collection: str = "trec_covid_qdrant_bm25"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    dense_binary_quantization: bool = False
    dense_binary_always_ram: bool = True
    dense_binary_query_encoding: str = "default"
    dense_search_quantization_rescore: bool | None = None
    dense_search_quantization_oversampling: float | None = None
    dense_search_quantization_ignore: bool = False
    # Deprecated/ignored with native Qdrant BM25, kept for backward compatibility.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


@dataclass
class MetricsConfig:
    enabled: list[str] = field(
        default_factory=lambda: [
            "ndcg@10",
            "ndcg@20",
            "map@10",
            "p@20",
            "recall@100",
            "latency",
            "qps",
            "error_rate",
        ]
    )
    ndcg_ks: list[int] = field(default_factory=lambda: [10, 20])
    map_k: int = 10
    precision_k: int = 20
    recall_k: int = 100


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    manticore: ManticoreConfig = field(default_factory=ManticoreConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class EngineTarget:
    source_path: str
    name: str
    type: str
    mode_filter: str | None
    cfg: AppConfig


_MANTICORE_FIELDS = set(vars(ManticoreConfig()).keys())
_QDRANT_FIELDS = set(vars(QdrantConfig()).keys())
_BENCHMARK_FIELDS = set(vars(BenchmarkConfig()).keys())
_DATASET_FIELDS = set(vars(DatasetConfig()).keys())
_METRICS_FIELDS = set(vars(MetricsConfig()).keys())
_SECTION_NAMES = {"manticore", "qdrant", "benchmark", "dataset", "metrics"}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _defaults_dict() -> dict[str, Any]:
    defaults = AppConfig()
    return {
        "dataset": vars(defaults.dataset),
        "manticore": vars(defaults.manticore),
        "qdrant": vars(defaults.qdrant),
        "benchmark": vars(defaults.benchmark),
        "metrics": vars(defaults.metrics),
    }


def _to_app_config(raw: dict[str, Any]) -> AppConfig:
    merged = _deep_update(_defaults_dict(), raw)
    # Backward compatibility for old configs: ignore removed single-cutoff ndcg_k.
    if isinstance(merged.get("metrics"), dict):
        merged["metrics"].pop("ndcg_k", None)
    # Backward compatibility: index_field_lengths -> fulltext_options
    if isinstance(merged.get("manticore"), dict):
        m = merged["manticore"]
        if "fulltext_options" not in m and "index_field_lengths" in m:
            try:
                on = 1 if int(m.get("index_field_lengths", 1)) > 0 else 0
            except Exception:
                on = 1
            m["fulltext_options"] = f"morphology='stem_en' index_field_lengths='{on}'"
        m.pop("index_field_lengths", None)
        # Backward compatibility: fulltext_ranker/fulltext_ranker_expr -> search_options
        ranker = str(m.pop("fulltext_ranker", "") or "").strip()
        ranker_expr = str(m.pop("fulltext_ranker_expr", "") or "").strip()
        search_opts = str(m.get("search_options", "") or "").strip()
        if ranker and "ranker=" not in search_opts.lower():
            if ranker.lower() == "expr":
                expr = ranker_expr or "1000*(bm25a(1.2,0.75)+sum(lcs*user_weight))"
                ranker_opt = f"ranker=expr('{expr}')"
            else:
                ranker_opt = f"ranker={ranker}"
            if search_opts:
                m["search_options"] = f"{ranker_opt}, {search_opts}"
            else:
                m["search_options"] = ranker_opt
    return AppConfig(
        dataset=DatasetConfig(**merged["dataset"]),
        manticore=ManticoreConfig(**merged["manticore"]),
        qdrant=QdrantConfig(**merged["qdrant"]),
        benchmark=BenchmarkConfig(**merged["benchmark"]),
        metrics=MetricsConfig(**merged["metrics"]),
    )


def _raw_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text()) or {}
    return {}


def load_engine_targets(path: str | Path) -> list[EngineTarget]:
    cfg_path = Path(path)
    raw = _raw_config(cfg_path)
    base_raw = dict(raw)
    global_sections: dict[str, Any] = {}
    for sec in ("dataset", "benchmark", "metrics"):
        val = base_raw.pop(sec, None)
        if isinstance(val, dict):
            global_sections[sec] = val
    explicit_entry_mode = any(
        isinstance(v, dict) and ("type" in v or "base" in v) for v in base_raw.values()
    )
    # legacy top-level sections mode (manticore/qdrant as plain sections)
    legacy_m = base_raw.get("manticore")
    legacy_q = base_raw.get("qdrant")
    if not explicit_entry_mode and (
        isinstance(legacy_m, dict) or isinstance(legacy_q, dict)
    ):
        merged_legacy = _deep_update(_deep_update({}, global_sections), {})
        if isinstance(legacy_m, dict):
            merged_legacy["manticore"] = legacy_m
        if isinstance(legacy_q, dict):
            merged_legacy["qdrant"] = legacy_q
        cfg = _to_app_config(merged_legacy)
        return [
            EngineTarget(source_path=str(cfg_path), name="manticore", type="manticore", mode_filter=None, cfg=cfg),
            EngineTarget(source_path=str(cfg_path), name="qdrant", type="qdrant", mode_filter=None, cfg=cfg),
        ]

    entry_defs: dict[str, dict[str, Any]] = {
        k: v for k, v in base_raw.items() if isinstance(v, dict)
    }
    if not entry_defs:
        cfg = _to_app_config(global_sections)
        return [
            EngineTarget(source_path=str(cfg_path), name="manticore", type="manticore", mode_filter=None, cfg=cfg),
            EngineTarget(source_path=str(cfg_path), name="qdrant", type="qdrant", mode_filter=None, cfg=cfg),
        ]

    resolved: dict[str, dict[str, Any]] = {}

    def _resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        ent = dict(entry_defs.get(name, {}))
        base_name = str(ent.pop("base", "")).strip()
        merged: dict[str, Any] = {}
        if base_name:
            merged = _deep_update(merged, _resolve(base_name))
        merged = _deep_update(merged, ent)
        resolved[name] = merged
        return merged

    def _infer_type(name: str, ent: dict[str, Any]) -> str:
        e_type = str(ent.get("type", "")).strip().lower()
        if e_type in ("manticore", "qdrant"):
            return e_type
        if isinstance(ent.get("manticore"), dict) and not isinstance(ent.get("qdrant"), dict):
            return "manticore"
        if isinstance(ent.get("qdrant"), dict) and not isinstance(ent.get("manticore"), dict):
            return "qdrant"
        m_hits = 0
        q_hits = 0
        for k in ent.keys():
            if k in ("type", "base") or k in _SECTION_NAMES:
                continue
            if k in _MANTICORE_FIELDS:
                m_hits += 1
            if k in _QDRANT_FIELDS:
                q_hits += 1
        if q_hits > m_hits:
            return "qdrant"
        if m_hits > q_hits:
            return "manticore"
        if name.lower().startswith("qdrant"):
            return "qdrant"
        return "manticore"

    def _infer_mode_filter(ent: dict[str, Any]) -> str | None:
        raw = str(ent.get("type", "")).strip().lower()
        if raw in ("vector",):
            return "vector"
        if raw in ("fulltext", "full-text"):
            return "fulltext"
        return None

    def _entry_to_overrides(ent: dict[str, Any], e_type: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sec in _SECTION_NAMES:
            sec_val = ent.get(sec)
            if isinstance(sec_val, dict):
                out[sec] = _deep_update(out.get(sec, {}), sec_val)

        for k, v in ent.items():
            if k in ("type", "base") or k in _SECTION_NAMES:
                continue
            candidates: list[str] = []
            if k in _MANTICORE_FIELDS:
                candidates.append("manticore")
            if k in _QDRANT_FIELDS:
                candidates.append("qdrant")
            if k in _BENCHMARK_FIELDS:
                candidates.append("benchmark")
            if k in _DATASET_FIELDS:
                candidates.append("dataset")
            if k in _METRICS_FIELDS:
                candidates.append("metrics")
            if not candidates:
                continue

            preferred = "manticore" if e_type == "manticore" else "qdrant"
            if preferred in candidates:
                sec = preferred
            elif len(candidates) == 1:
                sec = candidates[0]
            elif "benchmark" in candidates:
                sec = "benchmark"
            elif "dataset" in candidates:
                sec = "dataset"
            elif "metrics" in candidates:
                sec = "metrics"
            else:
                sec = candidates[0]
            out.setdefault(sec, {})[k] = v
        return out

    targets: list[EngineTarget] = []
    for name in sorted(entry_defs.keys()):
        ent = _resolve(name)
        e_type = _infer_type(name, ent)
        mode_filter = _infer_mode_filter(ent)
        overrides = _entry_to_overrides(ent, e_type)
        merged_raw = _deep_update(_deep_update({}, global_sections), overrides)
        cfg = _to_app_config(merged_raw)
        targets.append(
            EngineTarget(
                source_path=str(cfg_path),
                name=name,
                type=e_type,
                mode_filter=mode_filter,
                cfg=cfg,
            )
        )
    return targets


def load_config(path: str | Path) -> AppConfig:
    targets = load_engine_targets(path)
    for t in targets:
        if t.type == "manticore":
            return t.cfg
    return targets[0].cfg
