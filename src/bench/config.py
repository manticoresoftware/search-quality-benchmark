from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    source: str = "beir/trec-covid"
    prepared_dir: str = "data/prepared"
    max_docs: int = 0
    keep_qrels_docs: bool = True


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "openai/text-embedding-3-large"
    api_key: str = ""
    api_key_file: str = ""
    base_url: str = ""
    site_url: str = ""
    app_name: str = "trec-covid-bench"
    price_per_million_input_tokens: float = 0.0
    max_batch_texts: int = 32
    concurrency: int = 4
    retry_attempts: int = 5
    retry_backoff_seconds: float = 1.0
    timeout_seconds: int = 120


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
    vector_cache_path: str = "vector_cache/trec-covid-openai.sqlite"
    vector_cache_fetch_batch_size: int = 1000
    index_layout: str = "combined"
    search_options: str = "ranker=expr('1000*(bm25a(1.2,0.75)+sum(lcs*user_weight))')"
    fulltext_options: str = "morphology='stem_en' index_field_lengths='1'"
    fulltext_query_mode: str = "and"
    fulltext_query_variant: str = "short"
    fulltext_query_quorum: float = 0.5
    batch_size: int = 500
    index_concurrency: int = 1
    insert_retries: int = 3
    connect_timeout_seconds: int = 10
    query_timeout_seconds: int = 60


@dataclass
class MeilisearchConfig:
    host: str = "http://127.0.0.1:7700"
    api_key: str = ""
    api_key_file: str = ""
    index_name: str = "trec_covid"
    index_layout: str = "combined"
    timeout_sec: int = 30
    batch_size: int = 500
    index_concurrency: int = 1
    searchable_attributes: list[str] = field(default_factory=lambda: ["title", "abstract"])
    ranking_rules: list[str] = field(
        default_factory=lambda: ["words", "typo", "proximity", "attribute", "sort", "exactness"]
    )
    stop_words: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    typo_tolerance: bool = True
    embedder_name: str = "default"
    vector_index: str = "vector"
    hybrid_semantic_ratio: float = 0.5
    binary_quantized: bool = False


@dataclass
class OpenSearchConfig:
    url: str = "http://127.0.0.1:9200"
    username: str = ""
    password: str = ""
    api_key: str = ""
    api_key_file: str = ""
    verify_certs: bool = False
    timeout_sec: int = 30
    batch_size: int = 500
    index_concurrency: int = 1
    index_name: str = "trec_covid"
    index_layout: str = "combined"
    text_fields: list[str] = field(default_factory=lambda: ["title", "abstract"])
    vector_index: str = "vector"
    vector_field: str = "embedding"
    space_type: str = "cosinesimil"
    hybrid_text_boost: float = 1.0
    hybrid_vector_boost: float = 1.0


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
    index_concurrency: int = 1
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
            "mrr@10",
            "p@20",
            "recall@100",
            "latency",
            "qps",
            "error_rate",
        ]
    )
    ndcg_ks: list[int] = field(default_factory=lambda: [10, 20])
    map_k: int = 10
    mrr_k: int = 10
    precision_k: int = 20
    recall_k: int = 100


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    manticore: ManticoreConfig = field(default_factory=ManticoreConfig)
    meilisearch: MeilisearchConfig = field(default_factory=MeilisearchConfig)
    opensearch: OpenSearchConfig = field(default_factory=OpenSearchConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class EngineTarget:
    source_path: str
    name: str
    base_name: str
    type: str
    mode_filter: str | None
    cfg: AppConfig


def dataset_source_slug(source: str) -> str:
    raw = str(source or "").strip().lower()
    if not raw:
        return "default"
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "default"


def prepared_dir_for_source(
    base_prepared_dir: str,
    source: str,
    *,
    max_docs: int = 0,
    keep_qrels_docs: bool = True,
) -> str:
    base = Path(str(base_prepared_dir or "data/prepared"))
    slug = dataset_source_slug(source)
    suffix_parts: list[str] = []
    if int(max_docs or 0) > 0:
        suffix_parts.append(f"maxdocs-{int(max_docs)}")
    if not bool(keep_qrels_docs):
        suffix_parts.append("noqrels")
    if suffix_parts:
        slug = f"{slug}--{'-'.join(suffix_parts)}"
    if base.name == slug:
        return str(base)
    return str(base / slug)


def _apply_dataset_source_prepared_dir(cfg: AppConfig) -> AppConfig:
    cfg.dataset.prepared_dir = prepared_dir_for_source(
        cfg.dataset.prepared_dir,
        cfg.dataset.source,
        max_docs=cfg.dataset.max_docs,
        keep_qrels_docs=cfg.dataset.keep_qrels_docs,
    )
    return cfg


_MANTICORE_FIELDS = set(vars(ManticoreConfig()).keys())
_EMBEDDING_FIELDS = set(vars(EmbeddingConfig()).keys())
_MEILISEARCH_FIELDS = set(vars(MeilisearchConfig()).keys())
_OPENSEARCH_FIELDS = set(vars(OpenSearchConfig()).keys())
_QDRANT_FIELDS = set(vars(QdrantConfig()).keys())
_BENCHMARK_FIELDS = set(vars(BenchmarkConfig()).keys())
_DATASET_FIELDS = set(vars(DatasetConfig()).keys())
_METRICS_FIELDS = set(vars(MetricsConfig()).keys())
_SECTION_NAMES = {"embedding", "manticore", "meilisearch", "opensearch", "qdrant", "benchmark", "dataset", "metrics"}


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
        "embedding": vars(defaults.embedding),
        "manticore": vars(defaults.manticore),
        "meilisearch": vars(defaults.meilisearch),
        "opensearch": vars(defaults.opensearch),
        "qdrant": vars(defaults.qdrant),
        "benchmark": vars(defaults.benchmark),
        "metrics": vars(defaults.metrics),
    }


def _to_app_config(raw: dict[str, Any]) -> AppConfig:
    merged = _deep_update(_defaults_dict(), raw)
    return AppConfig(
        dataset=DatasetConfig(**merged["dataset"]),
        embedding=EmbeddingConfig(**merged["embedding"]),
        manticore=ManticoreConfig(**merged["manticore"]),
        meilisearch=MeilisearchConfig(**merged["meilisearch"]),
        opensearch=OpenSearchConfig(**merged["opensearch"]),
        qdrant=QdrantConfig(**merged["qdrant"]),
        benchmark=BenchmarkConfig(**merged["benchmark"]),
        metrics=MetricsConfig(**merged["metrics"]),
    )


def _raw_config(path: str | Path) -> dict[str, Any]:
    return _raw_config_with_includes(path)


def _raw_config_with_includes(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        return {}
    seen = _seen if _seen is not None else set()
    if cfg_path in seen:
        raise ValueError(f"Config include cycle detected at: {cfg_path}")
    seen.add(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    include_paths = raw.pop("include_configs", None)
    merged: dict[str, Any] = {}
    if isinstance(include_paths, list):
        for item in include_paths:
            inc = str(item or "").strip()
            if not inc:
                continue
            inc_path = (cfg_path.parent / inc).resolve()
            merged = _deep_update(merged, _raw_config_with_includes(inc_path, seen))
    elif include_paths is not None:
        raise ValueError(f"include_configs must be a list in config: {cfg_path}")
    merged = _deep_update(merged, raw)
    seen.remove(cfg_path)
    return merged


def load_engine_targets(path: str | Path) -> list[EngineTarget]:
    cfg_path = Path(path)
    raw = _raw_config(cfg_path)
    base_raw = dict(raw)
    global_sections: dict[str, Any] = {}
    for sec in ("dataset", "embedding", "benchmark", "metrics"):
        val = base_raw.pop(sec, None)
        if isinstance(val, dict):
            global_sections[sec] = val
    entry_defs: dict[str, dict[str, Any]] = {
        k: v for k, v in base_raw.items() if isinstance(v, dict)
    }
    if not entry_defs:
        raise ValueError(f"No engine entries found in config: {cfg_path}")

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
        if e_type in ("manticore", "meilisearch", "opensearch", "qdrant"):
            return e_type
        if isinstance(ent.get("manticore"), dict) and not isinstance(ent.get("qdrant"), dict):
            return "manticore"
        if isinstance(ent.get("opensearch"), dict) and not isinstance(ent.get("manticore"), dict) and not isinstance(ent.get("meilisearch"), dict) and not isinstance(ent.get("qdrant"), dict):
            return "opensearch"
        if isinstance(ent.get("meilisearch"), dict) and not isinstance(ent.get("manticore"), dict) and not isinstance(ent.get("qdrant"), dict):
            return "meilisearch"
        if isinstance(ent.get("qdrant"), dict) and not isinstance(ent.get("manticore"), dict):
            return "qdrant"
        m_hits = 0
        meili_hits = 0
        os_hits = 0
        q_hits = 0
        for k in ent.keys():
            if k in ("type", "base") or k in _SECTION_NAMES:
                continue
            if k in _MANTICORE_FIELDS:
                m_hits += 1
            if k in _EMBEDDING_FIELDS:
                pass
            if k in _MEILISEARCH_FIELDS:
                meili_hits += 1
            if k in _OPENSEARCH_FIELDS:
                os_hits += 1
            if k in _QDRANT_FIELDS:
                q_hits += 1
        if meili_hits > max(m_hits, os_hits, q_hits):
            return "meilisearch"
        if os_hits > max(m_hits, q_hits):
            return "opensearch"
        if q_hits > m_hits:
            return "qdrant"
        if m_hits > q_hits:
            return "manticore"
        if name.lower().startswith("meili"):
            return "meilisearch"
        if name.lower().startswith("opensearch") or name.lower().startswith("osearch") or name.lower().startswith("os_"):
            return "opensearch"
        if name.lower().startswith("qdrant"):
            return "qdrant"
        return "manticore"

    def _infer_mode_filter(ent: dict[str, Any]) -> str | None:
        raw = str(ent.get("type", "")).strip().lower()
        if raw in ("vector",):
            return "vector"
        if raw in ("hybrid",):
            return "hybrid"
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
            if k in _EMBEDDING_FIELDS:
                candidates.append("embedding")
            if k in _MEILISEARCH_FIELDS:
                candidates.append("meilisearch")
            if k in _OPENSEARCH_FIELDS:
                candidates.append("opensearch")
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

            preferred = e_type if e_type in ("manticore", "meilisearch", "opensearch", "qdrant") else "manticore"
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
        cfg = _apply_dataset_source_prepared_dir(cfg)
        targets.append(
            EngineTarget(
                source_path=str(cfg_path),
                name=name,
                base_name=str(entry_defs.get(name, {}).get("base", "") or "").strip(),
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
