from __future__ import annotations

from dataclasses import dataclass
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import heapq
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from .config import AppConfig
from .embedding_client import EmbeddingClient, resolved_embedding_model, resolved_embedding_similarity
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl, write_tsv
from .vector_cache import VectorCache, cache_key_from_doc, cache_key_from_text, normalize_text, parse_vector_from_fields


def _doc_attr(doc: Any, names: list[str], default: str = "") -> str:
    for name in names:
        if hasattr(doc, name):
            value = getattr(doc, name)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                return value
    return default


@dataclass
class PrepareResult:
    docs_path: Path
    topics_path: Path
    qrels_path: Path
    manifest_path: Path


@dataclass
class EmbeddingBuildStats:
    doc_total: int
    doc_ok: int
    query_total: int
    query_ok: int
    input_tokens: int
    estimated_cost_usd: float


@dataclass
class EmbeddingResetStats:
    deleted: int
    doc_total: int
    query_total: int


def _manifest_with_embeddings(
    base: dict[str, Any],
    *,
    cfg: AppConfig,
    stats: EmbeddingBuildStats,
) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "embedding_provider": cfg.embedding.provider,
            "embedding_model": resolved_embedding_model(cfg),
            "embedding_cache_path": cfg.manticore.vector_cache_path,
            "doc_embeddings_ok": stats.doc_ok,
            "doc_embeddings_total": stats.doc_total,
            "query_embeddings_ok": stats.query_ok,
            "query_embeddings_total": stats.query_total,
            "input_tokens": stats.input_tokens,
            "estimated_cost_usd": round(stats.estimated_cost_usd, 6),
        }
    )
    return payload


def _write_embedding_manifest(
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    *,
    cfg: AppConfig,
    doc_total: int,
    doc_ok: int,
    query_total: int,
    query_ok: int,
    input_tokens: int,
    estimated_cost_usd: float,
) -> None:
    write_json(
        manifest_path,
        _manifest_with_embeddings(
            manifest_payload,
            cfg=cfg,
            stats=EmbeddingBuildStats(
                doc_total=doc_total,
                doc_ok=doc_ok,
                query_total=query_total,
                query_ok=query_ok,
                input_tokens=input_tokens,
                estimated_cost_usd=estimated_cost_usd,
            ),
        ),
    )


def _populate_embeddings(
    cfg: AppConfig,
    docs: list[dict[str, Any]],
    topics: list[dict[str, str]],
    *,
    progress_callback: callable | None = None,
    manifest_path: Path | None = None,
    manifest_payload: dict[str, Any] | None = None,
) -> EmbeddingBuildStats:
    model = resolved_embedding_model(cfg)
    similarity = resolved_embedding_similarity(cfg)
    fields = parse_vector_from_fields(cfg.manticore.vector_from)
    cache = VectorCache(cfg.manticore.vector_cache_path)
    client = EmbeddingClient(cfg)
    doc_total = len(docs)
    query_total = len(topics)
    doc_ok = 0
    query_ok = 0
    input_tokens = 0
    estimated_cost_usd = 0.0
    max_batch = max(1, int(cfg.embedding.max_batch_texts or 32))
    embed_concurrency = max(1, int(cfg.embedding.concurrency or 1))

    def _emit(stage: str, *, processed: int, total: int, embedded_ok: int, query_phase: bool = False) -> None:
        if manifest_path is not None and manifest_payload is not None:
            _write_embedding_manifest(
                manifest_path,
                manifest_payload,
                cfg=cfg,
                doc_total=doc_total,
                doc_ok=doc_ok,
                query_total=query_total,
                query_ok=query_ok,
                input_tokens=input_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": stage,
                    "processed": processed,
                    "total": total,
                    "embedded_ok": embedded_ok,
                    "input_tokens": input_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                }
            )

    def _embed_missing(
        missing: list[tuple[str, str]],
        *,
        total_items: int,
        ok_count: int,
        stage: str,
    ) -> tuple[int, int, float]:
        nonlocal doc_ok, query_ok, input_tokens, estimated_cost_usd
        if not missing:
            return ok_count, input_tokens, estimated_cost_usd

        batches = [missing[i : i + max_batch] for i in range(0, len(missing), max_batch)]

        def _run_batch(chunk: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], Any]:
            return chunk, client.embed_texts([text for _, text in chunk])

        with ThreadPoolExecutor(max_workers=embed_concurrency) as executor:
            pending = {executor.submit(_run_batch, chunk) for chunk in batches[:embed_concurrency]}
            next_batch = embed_concurrency
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                first_error: Exception | None = None
                for fut in done:
                    try:
                        chunk, result = fut.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    cache.upsert_ok_many(
                        [
                            {
                                "cache_key": key,
                                "model": model,
                                "similarity": similarity,
                                "text_value": text,
                                "vector": vec,
                            }
                            for (key, text), vec in zip(chunk, result.vectors, strict=True)
                        ]
                    )
                    ok_count += len(chunk)
                    if stage == "doc_embed":
                        doc_ok = ok_count
                    else:
                        query_ok = ok_count
                    input_tokens += result.usage.input_tokens
                    estimated_cost_usd += result.usage.estimated_cost_usd
                    _emit(
                        stage=stage,
                        processed=min(ok_count, total_items),
                        total=total_items,
                        embedded_ok=ok_count,
                    )
                    if next_batch < len(batches):
                        pending.add(executor.submit(_run_batch, batches[next_batch]))
                        next_batch += 1
                if first_error is not None:
                    for fut in pending:
                        fut.cancel()
                    _emit(
                        stage=stage,
                        processed=min(ok_count, total_items),
                        total=total_items,
                        embedded_ok=ok_count,
                    )
                    raise first_error
        return ok_count, input_tokens, estimated_cost_usd

    try:
        doc_missing: list[tuple[str, str]] = []
        scan_batch = max(500, int(cfg.manticore.vector_cache_fetch_batch_size or 1000))
        for start in range(0, len(docs), scan_batch):
            docs_chunk = docs[start : start + scan_batch]
            keyed_chunk: list[tuple[str, str]] = []
            for doc in docs_chunk:
                key = cache_key_from_doc(doc, fields=fields, model=model, similarity=similarity)
                text_value = " ".join(normalize_text(str(doc.get(f, "") or "")) for f in fields).strip()
                keyed_chunk.append((key, text_value))
            status_map = cache.get_status_many([key for key, _ in keyed_chunk])
            for key, text_value in keyed_chunk:
                if status_map.get(key) == "ok":
                    doc_ok += 1
                else:
                    doc_missing.append((key, text_value))
            processed = min(start + len(docs_chunk), len(docs))
            if processed == len(docs) or processed == len(docs_chunk) or processed % 1000 == 0:
                _emit(stage="doc_cache_scan", processed=processed, total=len(docs), embedded_ok=doc_ok)

        doc_ok, _, _ = _embed_missing(
            doc_missing,
            total_items=len(docs),
            ok_count=doc_ok,
            stage="doc_embed",
        )

        query_missing: list[tuple[str, str]] = []
        keyed_topics: list[tuple[str, str]] = []
        for topic in topics:
            query_text = normalize_text(str(topic.get("query_long") or topic.get("query_short") or ""))
            keyed_topics.append((cache_key_from_text(query_text, model=model, similarity=similarity), query_text))
        for start in range(0, len(keyed_topics), scan_batch):
            topics_chunk = keyed_topics[start : start + scan_batch]
            status_map = cache.get_status_many([key for key, _ in topics_chunk])
            for key, query_text in topics_chunk:
                if status_map.get(key) == "ok":
                    query_ok += 1
                else:
                    query_missing.append((key, query_text))
            processed = min(start + len(topics_chunk), len(topics))
            if processed == len(topics) or processed == len(topics_chunk) or processed % 1000 == 0:
                _emit(stage="query_cache_scan", processed=processed, total=len(topics), embedded_ok=query_ok)

        query_ok, _, _ = _embed_missing(
            query_missing,
            total_items=len(topics),
            ok_count=query_ok,
            stage="query_embed",
        )
        cache.commit()
        if manifest_path is not None and manifest_payload is not None:
            _write_embedding_manifest(
                manifest_path,
                manifest_payload,
                cfg=cfg,
                doc_total=doc_total,
                doc_ok=doc_ok,
                query_total=query_total,
                query_ok=query_ok,
                input_tokens=input_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
    finally:
        cache.close()

    return EmbeddingBuildStats(
        doc_total=doc_total,
        doc_ok=doc_ok,
        query_total=query_total,
        query_ok=query_ok,
        input_tokens=input_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


def reset_embeddings(cfg: AppConfig) -> EmbeddingResetStats:
    prepared_dir = Path(cfg.dataset.prepared_dir)
    docs_path = prepared_dir / "docs.jsonl"
    topics_path = prepared_dir / "topics.jsonl"
    manifest_path = prepared_dir / "manifest.json"
    if not docs_path.exists() or not topics_path.exists():
        raise RuntimeError(
            f"Prepared dataset not found for reset: {prepared_dir}. Run `bench prepare` first."
        )

    docs = read_jsonl(docs_path)
    topics = read_jsonl(topics_path)
    model = resolved_embedding_model(cfg)
    similarity = resolved_embedding_similarity(cfg)
    fields = parse_vector_from_fields(cfg.manticore.vector_from)

    keys: list[str] = []
    for doc in docs:
        keys.append(cache_key_from_doc(doc, fields=fields, model=model, similarity=similarity))
    for topic in topics:
        text = normalize_text(str(topic.get("query_long") or topic.get("query_short") or ""))
        keys.append(cache_key_from_text(text, model=model, similarity=similarity))

    cache = VectorCache(cfg.manticore.vector_cache_path)
    try:
        deleted = cache.delete_many(keys)
    finally:
        cache.close()

    base_payload: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            base_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            base_payload = {}
    if not base_payload:
        base_payload = {
            "dataset_source": cfg.dataset.source,
            "dataset_max_docs": int(cfg.dataset.max_docs or 0),
            "dataset_keep_qrels_docs": bool(cfg.dataset.keep_qrels_docs),
            "doc_count": len(docs),
            "topic_count": len(topics),
        }

    _write_embedding_manifest(
        manifest_path,
        base_payload,
        cfg=cfg,
        doc_total=len(docs),
        doc_ok=0,
        query_total=len(topics),
        query_ok=0,
        input_tokens=0,
        estimated_cost_usd=0.0,
    )
    return EmbeddingResetStats(
        deleted=deleted,
        doc_total=len(docs),
        query_total=len(topics),
    )


def _doc_hash_u64(doc_id: str) -> int:
    d = hashlib.sha1(str(doc_id).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def _docs_iter_or_raise(ds: Any, source: str):
    try:
        yield from ds.docs_iter()
    except FileNotFoundError as exc:
        source_text = str(source or "").strip()
        if source_text.startswith("wapo/"):
            raise RuntimeError(
                "TREC-NEWS requires the Washington Post collection to be installed for ir_datasets. "
                "Request access at https://trec.nist.gov/data/wapost/ and follow "
                "https://ir-datasets.com/wapo.html#DataAccess . "
                f"Expected local file: {exc}"
            ) from exc
        raise


def _prepare_hf_beir_dataset(cfg: AppConfig, *, progress_callback: callable | None = None) -> PrepareResult:
    from huggingface_hub import hf_hub_download

    source = str(cfg.dataset.source)
    _, spec = source.split(":", 1)
    repo_id, split = spec.rsplit(":", 1)
    split = split.strip().lower()
    if split not in {"train", "valid", "test"}:
        raise ValueError(
            f"Unsupported Hugging Face BEIR split '{split}' in dataset source '{source}'. "
            "Expected one of: train, valid, test."
        )

    out_dir = ensure_dir(Path(cfg.dataset.prepared_dir))
    started = perf_counter()

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="ACORD Dataset & ReadMe.zip",
    )

    docs: list[dict[str, Any]] = []
    topics: list[dict[str, str]] = []
    qrels_rows: list[tuple[str, str, int]] = []
    qrels_doc_ids: set[str] = set()

    with ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        root = next(
            (name.rsplit("/", 1)[0] for name in names if name.endswith("/queries.jsonl")),
            "",
        )
        if not root:
            raise ValueError(f"Could not locate queries.jsonl inside '{zip_path}'.")

        corpus_name = f"{root}/corpus.jsonl"
        queries_name = f"{root}/queries.jsonl"
        qrels_name = f"{root}/qrels/{split}.tsv"
        for req in (corpus_name, queries_name, qrels_name):
            if req not in names:
                raise ValueError(f"Missing expected file '{req}' in '{zip_path}'.")

        with zf.open(corpus_name) as f:
            for idx, raw in enumerate(f, start=1):
                row = json.loads(raw)
                doc_id = str(row.get("_id", "")).strip()
                if not doc_id:
                    continue
                text = str(row.get("text", "")).strip()
                title = str(row.get("title", "")).strip()
                docs.append(
                    {
                        "id": idx,
                        "doc_id": doc_id,
                        "title": title,
                        "abstract": text,
                        "doi": "",
                        "date": "",
                    }
                )

        with zf.open(queries_name) as f:
            for raw in f:
                row = json.loads(raw)
                meta = row.get("metadata") or {}
                if str(meta.get("split", "")).strip().lower() != split:
                    continue
                topic_id = str(row.get("_id", "")).strip()
                text = str(row.get("text", "")).strip()
                if not topic_id or not text:
                    continue
                topics.append(
                    {
                        "topic_id": topic_id,
                        "query_short": text,
                        "query_long": text,
                    }
                )

        with zf.open(qrels_name) as f:
            reader = csv.DictReader((line.decode("utf-8") for line in f), delimiter="\t")
            for row in reader:
                topic_id = str(row.get("query-id", "")).strip()
                doc_id = str(row.get("corpus-id", "")).strip()
                if not topic_id or not doc_id:
                    continue
                try:
                    relevance = int(row.get("score", 0))
                except (TypeError, ValueError):
                    relevance = 0
                qrels_rows.append((topic_id, doc_id, relevance))
                qrels_doc_ids.add(doc_id)

    docs_path = out_dir / "docs.jsonl"
    topics_path = out_dir / "topics.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(docs_path, docs)
    write_jsonl(topics_path, topics)
    write_tsv(qrels_path, qrels_rows)
    manifest_payload = {
        "dataset_source": cfg.dataset.source,
        "dataset_max_docs": 0,
        "dataset_keep_qrels_docs": True,
        "doc_count": len(docs),
        "selected_qrel_docs": len(qrels_doc_ids),
        "selected_extra_docs": max(len(docs) - len(qrels_doc_ids), 0),
        "topic_count": len(topics),
        "qrels_count": len(qrels_rows),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    _write_embedding_manifest(
        manifest_path,
        manifest_payload,
        cfg=cfg,
        doc_total=len(docs),
        doc_ok=0,
        query_total=len(topics),
        query_ok=0,
        input_tokens=0,
        estimated_cost_usd=0.0,
    )
    stats = _populate_embeddings(
        cfg,
        docs,
        topics,
        progress_callback=progress_callback,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    write_json(manifest_path, _manifest_with_embeddings(manifest_payload, cfg=cfg, stats=stats))

    return PrepareResult(
        docs_path=docs_path,
        topics_path=topics_path,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
    )


_DL_TYPO_BASE_URL = "https://raw.githubusercontent.com/ielab/CharacterBERT-DR/main/data/dl-typo"
_DL_TYPO_CORPUS_SOURCE = "msmarco-passage/trec-dl-2019/judged"


def _read_remote_text(url: str) -> str:
    with urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8", "replace")


def _load_tsv_query_map(url: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in _read_remote_text(url).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        topic_id, sep, query = line.partition("\t")
        topic_id = topic_id.strip()
        query = query.strip()
        if not topic_id or not sep or not query:
            continue
        out[topic_id] = query
    return out


def _load_trec_qrels_rows(url: str) -> tuple[list[tuple[str, str, int]], set[str]]:
    rows: list[tuple[str, str, int]] = []
    doc_ids: set[str] = set()
    for raw_line in _read_remote_text(url).splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 4:
            continue
        topic_id, _, doc_id, relevance_raw = parts[:4]
        try:
            relevance = int(relevance_raw)
        except (TypeError, ValueError):
            relevance = 0
        if not topic_id or not doc_id:
            continue
        rows.append((topic_id, doc_id, relevance))
        doc_ids.add(doc_id)
    return rows, doc_ids


def _prepare_dl_typo_dataset(cfg: AppConfig, *, progress_callback: callable | None = None) -> PrepareResult:
    import ir_datasets

    out_dir = ensure_dir(Path(cfg.dataset.prepared_dir))
    started = perf_counter()

    typo_queries = _load_tsv_query_map(f"{_DL_TYPO_BASE_URL}/query.typo.tsv")
    clean_queries = _load_tsv_query_map(f"{_DL_TYPO_BASE_URL}/query.tsv")
    qrels_all_rows, qrels_doc_ids = _load_trec_qrels_rows(f"{_DL_TYPO_BASE_URL}/qrels.txt")

    topics: list[dict[str, str]] = []
    for topic_id, typo_query in sorted(typo_queries.items(), key=lambda item: item[0]):
        clean_query = str(clean_queries.get(topic_id) or typo_query).strip()
        typo_query = str(typo_query or "").strip()
        if not typo_query:
            continue
        topics.append(
            {
                "topic_id": topic_id,
                "query_short": typo_query,
                "query_long": typo_query,
                "query_clean": clean_query or typo_query,
            }
        )

    ds = ir_datasets.load(_DL_TYPO_CORPUS_SOURCE)
    max_docs = int(cfg.dataset.max_docs or 0)
    keep_qrels_docs = bool(cfg.dataset.keep_qrels_docs)

    docs: list[dict[str, Any]] = []
    kept_doc_ids: set[str] = set()
    selected_qrel_docs = 0
    selected_extra_docs = 0
    if max_docs <= 0:
        for doc in _docs_iter_or_raise(ds, _DL_TYPO_CORPUS_SOURCE):
            doc_id = _doc_attr(doc, ["doc_id", "id"])
            if not doc_id:
                continue
            kept_doc_ids.add(doc_id)
            docs.append(
                {
                    "id": len(docs) + 1,
                    "doc_id": doc_id,
                    "title": _doc_attr(doc, ["title"]),
                    "abstract": _doc_attr(doc, ["abstract", "text", "body"]),
                    "doi": _doc_attr(doc, ["doi"]),
                    "date": _doc_attr(doc, ["date", "publish_time"]),
                }
            )
    else:
        target_qrels = len(qrels_doc_ids) if keep_qrels_docs else 0
        target_extra = max(max_docs - target_qrels, 0)
        extra_heap: list[tuple[int, int, dict[str, Any]]] = []
        seq = 0
        for doc in _docs_iter_or_raise(ds, _DL_TYPO_CORPUS_SOURCE):
            doc_id = _doc_attr(doc, ["doc_id", "id"])
            if not doc_id:
                continue
            row = {
                "doc_id": doc_id,
                "title": _doc_attr(doc, ["title"]),
                "abstract": _doc_attr(doc, ["abstract", "text", "body"]),
                "doi": _doc_attr(doc, ["doi"]),
                "date": _doc_attr(doc, ["date", "publish_time"]),
            }
            if keep_qrels_docs and doc_id in qrels_doc_ids:
                kept_doc_ids.add(doc_id)
                docs.append({"id": 0, **row})
                selected_qrel_docs += 1
            elif target_extra > 0:
                score = _doc_hash_u64(doc_id)
                item = (-score, seq, row)
                seq += 1
                if len(extra_heap) < target_extra:
                    heapq.heappush(extra_heap, item)
                elif item > extra_heap[0]:
                    heapq.heapreplace(extra_heap, item)
        extras = [item[2] for item in sorted(extra_heap, reverse=True)]
        for row in extras:
            kept_doc_ids.add(str(row["doc_id"]))
            docs.append({"id": 0, **row})
            selected_extra_docs += 1
        docs.sort(key=lambda d: str(d.get("doc_id", "")))
        for idx, row in enumerate(docs, start=1):
            row["id"] = idx

    qrels_rows = [(t, d, rel) for (t, d, rel) in qrels_all_rows if d in kept_doc_ids]

    docs_path = out_dir / "docs.jsonl"
    topics_path = out_dir / "topics.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(docs_path, docs)
    write_jsonl(topics_path, topics)
    write_tsv(qrels_path, qrels_rows)
    manifest_payload = {
        "dataset_source": cfg.dataset.source,
        "dataset_upstream_source": _DL_TYPO_CORPUS_SOURCE,
        "dataset_max_docs": max_docs,
        "dataset_keep_qrels_docs": keep_qrels_docs,
        "doc_count": len(docs),
        "selected_qrel_docs": selected_qrel_docs if max_docs > 0 else len(qrels_doc_ids & kept_doc_ids),
        "selected_extra_docs": selected_extra_docs if max_docs > 0 else max(len(docs) - len(qrels_doc_ids & kept_doc_ids), 0),
        "topic_count": len(topics),
        "qrels_count": len(qrels_rows),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    _write_embedding_manifest(
        manifest_path,
        manifest_payload,
        cfg=cfg,
        doc_total=len(docs),
        doc_ok=0,
        query_total=len(topics),
        query_ok=0,
        input_tokens=0,
        estimated_cost_usd=0.0,
    )
    stats = _populate_embeddings(
        cfg,
        docs,
        topics,
        progress_callback=progress_callback,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    write_json(manifest_path, _manifest_with_embeddings(manifest_payload, cfg=cfg, stats=stats))

    return PrepareResult(
        docs_path=docs_path,
        topics_path=topics_path,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
    )


def prepare_dataset(cfg: AppConfig, *, progress_callback: callable | None = None) -> PrepareResult:
    if str(cfg.dataset.source).startswith("hf-beir:"):
        return _prepare_hf_beir_dataset(cfg, progress_callback=progress_callback)
    if str(cfg.dataset.source).strip().lower() == "dl-typo":
        return _prepare_dl_typo_dataset(cfg, progress_callback=progress_callback)

    import ir_datasets

    out_dir = ensure_dir(Path(cfg.dataset.prepared_dir))

    started = perf_counter()
    ds = ir_datasets.load(cfg.dataset.source)
    has_qrels = True
    if hasattr(ds, "has_qrels"):
        has_qrels = bool(ds.has_qrels())
    else:
        has_qrels = hasattr(ds, "qrels_iter")
    if not has_qrels:
        raise ValueError(
            f"Dataset source '{cfg.dataset.source}' does not expose qrels in ir_datasets. "
            "Choose a split/source that has relevance judgments."
        )

    topics: list[dict[str, str]] = []
    for t in ds.queries_iter():
        topic_id = _doc_attr(t, ["query_id", "topic_id", "id"])
        query_short = _doc_attr(t, ["title", "query", "question", "short_query", "text"])
        query_long = _doc_attr(t, ["description", "narrative", "text", "query", "title"])
        if not topic_id or not query_short:
            continue
        if not query_long:
            query_long = query_short
        topics.append(
            {
                "topic_id": topic_id,
                "query_short": query_short,
                "query_long": query_long,
            }
        )

    qrels_all_rows: list[tuple[str, str, int]] = []
    qrels_doc_ids: set[str] = set()
    for qr in ds.qrels_iter():
        topic_id = _doc_attr(qr, ["query_id", "topic_id"])
        doc_id = _doc_attr(qr, ["doc_id", "docno", "document_id"])
        relevance_raw = getattr(qr, "relevance", 0)
        try:
            relevance = int(relevance_raw)
        except (TypeError, ValueError):
            relevance = 0
        if not topic_id or not doc_id:
            continue
        qrels_all_rows.append((topic_id, doc_id, relevance))
        qrels_doc_ids.add(doc_id)

    max_docs = int(cfg.dataset.max_docs or 0)
    keep_qrels_docs = bool(cfg.dataset.keep_qrels_docs)

    docs: list[dict[str, Any]] = []
    kept_doc_ids: set[str] = set()
    selected_qrel_docs = 0
    selected_extra_docs = 0
    if max_docs <= 0:
        for idx, doc in enumerate(_docs_iter_or_raise(ds, cfg.dataset.source), start=1):
            doc_id = _doc_attr(doc, ["doc_id", "id"])
            if not doc_id:
                continue
            title = _doc_attr(doc, ["title"])
            abstract = _doc_attr(doc, ["abstract", "text", "body"])
            doi = _doc_attr(doc, ["doi"])
            date = _doc_attr(doc, ["date", "publish_time"])
            kept_doc_ids.add(doc_id)
            docs.append(
                {
                    "id": len(docs) + 1,
                    "doc_id": doc_id,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "date": date,
                }
            )
    else:
        target_qrels = len(qrels_doc_ids) if keep_qrels_docs else 0
        target_extra = max(max_docs - target_qrels, 0)
        extra_heap: list[tuple[int, int, dict[str, Any]]] = []
        seq = 0
        for doc in _docs_iter_or_raise(ds, cfg.dataset.source):
            doc_id = _doc_attr(doc, ["doc_id", "id"])
            if not doc_id:
                continue
            row = {
                "doc_id": doc_id,
                "title": _doc_attr(doc, ["title"]),
                "abstract": _doc_attr(doc, ["abstract", "text", "body"]),
                "doi": _doc_attr(doc, ["doi"]),
                "date": _doc_attr(doc, ["date", "publish_time"]),
            }
            if keep_qrels_docs and doc_id in qrels_doc_ids:
                docs.append({"id": 0, **row})
                kept_doc_ids.add(doc_id)
                selected_qrel_docs += 1
                continue
            if target_extra <= 0:
                continue
            h = _doc_hash_u64(doc_id)
            item = (-h, seq, row)
            seq += 1
            if len(extra_heap) < target_extra:
                heapq.heappush(extra_heap, item)
                continue
            worst_h = -extra_heap[0][0]
            if h < worst_h:
                heapq.heapreplace(extra_heap, item)
        if extra_heap:
            extras = sorted(extra_heap, key=lambda x: (-x[0], x[1]))
            for _, _, row in extras:
                doc_id = str(row["doc_id"])
                if doc_id in kept_doc_ids:
                    continue
                docs.append({"id": 0, **row})
                kept_doc_ids.add(doc_id)
                selected_extra_docs += 1
        for i, row in enumerate(docs, start=1):
            row["id"] = i

    qrels_rows: list[tuple[str, str, int]] = [
        (topic_id, doc_id, rel)
        for (topic_id, doc_id, rel) in qrels_all_rows
        if doc_id in kept_doc_ids
    ]

    docs_path = out_dir / "docs.jsonl"
    topics_path = out_dir / "topics.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(docs_path, docs)
    write_jsonl(topics_path, topics)
    write_tsv(qrels_path, qrels_rows)
    manifest_payload = {
        "dataset_source": cfg.dataset.source,
        "dataset_max_docs": max_docs,
        "dataset_keep_qrels_docs": keep_qrels_docs,
        "doc_count": len(docs),
        "selected_qrel_docs": selected_qrel_docs,
        "selected_extra_docs": selected_extra_docs,
        "topic_count": len(topics),
        "qrels_count": len(qrels_rows),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    _write_embedding_manifest(
        manifest_path,
        manifest_payload,
        cfg=cfg,
        doc_total=len(docs),
        doc_ok=0,
        query_total=len(topics),
        query_ok=0,
        input_tokens=0,
        estimated_cost_usd=0.0,
    )
    stats = _populate_embeddings(
        cfg,
        docs,
        topics,
        progress_callback=progress_callback,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    write_json(manifest_path, _manifest_with_embeddings(manifest_payload, cfg=cfg, stats=stats))

    return PrepareResult(
        docs_path=docs_path,
        topics_path=topics_path,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
    )
