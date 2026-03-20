from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import AppConfig
from .io_utils import ensure_dir, write_json, write_jsonl, write_tsv


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


def _doc_hash_u64(doc_id: str) -> int:
    d = hashlib.sha1(str(doc_id).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def prepare_dataset(cfg: AppConfig) -> PrepareResult:
    import ir_datasets

    out_dir = ensure_dir(Path(cfg.dataset.prepared_dir))

    started = perf_counter()
    ds = ir_datasets.load(cfg.dataset.source)

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
        for idx, doc in enumerate(ds.docs_iter(), start=1):
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
        for doc in ds.docs_iter():
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
    write_json(
        manifest_path,
        {
            "dataset_source": cfg.dataset.source,
            "dataset_max_docs": max_docs,
            "dataset_keep_qrels_docs": keep_qrels_docs,
            "doc_count": len(docs),
            "selected_qrel_docs": selected_qrel_docs,
            "selected_extra_docs": selected_extra_docs,
            "topic_count": len(topics),
            "qrels_count": len(qrels_rows),
            "elapsed_seconds": round(perf_counter() - started, 3),
        },
    )

    return PrepareResult(
        docs_path=docs_path,
        topics_path=topics_path,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
    )
