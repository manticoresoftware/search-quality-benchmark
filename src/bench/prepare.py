from __future__ import annotations

from dataclasses import dataclass
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


def prepare_dataset(cfg: AppConfig) -> PrepareResult:
    import ir_datasets

    out_dir = ensure_dir(Path(cfg.dataset.prepared_dir))

    started = perf_counter()
    ds = ir_datasets.load(cfg.dataset.source)

    docs: list[dict[str, Any]] = []
    kept_doc_ids: set[str] = set()
    doc_id_to_int: dict[str, int] = {}

    for idx, doc in enumerate(ds.docs_iter(), start=1):
        doc_id = _doc_attr(doc, ["doc_id", "id"])
        if not doc_id:
            continue
        title = _doc_attr(doc, ["title"])
        abstract = _doc_attr(doc, ["abstract", "text", "body"])
        doi = _doc_attr(doc, ["doi"])
        date = _doc_attr(doc, ["date", "publish_time"])
        doc_int_id = len(docs) + 1
        doc_id_to_int[doc_id] = doc_int_id
        kept_doc_ids.add(doc_id)
        docs.append(
            {
                "id": doc_int_id,
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "date": date,
            }
        )
        if idx % 100000 == 0:
            pass

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

    qrels_rows: list[tuple[str, str, int]] = []
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
        if doc_id in kept_doc_ids:
            qrels_rows.append((topic_id, doc_id, relevance))

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
            "doc_count": len(docs),
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
