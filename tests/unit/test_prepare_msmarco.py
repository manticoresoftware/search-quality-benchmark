from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

from bench.config import AppConfig
from bench.prepare import prepare_dataset


@dataclass
class _Doc:
    doc_id: str
    title: str
    text: str


@dataclass
class _Query:
    query_id: str
    query: str


@dataclass
class _Qrel:
    query_id: str
    doc_id: str
    relevance: int


class _Dataset:
    def docs_iter(self):
        yield _Doc("d1", "t1", "body1")
        yield _Doc("d2", "t2", "body2")

    def queries_iter(self):
        # MSMARCO-like query object often exposes just .query
        yield _Query("q1", "what is covid")

    def qrels_iter(self):
        yield _Qrel("q1", "d1", 1)
        yield _Qrel("q1", "d3", 1)  # missing doc should be filtered out


def test_prepare_msmarco_query_fallback_and_qrels_filter(tmp_path: Path, monkeypatch):
    fake_ir = types.SimpleNamespace(load=lambda source: _Dataset())
    monkeypatch.setitem(sys.modules, "ir_datasets", fake_ir)
    cfg = AppConfig()
    cfg.dataset.source = "msmarco-passage/dev/small"
    cfg.dataset.prepared_dir = str(tmp_path / "prepared")

    result = prepare_dataset(cfg)

    topics = result.topics_path.read_text(encoding="utf-8")
    assert '"query_short": "what is covid"' in topics
    assert '"query_long": "what is covid"' in topics

    qrels = result.qrels_path.read_text(encoding="utf-8")
    assert "q1\td1\t1" in qrels
    assert "q1\td3\t1" not in qrels


class _DatasetLite:
    def docs_iter(self):
        for i in range(1, 11):
            yield _Doc(f"d{i}", f"t{i}", f"body{i}")

    def queries_iter(self):
        yield _Query("q1", "what is covid")

    def qrels_iter(self):
        yield _Qrel("q1", "d9", 1)


def test_prepare_max_docs_keeps_qrels_and_is_deterministic(tmp_path: Path, monkeypatch):
    fake_ir = types.SimpleNamespace(load=lambda source: _DatasetLite())
    monkeypatch.setitem(sys.modules, "ir_datasets", fake_ir)
    cfg = AppConfig()
    cfg.dataset.source = "msmarco-passage/dev/small"
    cfg.dataset.max_docs = 3
    cfg.dataset.keep_qrels_docs = True
    cfg.dataset.prepared_dir = str(tmp_path / "prepared")

    result1 = prepare_dataset(cfg)
    docs1 = result1.docs_path.read_text(encoding="utf-8")
    assert docs1.count('"doc_id"') == 3
    assert '"doc_id": "d9"' in docs1

    result2 = prepare_dataset(cfg)
    docs2 = result2.docs_path.read_text(encoding="utf-8")
    assert docs1 == docs2
