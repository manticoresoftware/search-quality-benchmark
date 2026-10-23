from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types

from bench.config import AppConfig
from bench.prepare import EmbeddingBuildStats, prepare_dataset


@dataclass
class _Doc:
    doc_id: str
    title: str
    text: str


@dataclass
class _Query:
    query_id: str
    text: str
    query: str
    narrative: str


@dataclass
class _Qrel:
    query_id: str
    doc_id: str
    relevance: int


class _Dataset:
    def docs_iter(self):
        yield _Doc("d1", "title", "abstract")

    def queries_iter(self):
        yield _Query(
            "1",
            "what is the origin of COVID-19",
            "coronavirus origin",
            "seeking information about the SARS-CoV-2 origin",
        )

    def qrels_iter(self):
        yield _Qrel("1", "d1", 2)


def test_prepare_trec_covid_preserves_all_official_query_fields(
    tmp_path: Path, monkeypatch
) -> None:
    fake_ir = types.SimpleNamespace(load=lambda source: _Dataset())
    monkeypatch.setitem(sys.modules, "ir_datasets", fake_ir)

    captured: dict[str, object] = {}

    def _fake_populate(cfg, docs, topics, **kwargs):
        captured["topics"] = topics
        return EmbeddingBuildStats(
            doc_total=len(docs),
            doc_ok=len(docs),
            query_total=3,
            query_ok=3,
            input_tokens=0,
            estimated_cost_usd=0.0,
        )

    monkeypatch.setattr("bench.prepare._populate_embeddings", _fake_populate)
    cfg = AppConfig()
    cfg.dataset.source = "beir/trec-covid"
    cfg.dataset.prepared_dir = str(tmp_path / "prepared")

    result = prepare_dataset(cfg)

    topic = json.loads(result.topics_path.read_text(encoding="utf-8"))
    assert topic == {
        "topic_id": "1",
        "query": "coronavirus origin",
        "question": "what is the origin of COVID-19",
        "narrative": "seeking information about the SARS-CoV-2 origin",
    }
    assert captured["topics"] == [topic]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["query_schema_version"] == 2
    assert manifest["query_fields"] == ["query", "question", "narrative"]
    assert manifest["query_embeddings_total"] == 3
