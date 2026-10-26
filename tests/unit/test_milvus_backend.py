from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from bench.config import AppConfig, MilvusConfig, load_engine_targets
from bench.index_matrix import build_index_variants_from_config_paths
from bench.indexing import _run_index_milvus
from bench.milvus_backend import MilvusBackend
from bench.run_matrix import build_variants_from_config_paths


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict[str, Any]] = []
        self.functions: list[Any] = []

    def add_field(self, **kwargs: Any) -> None:
        self.fields.append(kwargs)

    def add_function(self, function: Any) -> None:
        self.functions.append(function)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _FakeMilvusClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.schema = _FakeSchema()
        self.index_params = _FakeIndexParams()
        self.created: dict[str, Any] | None = None
        self.inserted: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.hybrid_calls: list[dict[str, Any]] = []
        self.exists = False
        self.closed = False

    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        return self.schema

    def prepare_index_params(self) -> _FakeIndexParams:
        return self.index_params

    def has_collection(self, **kwargs: Any) -> bool:
        return self.exists

    def drop_collection(self, **kwargs: Any) -> None:
        self.exists = False

    def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs
        self.exists = True

    def insert(self, **kwargs: Any) -> dict[str, int]:
        self.inserted.extend(kwargs["data"])
        return {"insert_count": len(kwargs["data"])}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return [[{"entity": {"doc_id": "d1"}, "distance": 0.9}]]

    def hybrid_search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.hybrid_calls.append(kwargs)
        return [[{"entity": {"doc_id": "d2"}, "distance": 0.8}]]

    def flush(self, **kwargs: Any) -> None:
        return None

    def load_collection(self, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pymilvus(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("pymilvus")
    module.DataType = SimpleNamespace(
        INT64="INT64",
        VARCHAR="VARCHAR",
        TEXT="TEXT",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
        FLOAT_VECTOR="FLOAT_VECTOR",
    )
    module.FunctionType = SimpleNamespace(BM25="BM25")
    module.Function = lambda **kwargs: SimpleNamespace(**kwargs)
    module.AnnSearchRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    module.RRFRanker = lambda k: SimpleNamespace(k=k)
    module.MilvusClient = _FakeMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    return module


def test_milvus_schema_uses_text_bm25_and_dense_vector(fake_pymilvus: ModuleType) -> None:
    backend = MilvusBackend(
        MilvusConfig(
            collection_name="docs",
            text_fields=["title", "abstract"],
        )
    )

    backend.recreate_collection(dimension=3)

    fields = {field["field_name"]: field for field in backend.client.schema.fields}
    assert fields["doc_id"]["datatype"] == "VARCHAR"
    assert fields["content"]["datatype"] == "TEXT"
    assert "max_length" not in fields["content"]
    assert fields["content"]["enable_analyzer"] is True
    assert fields["sparse"]["datatype"] == "SPARSE_FLOAT_VECTOR"
    assert fields["embedding"]["datatype"] == "FLOAT_VECTOR"
    assert fields["embedding"]["dim"] == 3

    function = backend.client.schema.functions[0]
    assert function.input_field_names == ["content"]
    assert function.output_field_names == ["sparse"]
    assert function.function_type == "BM25"

    indexes = {index["field_name"]: index for index in backend.client.index_params.indexes}
    assert indexes["sparse"]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert indexes["sparse"]["metric_type"] == "BM25"
    assert indexes["embedding"]["metric_type"] == "COSINE"


def test_milvus_insert_accepts_long_text_and_validates_doc_id_utf8_bytes(
    fake_pymilvus: ModuleType,
) -> None:
    backend = MilvusBackend(
        MilvusConfig(
            collection_name="docs",
            content_max_length=1,  # Legacy option is ignored for TEXT fields.
            doc_id_max_length=8,
        )
    )
    count = backend.import_documents(
        [
            {
                "id": 7,
                "doc_id": "d1",
                "title": "one",
                "abstract": "two",
            }
        ],
        with_vector=False,
    )

    assert count == 1
    assert backend.client.inserted == [{"id": 7, "doc_id": "d1", "content": "one two"}]

    long_text = "€" * 22_000
    assert backend.import_documents(
        [{"id": 8, "doc_id": "d2", "title": long_text, "abstract": ""}],
        with_vector=False,
    ) == 1
    assert backend.client.inserted[-1]["content"] == long_text

    with pytest.raises(ValueError, match="9 bytes > 8 bytes"):
        backend.import_documents(
            [{"id": 9, "doc_id": "€€€", "title": "text", "abstract": ""}],
            with_vector=False,
        )

    backend.cfg.doc_id_max_length = 65_536
    with pytest.raises(ValueError, match="between 1 and 65535"):
        backend.recreate_collection()


def test_milvus_native_searches_use_bm25_cosine_and_rrf(fake_pymilvus: ModuleType) -> None:
    backend = MilvusBackend(
        MilvusConfig(
            collection_name="docs",
            hybrid_candidates=25,
            hybrid_rrf_k=42,
        )
    )

    assert backend.search_lexical("virus", 5) == [{"doc_id": "d1", "score": 0.9}]
    assert backend.client.search_calls[-1]["data"] == ["virus"]
    assert backend.client.search_calls[-1]["search_params"]["metric_type"] == "BM25"

    assert backend.search_vector([0.1, 0.2], 5) == [{"doc_id": "d1", "score": 0.9}]
    assert backend.client.search_calls[-1]["search_params"]["metric_type"] == "COSINE"

    assert backend.search_hybrid("virus", [0.1, 0.2], 5) == [
        {"doc_id": "d2", "score": 0.8}
    ]
    hybrid = backend.client.hybrid_calls[-1]
    assert hybrid["ranker"].k == 42
    assert [request.limit for request in hybrid["reqs"]] == [25, 25]
    assert hybrid["reqs"][0].data == ["virus"]


def test_milvus_profiles_appear_in_index_and_run_matrices(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
milvus:
  type: milvus
  milvus:
    collection_name: docs
milvus_fulltext:
  base: milvus
  type: fulltext
milvus_vector:
  base: milvus
  type: vector
milvus_hybrid:
  base: milvus
  type: hybrid
""".strip()
        + "\n",
        encoding="utf-8",
    )

    targets = [target for target in load_engine_targets(config_path) if target.type == "milvus"]
    run_rows = [row for row in build_variants_from_config_paths([config_path]) if row.engine == "milvus"]
    index_rows = [row for row in build_index_variants_from_config_paths([config_path]) if row.engine == "milvus"]

    assert {target.mode_filter for target in targets} >= {"fulltext", "vector", "hybrid"}
    assert {row.mode for row in run_rows} == {"fulltext", "vector", "hybrid"}
    assert [(row.mode, row.target_kind, row.target) for row in index_rows] == [
        ("fulltext", "fulltext", "docs"),
        ("vector", "hybrid", "docs"),
    ]
    assert index_rows[0].used_by == ("milvus_fulltext",)
    assert index_rows[1].used_by == ("milvus_hybrid", "milvus_vector")


def test_milvus_fulltext_indexing_recreates_and_finalizes_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class _Backend:
        def __init__(self, cfg: Any):
            self.cfg = cfg
            self.recreated_dimension: int | None = -1
            self.rows: list[dict[str, Any]] = []
            self.finalized = False
            self.closed = False
            instances.append(self)

        def ensure_accessible(self) -> None:
            return None

        def recreate_collection(self, *, dimension: int | None = None) -> None:
            self.recreated_dimension = dimension

        def import_documents(self, docs: list[dict[str, Any]], *, with_vector: bool) -> int:
            assert with_vector is False
            self.rows.extend(docs)
            return len(docs)

        def finalize_collection(self) -> None:
            self.finalized = True

        def close(self) -> None:
            self.closed = True

    cfg = AppConfig()
    cfg.milvus.batch_size = 1
    docs = [
        {"id": 1, "doc_id": "d1", "title": "one", "abstract": "two"},
        {"id": 2, "doc_id": "d2", "title": "three", "abstract": "four"},
    ]
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr("bench.indexing.read_jsonl", lambda path: docs)
    monkeypatch.setattr("bench.indexing.MilvusBackend", _Backend)

    results = _run_index_milvus(cfg, mode="fulltext", progress_callback=updates.append)

    backend = instances[0]
    assert results[0].rows_inserted == 2
    assert backend.recreated_dimension is None
    assert backend.rows == docs
    assert backend.finalized is True
    assert backend.closed is True
    assert any(update["stage"] == "finalize" for update in updates)
