from types import SimpleNamespace

import pytest
from qdrant_client import models

from bench.qdrant_backend import QdrantBackend
from bench.config import AppConfig
from bench.indexing import _run_index_qdrant


def test_payload_too_large_error_detection():
    assert QdrantBackend._is_payload_too_large_error(
        Exception("Payload error: JSON payload (73236200 bytes) is larger than allowed (limit: 33554432 bytes).")
    )
    assert not QdrantBackend._is_payload_too_large_error(Exception("connection timeout"))


def test_qdrant_fulltext_indexing_uses_thread_local_backends(monkeypatch):
    docs = [{"id": idx, "doc_id": f"doc-{idx}", "title": f"title {idx}", "abstract": f"abstract {idx}"} for idx in range(4)]
    bootstrap_instance_ids: list[int] = []
    worker_instance_ids: list[int] = []

    class FakeBackend:
        next_id = 0

        def __init__(self, cfg):
            self.cfg = cfg
            self.instance_id = FakeBackend.next_id
            FakeBackend.next_id += 1

        def ensure_accessible(self):
            bootstrap_instance_ids.append(self.instance_id)

        def recreate_bm25_collection(self):
            bootstrap_instance_ids.append(self.instance_id)

        def collection_supports(self, collection_name, mode):
            return False

        def upsert_bm25_text_batch(self, batch):
            import time

            worker_instance_ids.append(self.instance_id)
            time.sleep(0.01)
            return len(batch)

        def wait_for_points_count(self, collection_name, expected_count, timeout_seconds):
            return expected_count

    monkeypatch.setattr("bench.indexing.read_jsonl", lambda path: docs)
    monkeypatch.setattr("bench.indexing.QdrantBackend", FakeBackend)

    cfg = AppConfig()
    cfg.dataset.prepared_dir = "/tmp/unused"
    cfg.qdrant.batch_size = 1
    cfg.qdrant.index_concurrency = 2
    cfg.qdrant.timeout_seconds = 1

    results = _run_index_qdrant(cfg, mode="fulltext")

    assert len(results) == 1
    assert results[0].rows_inserted == len(docs)
    assert bootstrap_instance_ids
    assert worker_instance_ids
    assert set(worker_instance_ids).isdisjoint(set(bootstrap_instance_ids))


def _backend_with_fake_client(client):
    cfg = AppConfig().qdrant
    cfg.bm25_collection = "docs"
    cfg.dense_collection = "docs"
    cfg.hybrid_candidates = 25
    cfg.hybrid_rrf_k = 61
    cfg.hybrid_sparse_weight = 2.0
    cfg.hybrid_dense_weight = 3.0
    backend = QdrantBackend.__new__(QdrantBackend)
    backend.cfg = cfg
    backend.models = models
    backend.client = client
    return backend


def test_qdrant_native_hybrid_query_uses_one_collection_and_server_rrf():
    calls = []

    class FakeClient:
        def query_points(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                points=[SimpleNamespace(payload={"doc_id": "d1"}, score=0.75)]
            )

    backend = _backend_with_fake_client(FakeClient())
    rows = backend.query_hybrid("covid origin", [0.1, 0.2], limit=10)

    assert rows == [{"doc_id": "d1", "score": 0.75}]
    assert len(calls) == 1
    request = calls[0]
    assert request["collection_name"] == "docs"
    assert request["limit"] == 10
    assert [prefetch.using for prefetch in request["prefetch"]] == ["bm25", "dense"]
    assert [prefetch.limit for prefetch in request["prefetch"]] == [25, 25]
    assert request["query"].rrf.k == 61
    assert request["query"].rrf.weights == [2.0, 3.0]


def test_qdrant_native_hybrid_rejects_separate_collections():
    backend = _backend_with_fake_client(SimpleNamespace())
    backend.cfg.bm25_collection = "docs-bm25"
    backend.cfg.dense_collection = "docs-dense"

    with pytest.raises(RuntimeError, match="one combined collection"):
        backend.query_hybrid("covid origin", [0.1, 0.2], limit=10)


def test_qdrant_hybrid_collection_and_points_contain_both_vectors():
    recreate_calls = []
    upsert_calls = []

    class FakeClient:
        def recreate_collection(self, **kwargs):
            recreate_calls.append(kwargs)

        def upsert(self, **kwargs):
            upsert_calls.append(kwargs)

    backend = _backend_with_fake_client(FakeClient())
    backend.recreate_hybrid_collection(size=2)
    inserted = backend.upsert_hybrid_batch(
        [
            {
                "id": 1,
                "doc_id": "d1",
                "title": "covid",
                "abstract": "origin",
                "vector": [0.1, 0.2],
            }
        ]
    )

    assert inserted == 1
    assert set(recreate_calls[0]["vectors_config"]) == {"dense"}
    assert set(recreate_calls[0]["sparse_vectors_config"]) == {"bm25"}
    point = upsert_calls[0]["points"][0]
    assert point.vector["dense"] == [0.1, 0.2]
    assert point.vector["bm25"].model == "Qdrant/bm25"


def test_qdrant_fulltext_backend_guard_preserves_existing_combined_collection(
    monkeypatch,
):
    class ExistingCombinedBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def ensure_accessible(self):
            pass

        def collection_supports(self, collection_name, mode):
            return mode == "vector"

    monkeypatch.setattr("bench.indexing.QdrantBackend", ExistingCombinedBackend)
    monkeypatch.setattr(
        "bench.indexing.read_jsonl",
        lambda path: pytest.fail("guard must run before loading documents"),
    )
    cfg = AppConfig()

    with pytest.raises(RuntimeError, match="removes its dense vector field"):
        _run_index_qdrant(cfg, mode="fulltext")
