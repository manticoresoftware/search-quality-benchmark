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
