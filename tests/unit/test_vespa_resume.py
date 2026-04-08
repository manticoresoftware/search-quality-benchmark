import pytest

from bench.cli import build_parser
from bench.config import AppConfig
from bench.indexing import _resolve_vespa_resume_offset, _run_index_vespa


def _docs_cached(total: int) -> list[tuple[dict[str, object], str]]:
    return [({"id": idx, "doc_id": f"doc-{idx}", "title": "", "abstract": ""}, f"k{idx}") for idx in range(total)]


def test_index_parser_accepts_resume_flag():
    parser = build_parser()
    args = parser.parse_args(["index", "--engine", "vespa", "--mode", "vector", "--resume"])
    assert args.resume is True


def test_resolve_vespa_resume_offset_verifies_boundary_docs():
    docs_cached = _docs_cached(10)

    class FakeBackend:
        def __init__(self):
            self.checked: list[str] = []

        def get_document_count(self, collection_name):
            return 8

        def collection_name(self):
            return "demo"

        def document_exists(self, doc_id):
            self.checked.append(doc_id)
            return True

        def recreate_collection(self, *, dimension):
            raise AssertionError("should not recreate when resuming")

    backend = FakeBackend()
    offset = _resolve_vespa_resume_offset(
        backend=backend,
        docs_cached=docs_cached,
        total_cached=len(docs_cached),
        dimension=3,
    )

    assert offset == 8
    assert backend.checked == ["doc-3", "doc-6", "doc-7"]


def test_resolve_vespa_resume_offset_rejects_missing_boundary_docs():
    docs_cached = _docs_cached(10)

    class FakeBackend:
        def get_document_count(self, collection_name):
            return 8

        def collection_name(self):
            return "demo"

        def document_exists(self, doc_id):
            return doc_id != "doc-6"

        def recreate_collection(self, *, dimension):
            raise AssertionError("should not recreate when resuming")

    with pytest.raises(RuntimeError, match="boundary verification failed"):
        _resolve_vespa_resume_offset(
            backend=FakeBackend(),
            docs_cached=docs_cached,
            total_cached=len(docs_cached),
            dimension=3,
        )


def test_run_index_vespa_vector_resume_skips_existing_docs(monkeypatch):
    docs = [{"id": idx, "doc_id": f"doc-{idx}", "title": "", "abstract": ""} for idx in range(5)]
    docs_cached = [(doc, f"k{doc['id']}") for doc in docs]
    imported: list[str] = []

    class FakeVectorCache:
        def __init__(self, path):
            self.path = path

        def get_vectors_many(self, keys):
            return {str(k): [0.1, 0.2] for k in keys}

        def close(self):
            return None

    class FakeBackend:
        def __init__(self, cfg):
            self.cfg = cfg
            self.recreated = False

        def ensure_accessible(self):
            return None

        def get_document_count(self, collection_name):
            return 2

        def collection_name(self):
            return "demo"

        def document_exists(self, doc_id):
            return doc_id in {"doc-0", "doc-1"}

        def recreate_collection(self, *, dimension=None):
            self.recreated = True

        def import_documents(self, batch):
            imported.extend(str(row["doc_id"]) for row in batch)
            return len(batch)

    monkeypatch.setattr("bench.indexing.read_jsonl", lambda path: docs)
    monkeypatch.setattr("bench.indexing._collect_cached_docs", lambda **kwargs: (docs_cached, 0, []))
    monkeypatch.setattr("bench.indexing.VectorCache", FakeVectorCache)
    monkeypatch.setattr("bench.indexing.VespaBackend", FakeBackend)

    cfg = AppConfig()
    cfg.dataset.prepared_dir = "/tmp/unused"
    cfg.vespa.batch_size = 2
    cfg.vespa.index_concurrency = 1

    results = _run_index_vespa(cfg, mode="vector", resume=True)

    assert [r.rows_inserted for r in results] == [5]
    assert imported == ["doc-2", "doc-3", "doc-4"]
