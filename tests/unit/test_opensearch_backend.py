from typing import Any

import pytest

from bench.config import AppConfig, OpenSearchConfig
from bench.index_matrix import _model_opts_label
from bench.opensearch_backend import OpenSearchBackend
from bench.run_matrix import _opensearch_hybrid_label
from bench.runbench import _opensearch_hybrid_label as _artifact_hybrid_label


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def perform_request(
        self, method: str, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((method, path, body))
        return {"acknowledged": True}


class _FakeClient:
    def __init__(self) -> None:
        self.transport = _FakeTransport()
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_source": {"doc_id": "doc-1"},
                        "_score": 0.75,
                    }
                ]
            }
        }


def _backend(cfg: OpenSearchConfig) -> tuple[OpenSearchBackend, _FakeClient]:
    backend = object.__new__(OpenSearchBackend)
    client = _FakeClient()
    backend.cfg = cfg
    backend.client = client
    backend._configured_hybrid_pipelines = set()
    return backend, client


def test_opensearch_hybrid_uses_native_weighted_rrf_pipeline() -> None:
    backend, client = _backend(
        OpenSearchConfig(
            index_name="docs",
            vector_field="embedding",
            hybrid_text_boost=1.0,
            hybrid_vector_boost=3.0,
            hybrid_rrf_k=61,
            hybrid_candidates=40,
        )
    )

    rows = backend.search_hybrid("test query", [0.25, 0.5], limit=10)
    backend.search_hybrid("test query", [0.25, 0.5], limit=10)

    assert rows == [{"doc_id": "doc-1", "score": 0.75}]
    assert len(client.transport.calls) == 1
    method, path, pipeline = client.transport.calls[0]
    assert method == "PUT"
    assert path.startswith("/_search/pipeline/search-quality-benchmark-hybrid-rrf-")
    combination = pipeline["phase_results_processors"][0][
        "score-ranker-processor"
    ]["combination"]
    assert combination == {
        "technique": "rrf",
        "rank_constant": 61,
        "parameters": {"weights": [0.25, 0.75]},
    }

    assert len(client.search_calls) == 2
    request = client.search_calls[0]
    assert request["index"] == "docs"
    assert request["params"]["search_pipeline"] == path.rsplit("/", 1)[-1]
    hybrid = request["body"]["query"]["hybrid"]
    assert hybrid["pagination_depth"] == 40
    assert hybrid["queries"][0] == {
        "multi_match": {
            "query": "test query",
            "fields": ["title", "abstract"],
        }
    }
    assert hybrid["queries"][1]["knn"]["embedding"] == {
        "vector": [0.25, 0.5],
        "k": 40,
    }


@pytest.mark.parametrize(
    ("text_weight", "vector_weight"),
    [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)],
)
def test_opensearch_rejects_invalid_rrf_weights(
    text_weight: float, vector_weight: float
) -> None:
    backend, _ = _backend(
        OpenSearchConfig(
            hybrid_text_boost=text_weight,
            hybrid_vector_boost=vector_weight,
        )
    )

    with pytest.raises(ValueError, match="weights"):
        backend.ensure_hybrid_pipeline()


@pytest.mark.parametrize("rrf_k", [0, 10001])
def test_opensearch_rejects_invalid_rrf_k(rrf_k: int) -> None:
    backend, _ = _backend(OpenSearchConfig(hybrid_rrf_k=rrf_k))

    with pytest.raises(ValueError, match="between 1 and 10000"):
        backend.ensure_hybrid_pipeline()


def test_opensearch_native_rrf_is_recorded_in_labels() -> None:
    cfg = AppConfig(
        opensearch=OpenSearchConfig(
            hybrid_text_boost=2.0,
            hybrid_vector_boost=1.0,
            hybrid_rrf_k=42,
            hybrid_candidates=150,
        )
    )

    expected = "fusion=native-rrf k=42 weights=2:1 candidates=150"
    assert expected in _opensearch_hybrid_label(cfg)
    assert expected in _artifact_hybrid_label(cfg)
    assert "native-rrf k=42 weights=2:1 candidates=150" in _model_opts_label(
        cfg, "opensearch", "hybrid"
    )
