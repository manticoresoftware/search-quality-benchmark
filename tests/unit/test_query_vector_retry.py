from __future__ import annotations

from bench.config import AppConfig
from bench.runbench import _get_query_vectors_from_cache_or_embed


class _FakeCache:
    def __init__(self, _path: str):
        self.saved_ok: dict[str, list[float]] = {}

    def get(self, _cache_key: str):
        class _Entry:
            status = "null"
            vector = None

        return _Entry()

    def upsert_ok(self, *, cache_key: str, model: str, similarity: str, text_value: str, vector: list[float]) -> None:
        self.saved_ok[cache_key] = vector

    def upsert_null(self, **kwargs) -> None:
        raise AssertionError("query null cache entries should be retried before persisting null again")

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeManticore:
    def __init__(self, _cfg):
        pass

    def embed_texts(self, texts, **kwargs):
        assert texts == ["what does qhs stand for"]
        return [[0.1, 0.2, 0.3]]

    def close(self) -> None:
        pass


def test_query_vector_retries_null_cache_entries(monkeypatch):
    monkeypatch.setattr("bench.runbench.VectorCache", _FakeCache)
    monkeypatch.setattr("bench.runbench.ManticoreClient", _FakeManticore)

    cfg = AppConfig()
    out = _get_query_vectors_from_cache_or_embed(cfg, ["what does qhs stand for"])
    assert out == {"what does qhs stand for": [0.1, 0.2, 0.3]}


class _FakeManticoreSecondTry:
    calls = 0

    def __init__(self, _cfg):
        pass

    def embed_texts(self, texts, **kwargs):
        _FakeManticoreSecondTry.calls += 1
        assert texts == ["how many tablespoons in one fourth cup?"]
        if _FakeManticoreSecondTry.calls == 1:
            return [[]]
        return [[0.4, 0.5]]

    def close(self) -> None:
        pass


def test_query_vector_retries_multiple_times(monkeypatch):
    monkeypatch.setattr("bench.runbench.VectorCache", _FakeCache)
    monkeypatch.setattr("bench.runbench.ManticoreClient", _FakeManticoreSecondTry)
    monkeypatch.setattr("bench.runbench.sleep", lambda _: None)

    cfg = AppConfig()
    out = _get_query_vectors_from_cache_or_embed(cfg, ["how many tablespoons in one fourth cup?"])
    assert out == {"how many tablespoons in one fourth cup?": [0.4, 0.5]}
    assert _FakeManticoreSecondTry.calls == 2


class _FakeManticoreFifthTry:
    calls = 0

    def __init__(self, _cfg):
        pass

    def embed_texts(self, texts, **kwargs):
        _FakeManticoreFifthTry.calls += 1
        assert texts == ["late success query"]
        if _FakeManticoreFifthTry.calls < 5:
            return [[]]
        return [[0.7, 0.8]]

    def close(self) -> None:
        pass


def test_query_vector_retries_up_to_five_times(monkeypatch):
    monkeypatch.setattr("bench.runbench.VectorCache", _FakeCache)
    monkeypatch.setattr("bench.runbench.ManticoreClient", _FakeManticoreFifthTry)
    monkeypatch.setattr("bench.runbench.sleep", lambda _: None)

    cfg = AppConfig()
    out = _get_query_vectors_from_cache_or_embed(cfg, ["late success query"])
    assert out == {"late success query": [0.7, 0.8]}
    assert _FakeManticoreFifthTry.calls == 5
