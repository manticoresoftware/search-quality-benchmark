import socket
from urllib import error as urlerror

from bench.config import VespaConfig
from bench.vespa_backend import VespaBackend, _sanitize_vespa_query_text, _sanitize_vespa_string


def test_sanitize_vespa_string_removes_illegal_control_chars():
    assert _sanitize_vespa_string("abc\u0010def\tghi\njkl\r") == "abcdef\tghi\njkl\r"


def test_import_documents_sanitizes_string_fields(monkeypatch):
    backend = VespaBackend(VespaConfig(collection_name="fiqa_vespa"))
    captured: list[dict] = []

    def fake_request(method, base_url, path, *, params=None, payload=None, content_type="application/json"):
        captured.append(payload)
        return {}

    monkeypatch.setattr(backend, "_request", fake_request)

    inserted = backend.import_documents(
        [
            {
                "id": 13363,
                "doc_id": "137406",
                "title": "title\u0001",
                "abstract": "bad\u0010text",
                "doi": "\u0000doi",
                "date": "2024-01-01\u001f",
            }
        ]
    )

    assert inserted == 1
    assert captured == [
        {
            "fields": {
                "id": 13363,
                "doc_id": "137406",
                "title": "title",
                "abstract": "badtext",
                "doi": "doi",
                "date": "2024-01-01",
            }
        }
    ]


def test_sanitize_vespa_query_text_removes_quote_phrase_syntax():
    raw = (
        '“If Major League Baseball was to employ a rule stating that runners must avoid contact with the catcher '
        'similar to the slide or avoid rule employed in amateur baseball it would give the advantage to the catcher '
        'The catcher would have the benefit of dictating the course of action that a baserunner must take and would '
        'perhaps more importantly have peace of mind knowing that there is no chance of an ensuing collision '
        'If Major League Baseball was to make a rule stating that the catcher cannot block the plate the advantage '
        'would certainly go to the baserunner who would enjoy the luxury of a straight path to the most sacred ground '
        'on a baseball diamond”'
    )

    sanitized = _sanitize_vespa_query_text(raw)

    assert '"' not in sanitized
    assert "'" not in sanitized
    assert "“" not in sanitized
    assert "”" not in sanitized
    assert "Major League Baseball" in sanitized


def test_search_lexical_sanitizes_query_before_user_query(monkeypatch):
    backend = VespaBackend(VespaConfig(collection_name="arguana_vespa"))
    captured: list[dict] = []

    def fake_query(payload):
        captured.append(payload)
        return {"root": {"children": []}}

    monkeypatch.setattr(backend, "_query", fake_query)

    backend.search_lexical('"the the the the the the the the the the the"', limit=5)

    assert captured == [
        {
            "yql": "select doc_id from sources arguana_vespa where userQuery();",
            "query": "the the the the the the the the the the the",
            "hits": 5,
            "ranking.profile": "bm25",
        }
    ]


def test_retryable_request_error_detects_timeouts():
    assert VespaBackend._is_retryable_request_error(TimeoutError("timed out"))
    assert VespaBackend._is_retryable_request_error(socket.timeout("timed out"))
    assert VespaBackend._is_retryable_request_error(urlerror.URLError(socket.timeout("timed out")))
    assert not VespaBackend._is_retryable_request_error(RuntimeError("boom"))


def test_request_retries_timeout_then_succeeds(monkeypatch):
    backend = VespaBackend(VespaConfig(timeout_sec=120, retry_attempts=3, retry_backoff_seconds=0.0))
    attempts = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise socket.timeout("timed out")
        return FakeResponse()

    monkeypatch.setattr("bench.vespa_backend.urlrequest.urlopen", fake_urlopen)

    resp = backend._request("GET", backend.connection_target(), "/test")

    assert resp == {"ok": True}
    assert attempts["count"] == 3
