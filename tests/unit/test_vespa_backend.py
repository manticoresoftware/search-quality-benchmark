from bench.config import VespaConfig
from bench.vespa_backend import VespaBackend, _sanitize_vespa_string


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
