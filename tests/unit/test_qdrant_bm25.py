from bench.qdrant_backend import QdrantBackend


def test_payload_too_large_error_detection():
    assert QdrantBackend._is_payload_too_large_error(
        Exception("Payload error: JSON payload (73236200 bytes) is larger than allowed (limit: 33554432 bytes).")
    )
    assert not QdrantBackend._is_payload_too_large_error(Exception("connection timeout"))
