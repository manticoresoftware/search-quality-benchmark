from __future__ import annotations

import io
import logging
from typing import Any

import pytest

from bench.config import AppConfig
from bench.engines import _engine_ready


def test_milvus_readiness_failure_does_not_leak_library_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("pymilvus.test_probe")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    class _Backend:
        def __init__(self, cfg: Any) -> None:
            logger.error("grpc traceback that would corrupt curses")
            raise RuntimeError("Milvus connection reset")

    monkeypatch.setattr("bench.engines.MilvusBackend", _Backend)
    try:
        ready, error = _engine_ready(AppConfig(), "milvus")
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    assert ready is False
    assert error == "Milvus connection reset"
    assert output.getvalue() == ""
