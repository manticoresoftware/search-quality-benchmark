from __future__ import annotations

import io
import logging
from collections.abc import Callable

import pytest

from bench.index_matrix import _select_index_variants_curses
from bench.run_matrix import _select_variants_curses


@pytest.mark.parametrize(
    "selector",
    [_select_variants_curses, _select_index_variants_curses],
)
def test_curses_picker_suppresses_pymilvus_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    selector: Callable[[list[object]], list[object]],
) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("pymilvus.test_tui_picker")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    def fake_wrapper(_callback: object) -> list[int]:
        logger.error("RPC traceback that would corrupt curses")
        return []

    monkeypatch.setattr("curses.wrapper", fake_wrapper)
    try:
        assert selector([]) == []
        logger.error("visible after curses exits")
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    assert "RPC traceback" not in output.getvalue()
    assert "visible after curses exits" in output.getvalue()
