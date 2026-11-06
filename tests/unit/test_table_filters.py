from typing import Any

from bench import table_filters
from bench.table_filters import filter_block_height, filter_rows, wrap_text


class _FilterPromptScreen:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def getmaxyx(self) -> tuple[int, int]:
        return (24, 80)

    def timeout(self, delay: int) -> None:
        self.calls.append(("timeout", delay))

    def move(self, y: int, x: int) -> None:
        self.calls.append(("move", (y, x)))

    def clrtoeol(self) -> None:
        pass

    def addstr(self, *_args: Any) -> None:
        pass

    def refresh(self) -> None:
        pass

    def getstr(self, *_args: Any) -> bytes:
        self.calls.append(("getstr", None))
        return b"manticore"


def test_filter_rows_case_insensitive_substring():
    rows = [{"engine": "manticore"}, {"engine": "qdrant"}]
    out = filter_rows(rows, {"engine": "tic"}, lambda row, col: row.get(col, ""))
    assert out == [{"engine": "manticore"}]


def test_filter_block_height_wraps_long_filter():
    widths = {"model": 8}
    filters = {"model": "openai text embedding"}
    assert filter_block_height(filters, widths) == len(wrap_text(filters["model"], 8))


def test_filter_prompt_restores_blocking_input_before_read(monkeypatch: Any) -> None:
    screen = _FilterPromptScreen()
    monkeypatch.setattr(table_filters.curses, "echo", lambda: None)
    monkeypatch.setattr(table_filters.curses, "noecho", lambda: None)

    value = table_filters.prompt_filter_value(screen, prompt="Filter engine: ")

    assert value == "manticore"
    assert screen.calls.index(("timeout", -1)) < screen.calls.index(("getstr", None))
