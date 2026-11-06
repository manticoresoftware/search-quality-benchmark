from __future__ import annotations

import curses
from typing import Any

from bench.config import AppConfig
from bench.index_matrix import IndexVariant, TargetStatus, _select_index_variants_curses
from bench.table_layout import fit_column_widths


class _FakeScreen:
    def __init__(self, keys: list[int], *, height: int = 24, width: int = 80) -> None:
        self._keys = iter(keys)
        self.height = height
        self.width = width
        self.rendered: list[str] = []

    def keypad(self, _enabled: bool) -> None:
        pass

    def erase(self) -> None:
        pass

    def getmaxyx(self) -> tuple[int, int]:
        return self.height, self.width

    def addstr(self, y: int, x: int, value: str, *_attrs: Any) -> None:
        assert 0 <= y < self.height
        assert 0 <= x < self.width
        assert x + len(value) < self.width
        self.rendered.append(value)

    def refresh(self) -> None:
        pass

    def getch(self) -> int:
        return next(self._keys)


def _variant(*, mode: str = "hybrid", target_kind: str = "hybrid") -> IndexVariant:
    return IndexVariant(
        config_path="config.yaml",
        entry="arguana_manticore_hybrid",
        engine="manticore",
        mode=mode,
        model="profile:relevance",
        batch_size=250,
        target="arguana_manticore",
        cfg=AppConfig(),
        target_kind=target_kind,
        used_by=("arguana_manticore_fulltext", "arguana_manticore_vector"),
    )


def _install_fake_curses(monkeypatch: Any, screen: _FakeScreen) -> None:
    monkeypatch.setattr("bench.index_matrix.curses.curs_set", lambda _value: None)
    monkeypatch.setattr("bench.index_matrix.curses.has_colors", lambda: False)
    monkeypatch.setattr("bench.index_matrix.curses.wrapper", lambda callback: callback(screen))


def test_fit_column_widths_keeps_table_inside_terminal() -> None:
    preferred = {"sel": 3, "engine": 10, "target": 22, "model": 38}
    minimum = {"sel": 3, "engine": 8, "target": 12, "model": 12}

    widths = fit_column_widths(preferred, minimum, max_width=50)

    assert list(widths) == list(preferred)
    assert sum(widths.values()) + len(widths) - 1 <= 50
    assert all(widths[key] >= minimum[key] for key in widths)


def test_index_picker_renders_and_toggles_at_80_columns(monkeypatch: Any) -> None:
    screen = _FakeScreen([ord(" "), 10])
    _install_fake_curses(monkeypatch, screen)
    monkeypatch.setattr(
        "bench.index_matrix._resolve_target_status",
        lambda _variant: TargetStatus("pending", None, 8674),
    )
    variant = _variant()

    assert _select_index_variants_curses([variant]) == [variant]


def test_index_picker_ignores_selection_keys_when_no_engine_is_available(monkeypatch: Any) -> None:
    screen = _FakeScreen([ord(" "), 10])
    _install_fake_curses(monkeypatch, screen)
    monkeypatch.setattr(
        "bench.index_matrix._resolve_target_status",
        lambda _variant: TargetStatus("unavailable", None, 8674, "connection refused"),
    )

    assert _select_index_variants_curses([_variant()]) == []
    assert any("No available index targets" in line for line in screen.rendered)


def test_index_picker_selecting_hybrid_deselects_shared_fulltext(monkeypatch: Any) -> None:
    screen = _FakeScreen([ord(" "), curses.KEY_DOWN, ord(" "), 10])
    _install_fake_curses(monkeypatch, screen)
    monkeypatch.setattr(
        "bench.index_matrix._resolve_target_status",
        lambda _variant: TargetStatus("pending", None, 8674),
    )
    fulltext = _variant(mode="fulltext", target_kind="fulltext")
    hybrid = _variant()

    assert _select_index_variants_curses([fulltext, hybrid]) == [hybrid]


def test_index_picker_selecting_fulltext_deselects_shared_hybrid(monkeypatch: Any) -> None:
    screen = _FakeScreen([curses.KEY_DOWN, ord(" "), curses.KEY_UP, ord(" "), 10])
    _install_fake_curses(monkeypatch, screen)
    monkeypatch.setattr(
        "bench.index_matrix._resolve_target_status",
        lambda _variant: TargetStatus("pending", None, 8674),
    )
    fulltext = _variant(mode="fulltext", target_kind="fulltext")
    hybrid = _variant()

    assert _select_index_variants_curses([fulltext, hybrid]) == [fulltext]
