from __future__ import annotations

import sys
from typing import Any

import pytest

from bench import cli
from bench.config import AppConfig


class _BellStream:
    def __init__(self) -> None:
        self.value = ""
        self.flushed = False

    def isatty(self) -> bool:
        return True

    def write(self, value: str) -> None:
        self.value += value

    def flush(self) -> None:
        self.flushed = True


class _TTYInput:
    def isatty(self) -> bool:
        return True


class _MenuScreen:
    def __init__(self) -> None:
        self.timeouts: list[int] = []

    def keypad(self, _enabled: bool) -> None:
        pass

    def timeout(self, delay: int) -> None:
        self.timeouts.append(delay)

    def erase(self) -> None:
        pass

    def getmaxyx(self) -> tuple[int, int]:
        return (24, 100)

    def addstr(self, *_args: Any) -> None:
        pass

    def refresh(self) -> None:
        pass

    def getch(self) -> int:
        return ord("q")


def test_terminal_bell_writes_and_flushes_bel(monkeypatch: Any) -> None:
    stream = _BellStream()
    monkeypatch.setattr(cli.sys, "stdout", stream)

    cli._terminal_bell()

    assert stream.value == "\a"
    assert stream.flushed is True


def test_main_menu_clears_timeout_inherited_from_engines(monkeypatch: Any) -> None:
    screen = _MenuScreen()
    monkeypatch.setattr(cli.sys, "stdin", _TTYInput())
    monkeypatch.setattr(cli.sys, "stdout", _BellStream())
    monkeypatch.setattr(cli.curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(cli.curses, "has_colors", lambda: False)
    monkeypatch.setattr(cli.curses, "wrapper", lambda callback: callback(screen))

    assert cli._select_top_command_interactive() is None
    assert screen.timeouts == [-1]


def test_no_argument_tui_returns_to_main_menu_after_submenus(monkeypatch: Any) -> None:
    selections = iter(["engines", "prepare", None])
    commands: list[str] = []

    monkeypatch.setattr(sys, "argv", ["bench"])
    monkeypatch.setattr(cli, "resolve_config_path", lambda path: path)
    monkeypatch.setattr(cli, "load_config", lambda _path: AppConfig())
    monkeypatch.setattr(cli, "_select_top_command_interactive", lambda: next(selections))
    monkeypatch.setattr(
        cli,
        "_terminal_bell",
        lambda: pytest.fail("a canceled or non-task submenu must not ring"),
    )
    monkeypatch.setattr(
        cli,
        "cmd_engines",
        lambda _cfg, args: commands.append(args.command) or 1,
    )
    monkeypatch.setattr(
        cli,
        "cmd_prepare",
        lambda _cfg, args: commands.append(args.command) or 0,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert commands == ["engines", "prepare"]


def test_long_task_bells_before_main_menu_is_shown_again(monkeypatch: Any) -> None:
    selections = iter(["index", None])
    events: list[str] = []

    def select_command() -> str | None:
        events.append("menu")
        return next(selections)

    def run_index(_cfg: AppConfig, args: Any) -> int:
        events.append("index")
        cli._mark_long_task_started(args)
        return 0

    monkeypatch.setattr(sys, "argv", ["bench"])
    monkeypatch.setattr(cli, "resolve_config_path", lambda path: path)
    monkeypatch.setattr(cli, "load_config", lambda _path: AppConfig())
    monkeypatch.setattr(cli, "_select_top_command_interactive", select_command)
    monkeypatch.setattr(cli, "cmd_index", run_index)
    monkeypatch.setattr(cli, "_terminal_bell", lambda: events.append("bell"))

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert events == ["menu", "index", "bell", "menu"]


def test_explicit_subcommand_remains_one_shot(monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setattr(sys, "argv", ["bench", "engines"])
    monkeypatch.setattr(cli, "resolve_config_path", lambda path: path)
    monkeypatch.setattr(cli, "load_config", lambda _path: AppConfig())
    monkeypatch.setattr(
        cli,
        "_select_top_command_interactive",
        lambda: pytest.fail("explicit subcommands must not open the main menu"),
    )
    monkeypatch.setattr(
        cli,
        "_terminal_bell",
        lambda: pytest.fail("explicit subcommands must not ring the menu bell"),
    )
    monkeypatch.setattr(
        cli,
        "cmd_engines",
        lambda _cfg, args: calls.append(args.command) or 7,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 7
    assert calls == ["engines"]
