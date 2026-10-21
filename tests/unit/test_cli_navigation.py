from __future__ import annotations

import sys
from typing import Any

import pytest

from bench import cli
from bench.config import AppConfig


def test_no_argument_tui_returns_to_main_menu_after_submenus(monkeypatch: Any) -> None:
    selections = iter(["engines", "prepare", None])
    commands: list[str] = []

    monkeypatch.setattr(sys, "argv", ["bench"])
    monkeypatch.setattr(cli, "resolve_config_path", lambda path: path)
    monkeypatch.setattr(cli, "load_config", lambda _path: AppConfig())
    monkeypatch.setattr(cli, "_select_top_command_interactive", lambda: next(selections))
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
        "cmd_engines",
        lambda _cfg, args: calls.append(args.command) or 7,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 7
    assert calls == ["engines"]
