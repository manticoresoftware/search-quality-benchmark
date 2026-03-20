from __future__ import annotations

import curses
from typing import Any, Callable


def fit_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def wrap_text(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text[: max(0, width)]]
    src = str(text or "")
    if not src:
        return [""]
    out: list[str] = []
    while src:
        if len(src) <= width:
            out.append(src)
            break
        split_at = src.rfind(" ", 0, width + 1)
        if split_at <= 0:
            split_at = width
        out.append(src[:split_at].rstrip())
        src = src[split_at:].lstrip()
    return out or [""]


def matches_filters(
    row: Any,
    filters: dict[str, str],
    value_getter: Callable[[Any, str], Any],
) -> bool:
    for col, needle in filters.items():
        raw = str(needle or "").strip().lower()
        if not raw:
            continue
        hay = str(value_getter(row, col) or "").lower()
        if raw not in hay:
            return False
    return True


def filter_rows(
    rows: list[Any],
    filters: dict[str, str],
    value_getter: Callable[[Any, str], Any],
) -> list[Any]:
    return [row for row in rows if matches_filters(row, filters, value_getter)]


def prompt_filter_value(
    stdscr: Any,
    *,
    prompt: str,
    initial: str = "",
) -> str | None:
    h, w = stdscr.getmaxyx()
    y = max(0, h - 1)
    prompt_text = prompt[: max(0, w - 1)]
    curses.echo()
    try:
        stdscr.move(y, 0)
        stdscr.clrtoeol()
        stdscr.addstr(y, 0, prompt_text, curses.A_BOLD)
        stdscr.refresh()
        raw = stdscr.getstr(y, min(len(prompt_text), max(0, w - 1)), max(0, w - len(prompt_text) - 1))
    except KeyboardInterrupt:
        return None
    finally:
        curses.noecho()
    try:
        return raw.decode("utf-8")
    except Exception:
        return None


def draw_header_with_filters(
    stdscr: Any,
    *,
    y: int,
    cols: list[tuple[str, str]],
    widths: dict[str, int],
    max_width: int,
    active_col: str,
    sort_desc: bool | None,
    filters: dict[str, str],
    filter_lines: int,
) -> None:
    x = 0
    for title, key in cols:
        if x >= max_width:
            break
        label = title
        attrs = curses.A_UNDERLINE | curses.A_BOLD
        if key == active_col:
            if sort_desc is None:
                label = f"{title}*"
            else:
                label = f"{title}{'↓' if sort_desc else '↑'}"
            attrs |= curses.A_REVERSE
        stdscr.addstr(y, x, fit_text(label, widths[key]), attrs)
        filter_text = str(filters.get(key, "") or "").strip()
        wrapped = wrap_text(filter_text, widths[key]) if filter_text else []
        for i in range(filter_lines):
            line = wrapped[i] if i < len(wrapped) else ""
            stdscr.addstr(y + 1 + i, x, fit_text(line, widths[key]), curses.A_DIM)
        x += widths[key] + 1


def filter_block_height(filters: dict[str, str], widths: dict[str, int]) -> int:
    max_lines = 0
    for key, value in filters.items():
        filter_text = str(value or "").strip()
        if not filter_text:
            continue
        width = int(widths.get(key, 0))
        if width <= 0:
            continue
        max_lines = max(max_lines, len(wrap_text(filter_text, width)))
    return max_lines
