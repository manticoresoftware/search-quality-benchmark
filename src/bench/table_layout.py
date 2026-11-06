from __future__ import annotations

from .table_filters import fit_text


def fit_column_widths(
    preferred: dict[str, int],
    minimum: dict[str, int],
    max_width: int,
) -> dict[str, int]:
    """Shrink table columns to fit without letting curses wrap a row."""
    keys = list(preferred)
    if not keys:
        return {}

    content_budget = max(0, int(max_width) - (len(keys) - 1))
    widths = {key: 0 for key in keys}

    # Give every column a cell when the terminal is wide enough, then grow all
    # columns fairly toward their minimum and preferred sizes.
    for key in keys:
        if content_budget <= 0:
            break
        widths[key] = 1
        content_budget -= 1

    for limits in (minimum, preferred):
        while content_budget > 0:
            grew = False
            for key in keys:
                limit = max(0, int(limits.get(key, preferred[key])))
                if widths[key] >= limit:
                    continue
                widths[key] += 1
                content_budget -= 1
                grew = True
                if content_budget <= 0:
                    break
            if not grew:
                break
    return widths


def wrap_text_lines(text: str, width: int) -> list[str]:
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


def wrap_cell(text: str, width: int, max_lines: int = 2) -> list[str]:
    if max_lines <= 1:
        return [fit_text(text, width)]
    src = str(text or "")
    out: list[str] = []
    while src and len(out) < max_lines:
        if len(src) <= width:
            out.append(src.ljust(width))
            src = ""
            break
        if len(out) == max_lines - 1:
            out.append(fit_text(src, width))
            src = ""
            break
        out.append(src[:width].ljust(width))
        src = src[width:]
    while len(out) < max_lines:
        out.append("".ljust(width))
    return out


def dynamic_row_lines(total_rows: int, term_height: int) -> int:
    available = max(1, term_height - 4)
    target_rows = max(1, min(total_rows, 10))
    lines = available // target_rows
    if lines < 1:
        return 1
    if lines > 3:
        return 3
    return lines
