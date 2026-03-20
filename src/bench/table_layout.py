from __future__ import annotations

from .table_filters import fit_text


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
