from __future__ import annotations

import curses
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from .config import AppConfig, load_engine_targets
from .discovery import discover_config_paths
from .table_filters import (
    draw_header_with_filters,
    filter_block_height,
    filter_rows,
    fit_text,
    prompt_filter_value,
)

_fit = fit_text


@dataclass
class PrepareVariant:
    config_path: str
    entry: str
    source: str
    prepared_dir: str
    cfg: AppConfig
def build_prepare_variants_from_config_paths(paths: list[Path]) -> list[PrepareVariant]:
    out: list[PrepareVariant] = []
    for path in paths:
        try:
            targets = load_engine_targets(path)
        except Exception:
            continue
        for t in targets:
            out.append(
                PrepareVariant(
                    config_path=str(path),
                    entry=t.name,
                    source=t.cfg.dataset.source,
                    prepared_dir=t.cfg.dataset.prepared_dir,
                    cfg=t.cfg,
                )
            )

    # Keep one row per dataset source + prepared_dir, aggregating entry labels.
    merged: dict[tuple[str, str], PrepareVariant] = {}
    entries: dict[tuple[str, str], set[str]] = {}
    for v in out:
        key = (v.source, v.prepared_dir)
        if key not in merged:
            merged[key] = v
            entries[key] = set()
        entries[key].add(v.entry)
    variants: list[PrepareVariant] = []
    for key, v in merged.items():
        names = sorted(entries.get(key, set()))
        entry_label = ", ".join(names) if names else v.entry
        variants.append(
            PrepareVariant(
                config_path=v.config_path,
                entry=entry_label,
                source=v.source,
                prepared_dir=v.prepared_dir,
                cfg=v.cfg,
            )
        )
    return sorted(variants, key=lambda x: (x.source.lower(), x.prepared_dir.lower(), x.entry.lower()))


def _draw_row(
    stdscr: Any,
    y: int,
    v: PrepareVariant,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    x = 0
    stdscr.addstr(y, x, _fit("[x]" if selected else "[ ]", widths["sel"]), attrs)
    x += widths["sel"] + 1
    if curses.has_colors():
        stdscr.addstr(y, x, _fit(v.source, widths["source"]), attrs | curses.color_pair(1))
    else:
        stdscr.addstr(y, x, _fit(v.source, widths["source"]), attrs)
    x += widths["source"] + 1
    stdscr.addstr(y, x, _fit(v.prepared_dir, widths["prepared"]), attrs)
    x += widths["prepared"] + 1
    stdscr.addstr(y, x, _fit(v.entry, widths["entry"]), attrs)


def _select_prepare_variants_curses(variants: list[PrepareVariant]) -> list[PrepareVariant]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    all_variants = list(variants)

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset, variants, selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)  # source
        sort_cols = ["source", "prepared", "entry"]
        sort_idx = 0
        sort_key = sort_cols[sort_idx]
        filters: dict[str, str] = {}

        def _variant_id(v: PrepareVariant) -> tuple[str, str, str]:
            return (v.source, v.prepared_dir, v.entry)

        def _col_value(v: PrepareVariant, col: str) -> str:
            if col == "source":
                return v.source
            if col == "prepared":
                return v.prepared_dir
            if col == "entry":
                return v.entry
            return ""

        def _apply_view() -> None:
            nonlocal variants, selected_idx, current, offset
            selected_ids = {_variant_id(variants[i]) for i in selected_idx if 0 <= i < len(variants)}
            current_id = _variant_id(variants[current]) if 0 <= current < len(variants) else None
            variants = filter_rows(all_variants, filters, _col_value)
            variants.sort(key=lambda v: _col_value(v, sort_key).lower())
            selected_idx = {i for i, v in enumerate(variants) if _variant_id(v) in selected_ids}
            if current_id is not None:
                for i, v in enumerate(variants):
                    if _variant_id(v) == current_id:
                        current = i
                        break
            current = min(max(0, current), max(0, len(variants) - 1))
            offset = 0

        _apply_view()

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            filter_lines = filter_block_height(filters, widths={})
            stdscr.addstr(
                0,
                0,
                fit_text("↑/↓ move  ←/→ column  SPACE toggle  f filter  ENTER prepare selected  ESC/q cancel", max(1, w - 1)),
                curses.A_BOLD,
            )

            widths = {
                "sel": 3,
                "source": max(20, int(w * 0.36)),
                "prepared": max(24, int(w * 0.34)),
                "entry": max(18, w - 3 - 1 - max(20, int(w * 0.36)) - 1 - max(24, int(w * 0.34)) - 1 - 1),
            }
            filter_lines = filter_block_height(filters, widths)
            table_w = sum(widths.values()) + 3
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[("SEL", "sel"), ("SOURCE", "source"), ("PREPARED DIR", "prepared"), ("PROFILES", "entry")],
                widths=widths,
                max_width=table_w,
                active_col=sort_key,
                sort_desc=None,
                filters=filters,
                filter_lines=filter_lines,
            )
            rows_y = 2 + filter_lines
            avail_rows = max(1, h - rows_y - 2)

            if current < offset:
                offset = current
            if current >= offset + avail_rows:
                offset = current - avail_rows + 1

            for ridx in range(offset, min(len(variants), offset + avail_rows)):
                _draw_row(
                    stdscr,
                    y=rows_y + (ridx - offset),
                    v=variants[ridx],
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                )

            stdscr.addstr(
                h - 1,
                0,
                fit_text(f"Selected: {len(selected_idx)} / {len(variants)}", max(1, w - 1)),
                curses.A_DIM,
            )
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                current = max(0, current - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                current = min(len(variants) - 1, current + 1)
            elif ch == curses.KEY_LEFT:
                sort_idx = (sort_idx - 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_view()
            elif ch == curses.KEY_RIGHT:
                sort_idx = (sort_idx + 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_view()
            elif ch == ord(" "):
                if current in selected_idx:
                    selected_idx.remove(current)
                else:
                    selected_idx.add(current)
            elif ch in (ord("f"), ord("F")):
                val = prompt_filter_value(
                    stdscr,
                    prompt=f"Filter {sort_key} (empty clears): ",
                    initial=str(filters.get(sort_key, "") or ""),
                )
                if val is not None:
                    val = val.strip()
                    if val:
                        filters[sort_key] = val
                    else:
                        filters.pop(sort_key, None)
                    _apply_view()
            elif ch in (10, 13, curses.KEY_ENTER):
                return sorted(selected_idx)
            elif ch == 27 or ch in (ord("q"),):
                return []

    idxs = curses.wrapper(_inner)
    return [variants[i] for i in idxs]


def _parse_selection(raw: str, n: int) -> list[int]:
    out: set[int] = set()
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                x = int(a)
                y = int(b)
            except ValueError:
                continue
            lo, hi = min(x, y), max(x, y)
            for i in range(lo, hi + 1):
                if 1 <= i <= n:
                    out.add(i)
        else:
            try:
                i = int(p)
            except ValueError:
                continue
            if 1 <= i <= n:
                out.add(i)
    return sorted(out)


def select_prepare_variants_interactive(variants: list[PrepareVariant]) -> list[PrepareVariant]:
    if not variants:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_prepare_variants_curses(variants)
        except Exception:
            pass
    print("Select dataset sources to prepare (comma or ranges, e.g. 1,3-5):")
    for i, v in enumerate(variants, start=1):
        print(f"{i}. source={v.source}  prepared_dir={v.prepared_dir}  profiles={v.entry}")
    raw = input("Selection (empty to cancel): ").strip()
    if not raw:
        return []
    idxs = _parse_selection(raw, len(variants))
    return [variants[i - 1] for i in idxs]
