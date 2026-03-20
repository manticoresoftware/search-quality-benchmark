from __future__ import annotations

import curses
from dataclasses import asdict, is_dataclass
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
from .table_layout import dynamic_row_lines, wrap_cell, wrap_text_lines

_fit = fit_text


@dataclass
class IndexVariant:
    config_path: str
    entry: str
    engine: str
    mode: str
    model: str
    batch_size: int
    target: str
    cfg: AppConfig


def _display_mode(mode: str) -> str:
    if mode == "fulltext":
        return "full-text"
    if mode == "vector":
        return "vector"
    if mode == "both":
        return "both"
    return mode


def _model_opts_label(cfg: AppConfig, engine: str, mode: str) -> str:
    if mode == "vector":
        return cfg.manticore.vector_model
    if engine == "manticore":
        return (cfg.manticore.fulltext_options or "").strip() or "(none)"
    return "Qdrant/bm25"
def build_index_variants_from_config_paths(paths: list[Path]) -> list[IndexVariant]:
    out: list[IndexVariant] = []
    for path in paths:
        try:
            targets = load_engine_targets(path)
        except Exception:
            continue
        for target_cfg in targets:
            cfg = target_cfg.cfg
            engine = target_cfg.type
            modes = [target_cfg.mode_filter] if target_cfg.mode_filter in ("fulltext", "vector") else ["fulltext", "vector"]
            for mode in modes:
                if engine == "manticore":
                    target = cfg.manticore.fulltext_table if mode == "fulltext" else cfg.manticore.vector_table
                else:
                    target = cfg.qdrant.bm25_collection if mode == "fulltext" else cfg.qdrant.dense_collection
                out.append(
                    IndexVariant(
                        config_path=str(path),
                        entry=target_cfg.name,
                        engine=engine,
                        mode=mode,
                        model=_model_opts_label(cfg, engine=engine, mode=mode),
                        batch_size=(cfg.manticore.batch_size if engine == "manticore" else cfg.qdrant.batch_size),
                        target=target,
                        cfg=cfg,
                    )
                )

    dedup: dict[tuple[str, str, str, int, str], IndexVariant] = {}
    for v in out:
        key = (v.engine, v.mode, v.model, v.batch_size, v.target)
        if key in dedup:
            continue
        dedup[key] = v

    engine_order = {"manticore": 0, "qdrant": 1}
    mode_order = {"fulltext": 0, "vector": 1}
    return sorted(
        list(dedup.values()),
        key=lambda v: (engine_order.get(v.engine, 99), mode_order.get(v.mode, 99), v.entry, v.model, v.target),
    )


def _draw_row(
    stdscr: Any,
    y: int,
    v: IndexVariant,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    row_lines: int,
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    color_engine = 0
    color_mode = 0
    if curses.has_colors():
        if v.engine == "manticore":
            color_engine = curses.color_pair(1)
        elif v.engine == "qdrant":
            color_engine = curses.color_pair(2)
        if v.mode == "fulltext":
            color_mode = curses.color_pair(3)
        elif v.mode == "vector":
            color_mode = curses.color_pair(4)
        elif v.mode == "both":
            color_mode = curses.color_pair(5)

    x = 0
    stdscr.addstr(y, x, _fit("[x]" if selected else "[ ]", widths["sel"]), attrs)
    x += widths["sel"] + 1
    entry_lines = wrap_cell(v.entry, widths["entry"], max_lines=row_lines)
    stdscr.addstr(y, x, entry_lines[0], attrs)
    for i in range(1, row_lines):
        stdscr.addstr(y + i, x, entry_lines[i], attrs)
    x += widths["entry"] + 1
    stdscr.addstr(y, x, _fit(v.engine, widths["engine"]), attrs | color_engine)
    x += widths["engine"] + 1
    stdscr.addstr(y, x, _fit(_display_mode(v.mode), widths["mode"]), attrs | color_mode)
    x += widths["mode"] + 1
    stdscr.addstr(y, x, _fit(v.target, widths["target"]), attrs)
    x += widths["target"] + 1
    model_lines = wrap_cell(v.model, widths["model"], max_lines=row_lines)
    stdscr.addstr(y, x, model_lines[0], attrs)
    for i in range(1, row_lines):
        stdscr.addstr(y + i, x, model_lines[i], attrs)
    x += widths["model"] + 1
    stdscr.addstr(y, x, _fit(str(v.batch_size), widths["batch"]), attrs)


def _details_lines(v: IndexVariant) -> list[str]:
    def _mask_value(path: str, value: Any) -> Any:
        p = path.lower()
        if any(k in p for k in ("password", "api_key")) and str(value or "").strip():
            return "***"
        return value

    def _fmt(value: Any) -> str:
        if isinstance(value, list):
            return "[" + ", ".join(str(x) for x in value) + "]"
        if value is None:
            return "null"
        return str(value)

    def _flatten(prefix: str, obj: Any, out: list[str]) -> None:
        if is_dataclass(obj):
            obj = asdict(obj)
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                key = f"{prefix}.{k}" if prefix else str(k)
                _flatten(key, obj[k], out)
            return
        out.append(f"{prefix}: {_fmt(_mask_value(prefix, obj))}")

    lines: list[str] = [
        "Resolved Config",
        f"profile: {v.entry}",
        f"engine: {v.engine}",
        f"mode: {_display_mode(v.mode)}",
        f"target: {v.target}",
    ]
    flattened: list[str] = []
    visible_sections = {"dataset", "benchmark", "metrics", v.engine}
    cfg_obj = asdict(v.cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    _flatten("", filtered_cfg, flattened)
    lines.extend(flattened)
    return lines


def _draw_details_panel(stdscr: Any, v: IndexVariant, widths: dict[str, int], w: int, h: int) -> None:
    table_width = sum(widths.values()) + (len(widths) - 1)
    panel_x = table_width + 3
    panel_w = w - panel_x - 1
    if panel_w < 24:
        return
    lines = _details_lines(v)
    max_rows = max(0, h - 3)
    y = 1
    for i, line in enumerate(lines):
        if y > max_rows:
            break
        attrs = curses.A_BOLD if i == 0 else curses.A_DIM
        wrapped = wrap_text_lines(line, panel_w)
        for j, part in enumerate(wrapped):
            if y > max_rows:
                break
            part_attrs = attrs if j == 0 else curses.A_DIM
            stdscr.addstr(y, panel_x, _fit(part, panel_w), part_attrs)
            y += 1


def _select_index_variants_curses(variants: list[IndexVariant]) -> list[IndexVariant]:
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
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # manticore
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)    # full-text
            curses.init_pair(4, curses.COLOR_WHITE, -1)   # vector
            curses.init_pair(5, curses.COLOR_CYAN, -1)    # both

        widths = {
            "sel": 3,
            "entry": 18,
            "engine": 10,
            "mode": 9,
            "target": 22,
            "model": 48,
            "batch": 8,
        }
        sort_cols = ["entry", "engine", "mode", "target", "model", "batch"]
        sort_idx = 0
        sort_key = sort_cols[sort_idx]
        sort_desc = False
        filters: dict[str, str] = {}

        def _variant_id(v: IndexVariant) -> tuple[str, str, str, str, str, int]:
            return (v.entry, v.engine, v.mode, v.target, v.model, int(v.batch_size))

        def _mode_ord(mode: str) -> int:
            return 0 if mode == "fulltext" else (1 if mode == "vector" else 99)

        def _sort_value(v: IndexVariant, col: str) -> Any:
            if col == "entry":
                return v.entry.lower()
            if col == "engine":
                return v.engine.lower()
            if col == "mode":
                return _mode_ord(v.mode)
            if col == "target":
                return v.target.lower()
            if col == "model":
                return v.model.lower()
            if col == "batch":
                return int(v.batch_size)
            return v.entry.lower()

        def _filter_value(v: IndexVariant, col: str) -> Any:
            if col == "entry":
                return v.entry
            if col == "engine":
                return v.engine
            if col == "mode":
                return _display_mode(v.mode)
            if col == "target":
                return v.target
            if col == "model":
                return v.model
            if col == "batch":
                return v.batch_size
            return ""

        def _apply_sort() -> None:
            nonlocal variants, selected_idx, current, offset
            selected_ids = {_variant_id(variants[i]) for i in selected_idx if 0 <= i < len(variants)}
            current_id = _variant_id(variants[current]) if 0 <= current < len(variants) else None
            variants = filter_rows(all_variants, filters, _filter_value)
            variants.sort(key=lambda v: _sort_value(v, sort_key), reverse=sort_desc)
            selected_idx = {i for i, v in enumerate(variants) if _variant_id(v) in selected_ids}
            if current_id is not None:
                for i, v in enumerate(variants):
                    if _variant_id(v) == current_id:
                        current = i
                        break
            current = min(max(0, current), max(0, len(variants) - 1))
            offset = 0

        _apply_sort()
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            row_h = dynamic_row_lines(len(variants), h)
            stdscr.addstr(
                0,
                0,
                fit_text("↑/↓ move  ←/→ sort col  SPACE toggle  f filter  ENTER index selected  ESC/q cancel", max(1, w - 1)),
                curses.A_BOLD,
            )
            filter_lines = filter_block_height(filters, widths)
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[
                    ("SEL", "sel"),
                    ("PROFILE", "entry"),
                    ("ENGINE", "engine"),
                    ("MODE", "mode"),
                    ("TARGET", "target"),
                    ("MODEL/OPTS", "model"),
                    ("BATCH", "batch"),
                ],
                widths=widths,
                max_width=max(1, w - 1),
                active_col=sort_key,
                sort_desc=sort_desc,
                filters=filters,
                filter_lines=filter_lines,
            )
            rows_y = 2 + filter_lines
            visible = max(1, (h - rows_y - 2) // row_h)
            if current < offset:
                offset = current
            if current >= offset + visible:
                offset = current - visible + 1
            for i in range(visible):
                ridx = offset + i
                if ridx >= len(variants):
                    break
                _draw_row(
                    stdscr,
                    y=rows_y + i * row_h,
                    v=variants[ridx],
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    row_lines=row_h,
                )
            if variants:
                _draw_details_panel(stdscr, variants[current], widths=widths, w=w, h=h)
            stdscr.addstr(h - 1, 0, fit_text(f"Selected: {len(selected_idx)} / {len(variants)}", max(1, w - 1)), curses.A_DIM)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                current = max(0, current - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                current = min(len(variants) - 1, current + 1)
            elif ch == curses.KEY_LEFT:
                sort_idx = (sort_idx - 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_sort()
            elif ch == curses.KEY_RIGHT:
                sort_idx = (sort_idx + 1) % len(sort_cols)
                sort_key = sort_cols[sort_idx]
                _apply_sort()
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
                    _apply_sort()
            elif ch == ord(" "):
                if current in selected_idx:
                    selected_idx.remove(current)
                else:
                    selected_idx.add(current)
            elif ch in (10, 13, curses.KEY_ENTER):
                return sorted(selected_idx)
            elif ch == 27 or ch in (ord("q"),):
                return []

    picked = curses.wrapper(_inner)
    return [variants[i] for i in picked]


def select_index_variants_interactive(variants: list[IndexVariant]) -> list[IndexVariant]:
    if not variants:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_index_variants_curses(variants)
        except Exception:
            pass
    print("Available index runs:")
    for i, v in enumerate(variants, start=1):
        print(
            f"{i:>2}. {v.entry}\t{v.engine}\t{_display_mode(v.mode)}\t"
            f"target={v.target}\tmodel={v.model}\tbatch={v.batch_size}"
        )
    raw = input("Select numbers (e.g. 1,3-5): ").strip()
    idxs: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                if 1 <= x <= len(variants):
                    idxs.add(x - 1)
        else:
            x = int(part)
            if 1 <= x <= len(variants):
                idxs.add(x - 1)
    return [variants[i] for i in sorted(idxs)]
