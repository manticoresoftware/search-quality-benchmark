from __future__ import annotations

import curses
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from .config import AppConfig, load_engine_targets
from .discovery import discover_config_paths
from .embedding_client import resolved_embedding_model
from .table_filters import (
    draw_header_with_filters,
    filter_block_height,
    filter_rows,
    fit_text,
    prompt_filter_value,
)
from .table_layout import wrap_text_lines

_fit = fit_text


@dataclass
class PrepareVariant:
    config_path: str
    entry: str
    source: str
    prepared_dir: str
    cfg: AppConfig
    embeddings_status: str = "-"
    cost_status: str = "-"


@dataclass
class PrepareSelection:
    action: str
    variants: list[PrepareVariant]


def _embedding_status(cfg: AppConfig, prepared_dir: str) -> tuple[str, str]:
    manifest_path = Path(prepared_dir) / "manifest.json"
    if not manifest_path.exists():
        return "-", "-"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "-", "-"

    model = resolved_embedding_model(cfg)
    provider = str(cfg.embedding.provider or "")
    same_model = str(payload.get("embedding_model") or "") == model
    same_provider = str(payload.get("embedding_provider") or "") == provider
    if not (same_model and same_provider):
        return "0%", "-"

    doc_ok = int(payload.get("doc_embeddings_ok") or 0)
    doc_total = int(payload.get("doc_embeddings_total") or 0)
    query_ok = int(payload.get("query_embeddings_ok") or 0)
    query_total = int(payload.get("query_embeddings_total") or 0)
    total = doc_total + query_total
    ok = doc_ok + query_ok
    pct = f"{(100.0 * ok / total):.0f}%" if total > 0 else "-"
    cost_text = "-"
    if payload.get("estimated_cost_usd") is not None:
        try:
            cost_text = f"${float(payload['estimated_cost_usd']):.4f}"
        except Exception:
            cost_text = "-"
    return pct, cost_text


def _embedding_status_pair(cfg: AppConfig) -> tuple[str, str]:
    return _embedding_status(cfg, cfg.dataset.prepared_dir)


def build_prepare_variants_from_config_paths(paths: list[Path]) -> list[PrepareVariant]:
    out: list[PrepareVariant] = []
    for path in paths:
        try:
            targets = load_engine_targets(path)
        except Exception:
            continue
        for t in targets:
            embed_status, cost_status = _embedding_status_pair(t.cfg)
            out.append(
                PrepareVariant(
                    config_path=str(path),
                    entry=t.name,
                    source=t.cfg.dataset.source,
                    prepared_dir=t.cfg.dataset.prepared_dir,
                    cfg=t.cfg,
                    embeddings_status=embed_status,
                    cost_status=cost_status,
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
        embed_status, cost_status = _embedding_status_pair(v.cfg)
        variants.append(
            PrepareVariant(
                config_path=v.config_path,
                entry=entry_label,
                source=v.source,
                prepared_dir=v.prepared_dir,
                cfg=v.cfg,
                embeddings_status=embed_status,
                cost_status=cost_status,
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
    row_height: int,
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    columns = [
        ("[x]" if selected else "[ ]", widths["sel"], attrs),
        (v.source, widths["source"], attrs | curses.color_pair(1) if curses.has_colors() else attrs),
        (v.prepared_dir, widths["prepared"], attrs),
        (v.embeddings_status, widths["embeddings"], attrs),
        (v.cost_status, widths["cost"], attrs),
        (v.entry, widths["entry"], attrs),
    ]
    x = 0
    for text, width, cell_attrs in columns:
        lines = [ln.ljust(width)[:width] for ln in wrap_text_lines(text, width)]
        while len(lines) < row_height:
            lines.append("".ljust(width))
        for i in range(row_height):
            stdscr.addstr(y + i, x, lines[i], cell_attrs)
        x += width + 1


def _row_height(v: PrepareVariant, widths: dict[str, int]) -> int:
    values = [
        "[x]",
        v.source,
        v.prepared_dir,
        v.embeddings_status,
        v.cost_status,
        v.entry,
    ]
    width_list = [
        widths["sel"],
        widths["source"],
        widths["prepared"],
        widths["embeddings"],
        widths["cost"],
        widths["entry"],
    ]
    return max(1, max(len(wrap_text_lines(text, width)) for text, width in zip(values, width_list, strict=True)))


def _ensure_current_visible(current: int, offset: int, variants: list[PrepareVariant], widths: dict[str, int], avail_lines: int) -> int:
    if not variants:
        return 0
    heights = [_row_height(v, widths) for v in variants]
    if current < offset:
        offset = current
    while True:
        used = 0
        visible_end = offset
        while visible_end < len(variants) and used + heights[visible_end] <= avail_lines:
            used += heights[visible_end]
            visible_end += 1
        if current < visible_end:
            return offset
        offset += 1
        if offset >= len(variants):
            return max(0, len(variants) - 1)


def _select_prepare_variants_curses(variants: list[PrepareVariant]) -> PrepareSelection:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    all_variants = list(variants)

    def _inner(stdscr: Any) -> tuple[str, list[int]]:
        nonlocal current, offset, variants, selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)  # source
        sort_cols = ["source", "prepared", "embeddings", "cost", "entry"]
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
            if col == "embeddings":
                return v.embeddings_status
            if col == "cost":
                return v.cost_status
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
                fit_text("↑/↓ move  ←/→ column  SPACE toggle  f filter  ENTER prepare  r reset embeddings  ESC/q cancel", max(1, w - 1)),
                curses.A_BOLD,
            )

            widths = {
                "sel": 3,
                "source": max(20, int(w * 0.36)),
                "prepared": max(24, int(w * 0.28)),
                "embeddings": 10,
                "cost": 10,
                "entry": max(18, w - 3 - 1 - max(20, int(w * 0.36)) - 1 - max(24, int(w * 0.28)) - 1 - 10 - 1 - 10 - 1 - 1),
            }
            filter_lines = filter_block_height(filters, widths)
            table_w = sum(widths.values()) + 3
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[("SEL", "sel"), ("SOURCE", "source"), ("PREPARED DIR", "prepared"), ("EMBED", "embeddings"), ("COST", "cost"), ("PROFILES", "entry")],
                widths=widths,
                max_width=table_w,
                active_col=sort_key,
                sort_desc=None,
                filters=filters,
                filter_lines=filter_lines,
            )
            rows_y = 2 + filter_lines
            avail_lines = max(1, h - rows_y - 2)
            offset = _ensure_current_visible(current, offset, variants, widths, avail_lines)

            y = rows_y
            ridx = offset
            while ridx < len(variants) and y < h - 2:
                row_h = _row_height(variants[ridx], widths)
                if y + row_h > h - 1:
                    break
                _draw_row(
                    stdscr,
                    y=y,
                    v=variants[ridx],
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    row_height=row_h,
                )
                y += row_h
                ridx += 1

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
                return ("prepare", sorted(selected_idx))
            elif ch in (ord("r"), ord("R")):
                return ("reset", sorted(selected_idx))
            elif ch == 27 or ch in (ord("q"),):
                return ("cancel", [])

    action, idxs = curses.wrapper(_inner)
    return PrepareSelection(action=action, variants=[variants[i] for i in idxs])


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


def select_prepare_variants_interactive(variants: list[PrepareVariant]) -> PrepareSelection:
    if not variants:
        return PrepareSelection(action="cancel", variants=[])
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_prepare_variants_curses(variants)
        except Exception:
            pass
    print("Select dataset sources to prepare/reset (comma or ranges, e.g. 1,3-5):")
    for i, v in enumerate(variants, start=1):
        print(f"{i}. source={v.source}  prepared_dir={v.prepared_dir}  embed={v.embeddings_status}  cost={v.cost_status}  profiles={v.entry}")
    raw = input("Selection (empty to cancel): ").strip()
    if not raw:
        return PrepareSelection(action="cancel", variants=[])
    idxs = _parse_selection(raw, len(variants))
    variants_out = [variants[i - 1] for i in idxs]
    action_raw = input("Action: [p]repare or [r]eset embeddings? ").strip().lower()
    action = "reset" if action_raw == "r" else "prepare"
    return PrepareSelection(action=action, variants=variants_out)
