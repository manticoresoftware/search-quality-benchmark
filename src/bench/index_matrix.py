from __future__ import annotations

import curses
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from .config import AppConfig, load_engine_targets
from .discovery import discover_config_paths
from .manticore import ManticoreClient
from .meilisearch_backend import MeilisearchBackend
from .qdrant_backend import QdrantBackend
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
    base_entry: str = ""


@dataclass(frozen=True)
class TargetStatus:
    state: str
    actual_count: int | None
    expected_count: int | None
    error: str | None = None


def _display_mode(mode: str) -> str:
    if mode == "fulltext":
        return "full-text"
    if mode == "vector":
        return "vector"
    if mode == "hybrid":
        return "hybrid"
    if mode == "both":
        return "both"
    return mode


def _model_opts_label(cfg: AppConfig, engine: str, mode: str) -> str:
    if mode == "vector":
        if engine == "meilisearch":
            extra = [f"embedder:{cfg.meilisearch.embedder_name}"]
            if bool(cfg.meilisearch.binary_quantized):
                extra.append("q=1bit")
            return f"{cfg.manticore.vector_model} [{' '.join(extra)}]"
        return cfg.manticore.vector_model
    if engine == "manticore":
        return (cfg.manticore.fulltext_options or "").strip() or "(none)"
    if engine == "meilisearch":
        extras: list[str] = []
        if cfg.meilisearch.searchable_attributes:
            extras.append(f"searchable:{','.join(cfg.meilisearch.searchable_attributes)}")
        if cfg.meilisearch.ranking_rules:
            extras.append(f"rules:{','.join(cfg.meilisearch.ranking_rules)}")
        if not bool(cfg.meilisearch.typo_tolerance):
            extras.append("typo:off")
        return " ".join(extras) or "(meilisearch lexical)"
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
                elif engine == "meilisearch":
                    base = str(cfg.meilisearch.index_name or "").strip()
                    target = f"{base}_lexical" if mode == "fulltext" else f"{base}_{cfg.meilisearch.vector_index}"
                else:
                    target = cfg.qdrant.bm25_collection if mode == "fulltext" else cfg.qdrant.dense_collection
                out.append(
                    IndexVariant(
                        config_path=str(path),
                        entry=target_cfg.name,
                        base_entry=target_cfg.base_name,
                        engine=engine,
                        mode=mode,
                        model=_model_opts_label(cfg, engine=engine, mode=mode),
                        batch_size=(
                            cfg.manticore.batch_size
                            if engine == "manticore"
                            else (cfg.meilisearch.batch_size if engine == "meilisearch" else cfg.qdrant.batch_size)
                        ),
                        target=target,
                        cfg=cfg,
                    )
                )

    dedup: dict[tuple[str, str, str, int, str], IndexVariant] = {}
    for v in out:
        key = (v.engine, v.mode, v.model, v.batch_size, v.target)
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = v
            continue
        if prev.entry == v.base_entry:
            dedup[key] = v
            continue
        if v.entry == prev.base_entry:
            continue

    engine_order = {"manticore": 0, "qdrant": 1, "meilisearch": 2}
    mode_order = {"fulltext": 0, "vector": 1}
    return sorted(
        list(dedup.values()),
        key=lambda v: (engine_order.get(v.engine, 99), mode_order.get(v.mode, 99), v.entry, v.model, v.target),
    )


def _expected_doc_count(v: IndexVariant) -> int | None:
    manifest_path = Path(v.cfg.dataset.prepared_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    count = payload.get("doc_count")
    try:
        return int(count) if count is not None else None
    except Exception:
        return None


def _resolve_target_status(v: IndexVariant) -> TargetStatus:
    expected = _expected_doc_count(v)
    try:
        if v.engine == "manticore":
            client = ManticoreClient(v.cfg.manticore)
            try:
                if not client.table_exists(v.target):
                    return TargetStatus("pending", None, expected)
                actual = client.count_rows(v.target)
            finally:
                client.close()
        elif v.engine == "qdrant":
            backend = QdrantBackend(v.cfg.qdrant)
            collection_exists = getattr(backend.client, "collection_exists", None)
            if callable(collection_exists) and not bool(collection_exists(v.target)):
                return TargetStatus("pending", None, expected)
            actual = int(backend._count_points(v.target))
        else:
            backend = MeilisearchBackend(v.cfg.meilisearch)
            backend.ensure_accessible()
            if not backend.index_exists(v.target):
                return TargetStatus("pending", None, expected)
            actual = int(backend.get_document_count(v.target))
    except Exception as exc:
        return TargetStatus("pending", None, expected, str(exc))
    if expected is not None and actual != expected:
        return TargetStatus("unexpected", actual, expected)
    return TargetStatus("ok", actual, expected)


def _draw_row(
    stdscr: Any,
    y: int,
    v: IndexVariant,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    row_lines: int,
    target_status: TargetStatus,
) -> None:
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    color_engine = 0
    color_mode = 0
    color_target = 0
    if curses.has_colors():
        if v.engine == "manticore":
            color_engine = curses.color_pair(1)
        elif v.engine == "qdrant":
            color_engine = curses.color_pair(2)
        elif v.engine == "meilisearch":
            color_engine = curses.color_pair(5)
        if v.mode == "fulltext":
            color_mode = curses.color_pair(3)
        elif v.mode == "vector":
            color_mode = curses.color_pair(4)
        elif v.mode == "both":
            color_mode = curses.color_pair(5)
        if target_status.state == "ok":
            color_target = curses.color_pair(6)
        elif target_status.state == "pending":
            color_target = curses.color_pair(7)
        elif target_status.state == "unexpected":
            color_target = curses.color_pair(8)

    columns: list[tuple[str, int, int]] = [
        ("[x]" if selected else "[ ]", 0, widths["sel"]),
        (v.entry, attrs, widths["entry"]),
        (v.engine, attrs | color_engine, widths["engine"]),
        (_display_mode(v.mode), attrs | color_mode, widths["mode"]),
        (v.target, attrs | color_target, widths["target"]),
        (v.model, attrs, widths["model"]),
        (str(v.batch_size), attrs, widths["batch"]),
    ]
    x = 0
    for text, cell_attrs, width in columns:
        cell_lines = wrap_cell(text, width, max_lines=row_lines)
        for i in range(row_lines):
            stdscr.addstr(y + i, x, cell_lines[i], cell_attrs)
        x += width + 1


def _mask_detail_value(path: str, value: Any) -> Any:
    p = path.lower()
    if any(k in p for k in ("password", "api_key")) and str(value or "").strip():
        return "***"
    return value


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _flatten_detail_values(prefix: str, obj: Any, out: dict[str, str]) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_detail_values(key, obj[k], out)
        return
    out[prefix] = _format_detail_value(_mask_detail_value(prefix, obj))


def _variant_detail_values(v: IndexVariant) -> dict[str, str]:
    visible_sections = {"dataset", "benchmark", "metrics", v.engine}
    cfg_obj = asdict(v.cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    flattened: dict[str, str] = {}
    _flatten_detail_values("", filtered_cfg, flattened)
    return flattened


def _comparison_variants(variants: list[IndexVariant], current: IndexVariant) -> list[IndexVariant]:
    same_engine = [v for v in variants if v.engine == current.engine]
    return same_engine or [current]


def _common_detail_keys(variants: list[IndexVariant], current: IndexVariant) -> set[str]:
    variants = _comparison_variants(variants, current)
    if len(variants) <= 1:
        return set()
    flattened = [_variant_detail_values(v) for v in variants]
    common_keys = set(flattened[0].keys())
    for values in flattened[1:]:
        common_keys &= set(values.keys())
    out: set[str] = set()
    for key in common_keys:
        first = flattened[0][key]
        if all(values.get(key) == first for values in flattened[1:]):
            out.add(key)
    return out


def _details_lines(
    v: IndexVariant,
    *,
    hidden_common_keys: set[str] | None = None,
    comparison_scope: str | None = None,
    target_status: TargetStatus | None = None,
) -> list[str]:
    hidden_common_keys = hidden_common_keys or set()

    lines: list[str] = [
        "Resolved Config",
        f"profile: {v.entry}",
        f"engine: {v.engine}",
        f"mode: {_display_mode(v.mode)}",
        f"target: {v.target}",
    ]
    if target_status is not None:
        state_label = {
            "ok": "indexed",
            "pending": "pending indexing",
            "unexpected": "row count mismatch",
        }.get(target_status.state, target_status.state)
        if target_status.actual_count is not None and target_status.expected_count is not None:
            lines.append(f"target status: {state_label} ({target_status.actual_count:,}/{target_status.expected_count:,})")
        elif target_status.actual_count is not None:
            lines.append(f"target status: {state_label} ({target_status.actual_count:,})")
        elif target_status.expected_count is not None:
            lines.append(f"target status: {state_label} (expected {target_status.expected_count:,})")
        else:
            lines.append(f"target status: {state_label}")
        if target_status.error:
            lines.append(f"target error: {target_status.error}")
    if hidden_common_keys:
        scope = comparison_scope or "visible rows"
        lines.append(f"common across {scope}: {len(hidden_common_keys)} hidden")
    flattened = _variant_detail_values(v)
    lines.extend(
        f"{key}: {value}"
        for key, value in flattened.items()
        if key not in hidden_common_keys
    )
    return lines


def _draw_details_panel(
    stdscr: Any,
    v: IndexVariant,
    variants: list[IndexVariant],
    widths: dict[str, int],
    w: int,
    h: int,
    target_status: TargetStatus,
) -> None:
    table_width = sum(widths.values()) + (len(widths) - 1)
    panel_x = table_width + 3
    panel_w = w - panel_x - 1
    if panel_w < 24:
        return
    comparison_variants = _comparison_variants(variants, v)
    lines = _details_lines(
        v,
        hidden_common_keys=_common_detail_keys(variants, v),
        comparison_scope=f"visible {v.engine} rows ({len(comparison_variants)})",
        target_status=target_status,
    )
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
            curses.init_pair(5, curses.COLOR_CYAN, -1)    # meilisearch
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)   # target ok
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_RED)     # target pending
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # target unexpected

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
        status_cache: dict[tuple[str, str, str, str, str, int], TargetStatus] = {}

        def _variant_id(v: IndexVariant) -> tuple[str, str, str, str, str, int]:
            return (v.entry, v.engine, v.mode, v.target, v.model, int(v.batch_size))

        def _get_status(v: IndexVariant) -> TargetStatus:
            key = _variant_id(v)
            if key not in status_cache:
                status_cache[key] = _resolve_target_status(v)
            return status_cache[key]

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
                    target_status=_get_status(variants[ridx]),
                )
            if variants:
                _draw_details_panel(
                    stdscr,
                    variants[current],
                    variants,
                    widths=widths,
                    w=w,
                    h=h,
                    target_status=_get_status(variants[current]),
                )
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
