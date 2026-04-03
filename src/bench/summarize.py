from __future__ import annotations

import curses
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from urllib.parse import quote

from .io_utils import read_json
from .reporting import metric_definitions_markdown
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
class ArtifactRow:
    run_id: str
    profile: str
    run_at_unix: int
    run_at: str
    path: Path
    engine: str
    mode: str
    model: str
    dataset_source: str
    doc_count: int | None
    runtime_seconds: float
    concurrency: int | None
    relevance: dict[str, float]
    performance: dict[str, float]
    status: str


_DATASET_URLS: dict[str, str] = {
    "beir/trec-covid": "https://ir-datasets.com/beir.html#beir/trec-covid",
    "beir/nfcorpus/test": "https://ir-datasets.com/beir.html#beir/nfcorpus/test",
    "msmarco-passage/dev/small": "https://ir-datasets.com/msmarco-passage.html#msmarco-passage/dev/small",
    "msmarco-passage/trec-dl-2019/judged": "https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged": "https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2020/judged",
    "hf-beir:theatticusproject/acord:test": "https://huggingface.co/datasets/theatticusproject/acord",
}


def _dataset_label(dataset: str) -> str:
    url = _DATASET_URLS.get(dataset)
    if url:
        return f"[`{dataset}`]({url})"
    return f"`{dataset}`"


def _dataset_group_label(dataset: str, doc_count: int | None) -> str:
    base = _dataset_label(dataset)
    if doc_count is None:
        return f"{base} (`docs: unknown`)"
    return f"{base} (`docs: {int(doc_count):,}`)"


def _system_label(engine: str, mode: str) -> str:
    if engine == "manticore" and mode == "fulltext":
        return "Manticore full-text (manticore)"
    if engine == "manticore" and mode == "vector":
        return "Manticore vector (manticore)"
    if engine == "manticore" and mode == "hybrid":
        return "Manticore hybrid (manticore)"
    if engine == "qdrant" and mode == "fulltext":
        return "Qdrant full-text (qdrant)"
    if engine == "qdrant" and mode == "vector":
        return "Qdrant vector (qdrant)"
    if engine == "qdrant" and mode == "hybrid":
        return "Qdrant hybrid (qdrant)"
    if engine == "meilisearch" and mode == "fulltext":
        return "Meilisearch full-text (meilisearch)"
    if engine == "meilisearch" and mode == "vector":
        return "Meilisearch vector (meilisearch)"
    if engine == "meilisearch" and mode == "hybrid":
        return "Meilisearch hybrid (meilisearch)"
    if engine == "opensearch" and mode == "fulltext":
        return "OpenSearch full-text (opensearch)"
    if engine == "opensearch" and mode == "vector":
        return "OpenSearch vector (opensearch)"
    if engine == "opensearch" and mode == "hybrid":
        return "OpenSearch hybrid (opensearch)"
    if engine == "elasticsearch" and mode == "fulltext":
        return "Elasticsearch full-text (elasticsearch)"
    if engine == "elasticsearch" and mode == "vector":
        return "Elasticsearch vector (elasticsearch)"
    if engine == "elasticsearch" and mode == "hybrid":
        return "Elasticsearch hybrid (elasticsearch)"
    if engine == "typesense" and mode == "fulltext":
        return "Typesense full-text (typesense)"
    if engine == "typesense" and mode == "vector":
        return "Typesense vector (typesense)"
    if engine == "typesense" and mode == "hybrid":
        return "Typesense hybrid (typesense)"
    return f"{engine} {mode}"


def _display_mode(mode: str) -> str:
    m = mode.lower()
    if m == "fulltext":
        return "full-text"
    if m == "vector":
        return "vector"
    if m == "hybrid":
        return "hybrid"
    return mode


def _run_at_from_run_id(run_id: str) -> str:
    try:
        ts = int(run_id)
        if ts > 10_000_000_000:  # nanoseconds run_id
            ts = ts // 1_000_000_000
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _run_at_from_unix(ts: int) -> str:
    try:
        if ts > 0:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "-"


def _extract_doc_count(payload: dict[str, Any]) -> int | None:
    live_state = payload.get("live_state_lines", []) or []
    if not isinstance(live_state, list):
        return None
    for line in live_state:
        text = str(line or "")
        m = re.search(r"rows \([^)]+\):\s*([0-9][0-9,]*)", text)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except Exception:
                return None
    return None


def discover_artifacts() -> list[ArtifactRow]:
    out: list[ArtifactRow] = []
    root = Path("artifacts")
    if not root.exists():
        return out
    for p in root.iterdir():
        if not p.is_dir():
            continue
        mpath = p / "metrics.json"
        if not mpath.exists():
            continue
        try:
            payload = read_json(mpath)
            run_id_val = str(payload.get("run_id", p.name))
            run_at_unix_val = int(payload.get("run_at_unix") or 0)
            if run_at_unix_val <= 0:
                try:
                    ts = int(run_id_val)
                    if ts > 10_000_000_000:
                        ts = ts // 1_000_000_000
                    run_at_unix_val = int(ts)
                except Exception:
                    run_at_unix_val = int(p.stat().st_mtime)
            rel = payload.get("relevance", {})
            perf = payload.get("performance", {})
            mode = str(payload.get("mode", ""))
            rel_mode = rel.get(mode, {}) if isinstance(rel, dict) else {}
            perf_mode = perf.get(mode, {}) if isinstance(perf, dict) else {}
            raw_model = payload.get("model")
            if raw_model is None or str(raw_model).strip() == "":
                if mode == "vector":
                    model = "unknown (legacy)"
                else:
                    model = "n/a"
            else:
                model = str(raw_model)
            out.append(
                ArtifactRow(
                    run_id=run_id_val,
                    profile=str(payload.get("profile", "") or ""),
                    run_at_unix=run_at_unix_val,
                    run_at=_run_at_from_unix(run_at_unix_val),
                    path=p,
                    engine=str(payload.get("engine", "")),
                    mode=mode,
                    model=model,
                    dataset_source=str(payload.get("dataset_source", "") or ""),
                    doc_count=_extract_doc_count(payload),
                    runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
                    concurrency=(
                        int((payload.get("benchmark_settings", {}) or {}).get("concurrency"))
                        if (payload.get("benchmark_settings", {}) or {}).get("concurrency") is not None
                        else None
                    ),
                    relevance={str(k): float(v) for k, v in rel_mode.items()},
                    performance={str(k): float(v) for k, v in perf_mode.items()},
                    status="ok",
                )
            )
        except Exception:
            out.append(
                ArtifactRow(
                    run_id=p.name,
                    profile="",
                    run_at_unix=int(p.stat().st_mtime),
                    run_at=_run_at_from_unix(int(p.stat().st_mtime)),
                    path=p,
                    engine="",
                    mode="",
                    model="n/a",
                    dataset_source="",
                    doc_count=None,
                    runtime_seconds=0.0,
                    concurrency=None,
                    relevance={},
                    performance={},
                    status="invalid",
                )
            )
    return sorted(out, key=lambda x: (x.run_at_unix, x.run_id), reverse=True)


def _parse_selection(raw: str, n: int) -> list[int]:
    picks: set[int] = set()
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
            for i in range(lo, hi + 1):
                if 1 <= i <= n:
                    picks.add(i)
        else:
            i = int(part)
            if 1 <= i <= n:
                picks.add(i)
    return sorted(picks)


def _color_pair_for_engine(engine: str) -> int:
    e = engine.lower()
    if e == "manticore":
        return 1
    if e == "qdrant":
        return 2
    if e == "meilisearch":
        return 5
    if e == "opensearch":
        return 9
    if e == "elasticsearch":
        return 12
    return 0


def _color_pair_for_mode(mode: str) -> int:
    m = mode.lower()
    if m == "fulltext":
        return 3
    if m == "vector":
        return 4
    if m == "hybrid":
        return 6
    return 0


def _draw_row(
    stdscr: Any,
    y: int,
    row: ArtifactRow,
    idx: int,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    row_height: int,
) -> None:
    prefix = "[x]" if selected else "[ ]"
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    e_pair = _color_pair_for_engine(row.engine)
    m_pair = _color_pair_for_mode(row.mode)
    columns: list[tuple[str, int, int]] = [
        (prefix, attrs, widths["sel"]),
        (str(idx), attrs, widths["idx"]),
        (row.run_id, attrs, widths["run_id"]),
        (row.run_at, attrs, widths["run_at"]),
        (row.dataset_source or "-", attrs, widths["dataset"]),
        (row.engine or "-", attrs | curses.color_pair(e_pair), widths["engine"]),
        (_display_mode(row.mode or "-"), attrs | curses.color_pair(m_pair), widths["mode"]),
        (f"{int(row.doc_count):,}" if row.doc_count is not None else "-", attrs, widths["docs"]),
        (row.model or "n/a", attrs, widths["model"]),
        (row.status, attrs, widths["status"]),
        (f"{row.relevance.get('ndcg@10', 0.0):.4f}", attrs, widths["ndcg"]),
        (f"{row.relevance.get('mrr@10', 0.0):.4f}", attrs, widths["mrr"]),
        (f"{row.relevance.get('map@10', 0.0):.4f}", attrs, widths["map"]),
        (f"{row.relevance.get('p@20', 0.0):.4f}", attrs, widths["p20"]),
        (f"{row.relevance.get('recall@100', 0.0):.4f}", attrs, widths["recall"]),
    ]
    x = 0
    for text, cell_attrs, width in columns:
        raw_lines = wrap_text_lines(text, width)
        cell_lines = [ln.ljust(width)[:width] for ln in raw_lines]
        while len(cell_lines) < row_height:
            cell_lines.append("".ljust(width))
        for i in range(row_height):
            stdscr.addstr(y + i, x, cell_lines[i], cell_attrs)
        x += width + 1


def _row_height(row: ArtifactRow, widths: dict[str, int]) -> int:
    values = [
        "[x]",
        row.run_id,
        row.run_at,
        row.dataset_source or "-",
        row.engine or "-",
        _display_mode(row.mode or "-"),
        f"{int(row.doc_count):,}" if row.doc_count is not None else "-",
        row.model or "n/a",
        row.status,
        f"{row.relevance.get('ndcg@10', 0.0):.4f}",
        f"{row.relevance.get('mrr@10', 0.0):.4f}",
        f"{row.relevance.get('map@10', 0.0):.4f}",
        f"{row.relevance.get('p@20', 0.0):.4f}",
        f"{row.relevance.get('recall@100', 0.0):.4f}",
    ]
    widths_list = [
        widths["sel"],
        widths["run_id"],
        widths["run_at"],
        widths["dataset"],
        widths["engine"],
        widths["mode"],
        widths["docs"],
        widths["model"],
        widths["status"],
        widths["ndcg"],
        widths["mrr"],
        widths["map"],
        widths["p20"],
        widths["recall"],
    ]
    return max(1, max(len(wrap_text_lines(text, width)) for text, width in zip(values, widths_list, strict=True)))


def _ensure_current_visible(current: int, offset: int, rows: list[ArtifactRow], widths: dict[str, int], avail_lines: int) -> int:
    if not rows:
        return 0
    heights = [_row_height(r, widths) for r in rows]
    # Move offset forward if current is above it.
    if current < offset:
        offset = current
    # Ensure current fits into the window.
    while True:
        used = 0
        visible_end = offset
        while visible_end < len(rows) and used + heights[visible_end] <= avail_lines:
            used += heights[visible_end]
            visible_end += 1
        if current < visible_end:
            return offset
        offset += 1
        if offset >= len(rows):
            return max(0, len(rows) - 1)


def _select_artifacts_curses(rows: list[ArtifactRow]) -> list[ArtifactRow]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    all_rows = list(rows)

    def _after_delete_reindex(selected: set[int], deleted_idx: int) -> set[int]:
        out: set[int] = set()
        for i in selected:
            if i == deleted_idx:
                continue
            out.add(i - 1 if i > deleted_idx else i)
        return out

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset, selected_idx, rows
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)     # manticore
            curses.init_pair(2, curses.COLOR_YELLOW, -1)    # qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)      # full-text
            curses.init_pair(4, curses.COLOR_RED, -1)       # vector
            curses.init_pair(5, curses.COLOR_CYAN, -1)      # meilisearch
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # hybrid
            curses.init_pair(9, curses.COLOR_MAGENTA, -1)   # opensearch
            curses.init_pair(12, curses.COLOR_BLUE, -1)     # elasticsearch

        widths = {
            "sel": 3,
            "idx": 3,
            "run_id": 10,
            "run_at": 19,
            "dataset": 24,
            "engine": 10,
            "mode": 9,
            "docs": 10,
            "model": 24,
            "status": 7,
            "ndcg": 8,
            "mrr": 8,
            "map": 8,
            "p20": 8,
            "recall": 10,
        }
        sort_by = "run_at"
        sort_desc = True
        sort_cols = ["run_at", "dataset", "engine", "mode", "docs", "model", "status", "ndcg", "mrr", "map", "p20", "recall", "run_id"]
        sort_idx = sort_cols.index(sort_by)
        filters: dict[str, str] = {}

        def _default_sort_desc(col: str) -> bool:
            return col in {"run_at", "ndcg", "mrr", "map", "p20", "recall"}

        def _row_sort_key(row: ArtifactRow, col: str) -> Any:
            if col == "run_at":
                return float(row.run_at_unix or 0)
            if col == "dataset":
                return (row.dataset_source or "").lower()
            if col == "engine":
                return (row.engine or "").lower()
            if col == "mode":
                return (row.mode or "").lower()
            if col == "docs":
                return int(row.doc_count or 0)
            if col == "model":
                return (row.model or "").lower()
            if col == "status":
                return (row.status or "").lower()
            if col == "ndcg":
                return float(row.relevance.get("ndcg@10", 0.0))
            if col == "mrr":
                return float(row.relevance.get("mrr@10", 0.0))
            if col == "map":
                return float(row.relevance.get("map@10", 0.0))
            if col == "p20":
                return float(row.relevance.get("p@20", 0.0))
            if col == "recall":
                return float(row.relevance.get("recall@100", 0.0))
            if col == "run_id":
                return row.run_id
            return row.run_id

        def _filter_value(row: ArtifactRow, col: str) -> Any:
            if col == "run_id":
                return row.run_id
            if col == "run_at":
                return row.run_at
            if col == "dataset":
                return row.dataset_source
            if col == "engine":
                return row.engine
            if col == "mode":
                return _display_mode(row.mode)
            if col == "docs":
                return f"{int(row.doc_count):,}" if row.doc_count is not None else "-"
            if col == "model":
                return row.model
            if col == "status":
                return row.status
            if col == "ndcg":
                return f"{row.relevance.get('ndcg@10', 0.0):.4f}"
            if col == "mrr":
                return f"{row.relevance.get('mrr@10', 0.0):.4f}"
            if col == "map":
                return f"{row.relevance.get('map@10', 0.0):.4f}"
            if col == "p20":
                return f"{row.relevance.get('p@20', 0.0):.4f}"
            if col == "recall":
                return f"{row.relevance.get('recall@100', 0.0):.4f}"
            return ""

        def _apply_sort(col: str, desc: bool, *, preserve_current_item: bool = True) -> None:
            nonlocal rows, selected_idx, current, offset
            if not rows:
                selected_ids: set[str] = set()
                current_id = None
            else:
                selected_ids = {rows[i].run_id for i in selected_idx if 0 <= i < len(rows)}
                current_id = rows[current].run_id if preserve_current_item and 0 <= current < len(rows) else None
            rows = filter_rows(all_rows, filters, _filter_value)
            rows.sort(key=lambda r: _row_sort_key(r, col), reverse=desc)
            selected_idx = {i for i, r in enumerate(rows) if r.run_id in selected_ids}
            if current_id is not None:
                for i, r in enumerate(rows):
                    if r.run_id == current_id:
                        current = i
                        break
            current = min(max(0, current), max(0, len(rows) - 1))
            offset = 0

        _apply_sort(sort_by, sort_desc)

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            help1 = "↑/↓ move  ←/→ sort col  SPACE toggle  f filter  a select all  d delete  ENTER submit  ESC/q cancel"
            stdscr.addstr(0, 0, fit_text(help1, max(1, w - 1)), curses.A_BOLD)
            filter_lines = filter_block_height(filters, widths)
            draw_header_with_filters(
                stdscr,
                y=1,
                cols=[
                    ("SEL", "sel"),
                    ("#", "idx"),
                    ("RUN_ID", "run_id"),
                    ("RUN_AT", "run_at"),
                    ("DATASET", "dataset"),
                    ("ENGINE", "engine"),
                    ("MODE", "mode"),
                    ("DOCS", "docs"),
                    ("MODEL/OPTS", "model"),
                    ("STATUS", "status"),
                    ("NDCG@10", "ndcg"),
                    ("MRR@10", "mrr"),
                    ("MAP@10", "map"),
                    ("P@20", "p20"),
                    ("RECALL@100", "recall"),
                ],
                widths=widths,
                max_width=max(1, w - 1),
                active_col=sort_by,
                sort_desc=sort_desc,
                filters=filters,
                filter_lines=filter_lines,
            )

            rows_y = 2 + filter_lines
            avail_lines = max(1, h - rows_y - 2)
            offset = _ensure_current_visible(current, offset, rows, widths, avail_lines)
            y = rows_y
            ridx = offset
            while ridx < len(rows) and y < h - 2:
                row_h = _row_height(rows[ridx], widths)
                if y + row_h > h - 1:
                    break
                _draw_row(
                    stdscr=stdscr,
                    y=y,
                    row=rows[ridx],
                    idx=ridx + 1,
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    row_height=row_h,
                )
                y += row_h
                ridx += 1

            sort_hint = f"{sort_by}{' desc' if sort_desc else ' asc'}"
            summary = f"Selected: {len(selected_idx)} / {len(rows)}  |  Sort: {sort_hint}"
            stdscr.addstr(h - 1, 0, fit_text(summary, max(1, w - 1)), curses.A_DIM)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                current = max(0, current - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                current = min(len(rows) - 1, current + 1)
            elif ch == curses.KEY_LEFT:
                sort_idx = (sort_idx - 1) % len(sort_cols)
                sort_by = sort_cols[sort_idx]
                sort_desc = _default_sort_desc(sort_by)
                _apply_sort(sort_by, sort_desc, preserve_current_item=False)
            elif ch == curses.KEY_RIGHT:
                sort_idx = (sort_idx + 1) % len(sort_cols)
                sort_by = sort_cols[sort_idx]
                sort_desc = _default_sort_desc(sort_by)
                _apply_sort(sort_by, sort_desc, preserve_current_item=False)
            elif ch == ord(" "):
                if current in selected_idx:
                    selected_idx.remove(current)
                else:
                    selected_idx.add(current)
            elif ch in (ord("a"), ord("A")):
                if len(selected_idx) == len(rows):
                    selected_idx.clear()
                else:
                    selected_idx = set(range(len(rows)))
            elif ch in (ord("f"), ord("F")):
                val = prompt_filter_value(
                    stdscr,
                    prompt=f"Filter {sort_by} (empty clears): ",
                    initial=str(filters.get(sort_by, "") or ""),
                )
                if val is not None:
                    val = val.strip()
                    if val:
                        filters[sort_by] = val
                    else:
                        filters.pop(sort_by, None)
                    _apply_sort(sort_by, sort_desc)
            elif ch in (ord("d"), ord("D")):
                if not rows:
                    continue
                row = rows[current]
                prompt = f"Delete artifact '{row.run_id}'? Type y to confirm."
                stdscr.addstr(h - 1, 0, _fit(prompt, max(1, w - 1)), curses.A_BOLD)
                stdscr.clrtoeol()
                stdscr.refresh()
                confirm = stdscr.getch()
                if confirm not in (ord("y"), ord("Y")):
                    continue
                try:
                    if row.path.exists():
                        shutil.rmtree(row.path)
                except Exception as exc:
                    msg = f"Delete failed: {exc}"
                    stdscr.addstr(h - 1, 0, _fit(msg, max(1, w - 1)), curses.A_BOLD)
                    stdscr.clrtoeol()
                    stdscr.refresh()
                    stdscr.getch()
                    continue

                del rows[current]
                selected_idx = _after_delete_reindex(selected_idx, current)
                if not rows:
                    return []
                current = min(current, len(rows) - 1)
                if current < offset:
                    offset = current
                offset = _ensure_current_visible(current, offset, rows, widths, avail_lines)
            elif ch in (10, 13, curses.KEY_ENTER):
                return sorted(selected_idx)
            elif ch == 27 or ch in (ord("q"),):
                return []

    picked = curses.wrapper(_inner)
    return [rows[i] for i in picked]


def select_artifacts_interactive(rows: list[ArtifactRow]) -> list[ArtifactRow]:
    if not rows:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_artifacts_curses(rows)
        except Exception:
            # Fall back to text mode on terminals that don't support curses well.
            pass
    print("Available artifacts:")
    print("")
    for i, r in enumerate(rows, start=1):
        ndcg_val = r.relevance.get("ndcg@10", 0.0)
        docs_text = f"{int(r.doc_count):,}" if r.doc_count is not None else "-"
        print(
            f"{i:>2}.\trun_id={r.run_id}\trun_at={r.run_at}\tdataset={r.dataset_source or '-'}\tengine={r.engine or '-'}\tmode={_display_mode(r.mode or '-')}\t"
            f"docs={docs_text}\tmodel={r.model or 'n/a'}\tstatus={r.status}\tndcg@10={ndcg_val:.4f}\t"
            f"runtime={r.runtime_seconds:.3f}s\tconcurrency={r.concurrency if r.concurrency is not None else '-'}"
        )
    print("")
    raw = input("Select artifact numbers (e.g. 1,3-5): ").strip()
    idxs = _parse_selection(raw, len(rows))
    return [rows[i - 1] for i in idxs]


def _collect_relevance_keys(rows: list[ArtifactRow]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        for k in r.relevance.keys():
            keys.add(k)
    ordered = sorted([k for k in keys if k.startswith("ndcg@")], key=lambda x: int(x.split("@", 1)[1]))
    if "map@10" in keys:
        ordered.append("map@10")
    if "mrr@10" in keys:
        ordered.append("mrr@10")
    if "p@20" in keys:
        ordered.append("p@20")
    if "recall@100" in keys:
        ordered.append("recall@100")
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    return ordered


def _relevance_sort_key(r: ArtifactRow) -> tuple[float, float, float, float, float, float, str, float]:
    ndcg10 = float(r.relevance.get("ndcg@10", 0.0))
    ndcg20 = float(r.relevance.get("ndcg@20", 0.0))
    map10 = float(r.relevance.get("map@10", 0.0))
    mrr10 = float(r.relevance.get("mrr@10", 0.0))
    p20 = float(r.relevance.get("p@20", 0.0))
    recall100 = float(r.relevance.get("recall@100", 0.0))
    system = _system_label(r.engine, r.mode).lower()
    run_at_unix = float(r.run_at_unix or 0.0)
    # Primary metric columns: descending (higher is better), then system asc, then run-at desc.
    return (-ndcg10, -ndcg20, -map10, -mrr10, -p20, -recall100, system, -run_at_unix)


def _performance_sort_key(r: ArtifactRow) -> tuple[float, float, float, float]:
    p = r.performance
    qps = float(p.get("qps", 0.0))
    p50 = float(p.get("latency_p50_ms", 0.0))
    p95 = float(p.get("latency_p95_ms", 0.0))
    p99 = float(p.get("latency_p99_ms", 0.0))
    # QPS descending, latencies ascending.
    return (-qps, p50, p95, p99)


def _profile_system_model_label(r: ArtifactRow) -> str:
    profile = f"**{r.profile}**" if (r.profile or "").strip() else "-"
    return " / ".join([profile, _system_label(r.engine, r.mode), r.model or "n/a"])


def _best_metric_bar_rows(rows: list[ArtifactRow], metric: str) -> list[tuple[str, float]]:
    best: dict[tuple[str, str, int | None], dict[str, float]] = {}
    for row in rows:
        if row.mode not in {"fulltext", "vector", "hybrid"}:
            continue
        if metric not in row.relevance:
            continue
        key = (row.engine or "-", row.mode, row.doc_count)
        ndcg_val = float(row.relevance.get("ndcg@10", 0.0))
        mrr_val = float(row.relevance.get("mrr@10", 0.0))
        prev = best.get(key)
        current_metric_val = float(row.relevance.get(metric, 0.0))
        if prev is None or current_metric_val > float(prev.get(metric, 0.0)):
            best[key] = {"ndcg@10": ndcg_val, "mrr@10": mrr_val}
    if not best:
        return []

    items = sorted(
        best.items(),
        key=lambda item: (-float(item[1].get(metric, 0.0)), (item[0][0] or "").lower(), item[0][1], item[0][2] or 0),
    )
    pairs: list[tuple[str, float]] = []
    show_docs = len({r.doc_count for r in rows}) > 1
    for (engine, mode, doc_count), metrics in items:
        label = f"{engine} / {_display_mode(mode)}"
        if show_docs:
            docs_text = str(int(doc_count)) if doc_count is not None else "unknown"
            label = f"{label} / {docs_text}"
        pairs.append((label, float(metrics.get(metric, 0.0))))
    return pairs


def _metric_table_lines(rows: list[ArtifactRow], metric: str) -> list[str]:
    pairs = _best_metric_bar_rows(rows, metric)
    if not pairs:
        return []
    width = 24
    lines = [f"Best {metric.upper()}:", ""]
    lines.append(f"| Category | {metric.upper()} |")
    lines.append("|---|---:|")
    for label, val in pairs:
        lines.append(f"| {label} | `{_bar_metric_text(val, width)}` |")
    return lines


def _canonical_metric_table_lines(rows: list[ArtifactRow]) -> list[str]:
    if not rows:
        return []
    dataset = str(rows[0].dataset_source or "")
    metric = _canonical_metric_for_dataset(dataset)
    pairs = _best_metric_bar_rows(rows, metric)
    if not pairs:
        return []
    width = 24
    lines = [f"Best canonical metric (`{metric.upper()}`):", ""]
    lines.append(f"| Category | {metric.upper()} |")
    lines.append("|---|---:|")
    for label, val in pairs:
        lines.append(f"| {label} | `{_bar_metric_text(val, width)}` |")
    return lines


def _bar_metric_text(val: float, width: int = 24) -> str:
    blocks = max(1, int(round(val * width))) if val > 0 else 0
    bar = "█" * blocks if blocks > 0 else "·"
    return f"{bar} {val:.4f}"


def _row_relevance_composite(row: ArtifactRow) -> float:
    ndcg10 = float(row.relevance.get("ndcg@10", 0.0))
    mrr10 = float(row.relevance.get("mrr@10", 0.0))
    map10 = float(row.relevance.get("map@10", 0.0))
    has_map10 = "map@10" in row.relevance
    if has_map10:
        return 0.6 * ndcg10 + 0.3 * mrr10 + 0.1 * map10
    return 0.7 * ndcg10 + 0.3 * mrr10


def _canonical_metric_for_dataset(dataset_source: str) -> str:
    ds = str(dataset_source or "").strip().lower()
    if ds.startswith("msmarco-passage/"):
        return "mrr@10"
    if ds.startswith("beir/"):
        return "ndcg@10"
    if ds.startswith("hf-beir:"):
        return "ndcg@10"
    return "ndcg@10"


def _row_canonical_metric_value(row: ArtifactRow) -> float:
    metric = _canonical_metric_for_dataset(row.dataset_source)
    return float(row.relevance.get(metric, 0.0))


def _eligible_for_global_relevance(row: ArtifactRow) -> bool:
    if row.status != "ok":
        return False
    if float(row.performance.get("error_rate", 0.0) or 0.0) > 0.0:
        return False
    return bool(row.engine and row.mode)


def _global_leaderboard_lines(rows: list[ArtifactRow]) -> list[str]:
    eligible = [row for row in rows if _eligible_for_global_relevance(row)]
    if not eligible:
        return []

    grouped: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in eligible:
        grouped.setdefault((row.dataset_source or "-", row.doc_count), []).append(row)

    per_variant_scores: dict[tuple[str, str], list[float]] = {}
    per_variant_ndcg: dict[tuple[str, str], list[float]] = {}
    per_variant_mrr: dict[tuple[str, str], list[float]] = {}

    for _, group_rows in grouped.items():
        best_by_variant: dict[tuple[str, str], ArtifactRow] = {}
        for row in group_rows:
            key = (row.engine, row.mode)
            prev = best_by_variant.get(key)
            if prev is None or _relevance_sort_key(row) < _relevance_sort_key(prev):
                best_by_variant[key] = row
        if not best_by_variant:
            continue
        scored_rows = [(variant, row, _row_relevance_composite(row)) for variant, row in best_by_variant.items()]
        values = [score for _, _, score in scored_rows]
        lo = min(values)
        hi = max(values)
        for variant, row, score in scored_rows:
            if hi > lo:
                norm = (score - lo) / (hi - lo)
            else:
                norm = 1.0
            per_variant_scores.setdefault(variant, []).append(norm)
            per_variant_ndcg.setdefault(variant, []).append(float(row.relevance.get("ndcg@10", 0.0)))
            per_variant_mrr.setdefault(variant, []).append(float(row.relevance.get("mrr@10", 0.0)))

    if not per_variant_scores:
        return []

    total_groups = len(grouped)
    leaderboard: list[tuple[str, str, float, int, float, float]] = []
    for (engine, mode), scores in per_variant_scores.items():
        groups_played = len(scores)
        coverage = groups_played / total_groups if total_groups > 0 else 0.0
        avg_score = sum(scores) / groups_played if groups_played else 0.0
        avg_ndcg = sum(per_variant_ndcg.get((engine, mode), [])) / max(1, len(per_variant_ndcg.get((engine, mode), [])))
        avg_mrr = sum(per_variant_mrr.get((engine, mode), [])) / max(1, len(per_variant_mrr.get((engine, mode), [])))
        final_score = avg_score * coverage
        leaderboard.append((engine, mode, final_score, groups_played, avg_ndcg, avg_mrr))

    leaderboard.sort(key=lambda item: (-item[2], -item[4], -item[5], item[0].lower(), item[1].lower()))

    lines = ["## Global Relevance Leaderboard", ""]
    lines.append(
        "Scoring: for each dataset/doc-count group, take the best run per engine, "
        "compute a relevance composite (`NDCG@10`, `MRR@10`, optional `MAP@10`), "
        "normalize it within the group, then average globally. Coverage across groups is included in the final score."
    )
    lines.append("")
    lines.append("| Rank | Engine | Mode | Groups | Coverage | Avg NDCG@10 | Avg MRR@10 | Score |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for idx, (engine, mode, score, groups_played, avg_ndcg, avg_mrr) in enumerate(leaderboard, start=1):
        coverage = groups_played / total_groups if total_groups > 0 else 0.0
        lines.append(
            f"| {idx} | {engine} | {_display_mode(mode)} | {groups_played}/{total_groups} | {coverage:.0%} | "
            f"{avg_ndcg:.4f} | {avg_mrr:.4f} | {score:.4f} |"
        )
    lines.append("")
    return lines


def _global_efficiency_leaderboard_lines(rows: list[ArtifactRow]) -> list[str]:
    eligible = [row for row in rows if _eligible_for_global_relevance(row)]
    if not eligible:
        return []

    grouped: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in eligible:
        grouped.setdefault((row.dataset_source or "-", row.doc_count), []).append(row)

    per_engine_scores: dict[str, list[float]] = {}
    per_engine_ndcg: dict[str, list[float]] = {}
    per_engine_mrr: dict[str, list[float]] = {}
    per_engine_qps: dict[str, list[float]] = {}
    per_engine_best_modes: dict[str, dict[str, int]] = {}

    for _, group_rows in grouped.items():
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None:
                best_by_engine[row.engine] = row
                continue
            prev_eff = _row_relevance_composite(prev) * math.log1p(float(prev.performance.get("qps", 0.0)))
            cur_eff = _row_relevance_composite(row) * math.log1p(float(row.performance.get("qps", 0.0)))
            if cur_eff > prev_eff or (cur_eff == prev_eff and _relevance_sort_key(row) < _relevance_sort_key(prev)):
                best_by_engine[row.engine] = row
        if not best_by_engine:
            continue
        scored_rows = []
        for engine, row in best_by_engine.items():
            rel = _row_relevance_composite(row)
            qps = float(row.performance.get("qps", 0.0))
            eff = rel * math.log1p(qps)
            scored_rows.append((engine, row, eff))
        values = [score for _, _, score in scored_rows]
        lo = min(values)
        hi = max(values)
        for engine, row, score in scored_rows:
            if hi > lo:
                norm = (score - lo) / (hi - lo)
            else:
                norm = 1.0
            per_engine_scores.setdefault(engine, []).append(norm)
            per_engine_ndcg.setdefault(engine, []).append(float(row.relevance.get("ndcg@10", 0.0)))
            per_engine_mrr.setdefault(engine, []).append(float(row.relevance.get("mrr@10", 0.0)))
            per_engine_qps.setdefault(engine, []).append(float(row.performance.get("qps", 0.0)))
            mode_counts = per_engine_best_modes.setdefault(engine, {})
            mode_counts[row.mode] = mode_counts.get(row.mode, 0) + 1

    if not per_engine_scores:
        return []

    total_groups = len(grouped)
    leaderboard: list[tuple[str, float, int, float, float, float, str]] = []
    for engine, scores in per_engine_scores.items():
        groups_played = len(scores)
        coverage = groups_played / total_groups if total_groups > 0 else 0.0
        avg_score = sum(scores) / groups_played if groups_played else 0.0
        avg_ndcg = sum(per_engine_ndcg.get(engine, [])) / max(1, len(per_engine_ndcg.get(engine, [])))
        avg_mrr = sum(per_engine_mrr.get(engine, [])) / max(1, len(per_engine_mrr.get(engine, [])))
        avg_qps = sum(per_engine_qps.get(engine, [])) / max(1, len(per_engine_qps.get(engine, [])))
        mode_counts = per_engine_best_modes.get(engine, {})
        best_mode = "-"
        if mode_counts:
            best_mode = sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        final_score = avg_score * coverage
        leaderboard.append((engine, final_score, groups_played, avg_ndcg, avg_mrr, avg_qps, best_mode))

    leaderboard.sort(key=lambda item: (-item[1], -item[5], -item[3], -item[4], item[0].lower()))

    lines = ["## Global Efficiency Leaderboard", ""]
    lines.append(
        "Scoring: for each dataset/doc-count group, take the best run per engine by "
        "`relevance * log(1 + qps)`, normalize it within the group, then average globally. "
        "Coverage across groups is included in the final score."
    )
    lines.append("")
    lines.append("| Rank | Engine | Best mode | Groups | Coverage | Avg NDCG@10 | Avg MRR@10 | Avg QPS | Score |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for idx, (engine, score, groups_played, avg_ndcg, avg_mrr, avg_qps, best_mode) in enumerate(leaderboard, start=1):
        coverage = groups_played / total_groups if total_groups > 0 else 0.0
        lines.append(
            f"| {idx} | {engine} | {_display_mode(best_mode)} | {groups_played}/{total_groups} | {coverage:.0%} | "
            f"{avg_ndcg:.4f} | {avg_mrr:.4f} | {avg_qps:.2f} | {score:.4f} |"
        )
    lines.append("")
    return lines


def _row_map10_value(row: ArtifactRow) -> float:
    return float(row.relevance.get("map@10", 0.0))


def _global_variant_leaderboard_lines(rows: list[ArtifactRow]) -> list[str]:
    eligible = [row for row in rows if _eligible_for_global_relevance(row)]
    if not eligible:
        return []

    grouped: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in eligible:
        grouped.setdefault((row.dataset_source or "-", row.doc_count), []).append(row)
    total_groups = len(grouped)
    if total_groups <= 0:
        return []

    common_notes = (
        "For every variant: runs are first split by `(dataset_source, doc_count)`. "
        "Inside each group, keep the best run per engine according to the variant score. "
        "Then average the raw per-group scores globally across full-coverage entries. "
        "Runs with `status != ok` or `error_rate > 0` are excluded."
    )

    variants: list[tuple[str, str, Any, Any, bool]] = [
        (
            "Canonical metric",
            "Per-group raw score formula: `score = canonical_metric(dataset_source)`, where "
            "`canonical_metric = MRR@10` for `msmarco-passage/*`, and `NDCG@10` for `beir/*` and `hf-beir:*`. "
            "This follows the common practice of using the dataset family's primary metric instead of a universal weighted composite. "
            + common_notes,
            lambda row: True,
            lambda row: (
                _row_canonical_metric_value(row),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            True,
        ),
        (
            "NDCG-first",
            "Per-group engine score tuple: `(NDCG@10, MRR@10, MAP@10)`. "
            "Ranking is lexicographic on that tuple, so the primary objective is maximizing `NDCG@10`, "
            "with `MRR@10` and `MAP@10` only used as tie-breakers. "
            + common_notes,
            lambda row: True,
            lambda row: (float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0)), float(row.relevance.get("map@10", 0.0))),
            False,
        ),
        (
            "MRR-first",
            "Per-group engine score tuple: `(MRR@10, NDCG@10, MAP@10)`. "
            "Ranking is lexicographic on that tuple, so the primary objective is maximizing `MRR@10`, "
            "with `NDCG@10` and `MAP@10` only used as tie-breakers. "
            + common_notes,
            lambda row: True,
            lambda row: (float(row.relevance.get("mrr@10", 0.0)), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("map@10", 0.0))),
            False,
        ),
        (
            "NDCG only",
            "Per-group raw score formula: `score = NDCG@10`. "
            "Tie-break display columns still show average `MRR@10` and `QPS`, but they do not affect ranking. "
            + common_notes,
            lambda row: True,
            lambda row: (float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0)), float(row.relevance.get("map@10", 0.0))),
            False,
        ),
        (
            "MRR only",
            "Per-group raw score formula: `score = MRR@10`. "
            "Tie-break display columns still show average `NDCG@10` and `QPS`, but they do not affect ranking. "
            + common_notes,
            lambda row: True,
            lambda row: (float(row.relevance.get("mrr@10", 0.0)), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("map@10", 0.0))),
            False,
        ),
        (
            "Balanced Relevance",
            "Per-group raw score formula: "
            "`score = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10`. "
            "If `MAP@10` is missing, fallback formula is `score = 0.7*NDCG@10 + 0.3*MRR@10`. "
            + common_notes,
            lambda row: True,
            lambda row: (_row_relevance_composite(row), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0))),
            False,
        ),
        (
            "Efficiency logQPS",
            "Per-group raw score formula: "
            "`relevance = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing), then "
            "`score = relevance * log(1 + QPS)`. "
            "This rewards throughput sublinearly so QPS helps but does not dominate too aggressively. "
            + common_notes,
            lambda row: True,
            lambda row: (
                _row_relevance_composite(row) * math.log1p(float(row.performance.get("qps", 0.0))),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            False,
        ),
        (
            "Efficiency sqrtQPS",
            "Per-group raw score formula: "
            "`relevance = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing), then "
            "`score = relevance * sqrt(max(QPS, 0))`. "
            "This rewards throughput more aggressively than `log(1 + QPS)`. "
            + common_notes,
            lambda row: True,
            lambda row: (
                _row_relevance_composite(row) * math.sqrt(max(0.0, float(row.performance.get("qps", 0.0)))),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            False,
        ),
        (
            "Full-text only",
            "Filter: keep only `mode = fulltext`. "
            "Per-group raw score formula: "
            "`score = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing). "
            + common_notes,
            lambda row: row.mode == "fulltext",
            lambda row: (_row_relevance_composite(row), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0))),
            False,
        ),
        (
            "Vector only",
            "Filter: keep only `mode = vector`. "
            "Per-group raw score formula: "
            "`score = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing). "
            + common_notes,
            lambda row: row.mode == "vector",
            lambda row: (_row_relevance_composite(row), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0))),
            False,
        ),
        (
            "Hybrid only",
            "Filter: keep only `mode = hybrid`. "
            "Per-group raw score formula: "
            "`score = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing). "
            + common_notes,
            lambda row: row.mode == "hybrid",
            lambda row: (_row_relevance_composite(row), float(row.relevance.get("ndcg@10", 0.0)), float(row.relevance.get("mrr@10", 0.0))),
            False,
        ),
        (
            "Full-text efficiency",
            "Filter: keep only `mode = fulltext`. "
            "Per-group raw score formula: "
            "`relevance = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing), then "
            "`score = relevance * log(1 + QPS)`. "
            + common_notes,
            lambda row: row.mode == "fulltext",
            lambda row: (
                _row_relevance_composite(row) * math.log1p(float(row.performance.get("qps", 0.0))),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            False,
        ),
        (
            "Vector efficiency",
            "Filter: keep only `mode = vector`. "
            "Per-group raw score formula: "
            "`relevance = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing), then "
            "`score = relevance * log(1 + QPS)`. "
            + common_notes,
            lambda row: row.mode == "vector",
            lambda row: (
                _row_relevance_composite(row) * math.log1p(float(row.performance.get("qps", 0.0))),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            False,
        ),
        (
            "Hybrid efficiency",
            "Filter: keep only `mode = hybrid`. "
            "Per-group raw score formula: "
            "`relevance = 0.6*NDCG@10 + 0.3*MRR@10 + 0.1*MAP@10` "
            "(or `0.7*NDCG@10 + 0.3*MRR@10` when `MAP@10` is missing), then "
            "`score = relevance * log(1 + QPS)`. "
            + common_notes,
            lambda row: row.mode == "hybrid",
            lambda row: (
                _row_relevance_composite(row) * math.log1p(float(row.performance.get("qps", 0.0))),
                float(row.relevance.get("ndcg@10", 0.0)),
                float(row.relevance.get("mrr@10", 0.0)),
            ),
            False,
        ),
    ]

    lines: list[str] = []

    for variant_name, description, include_fn, score_fn, canonical_only in variants:
        if variant_name != "Canonical metric":
            continue
        per_variant_scores: dict[tuple[str, str], list[float]] = {}
        per_variant_ndcg: dict[tuple[str, str], list[float]] = {}
        per_variant_mrr: dict[tuple[str, str], list[float]] = {}
        for _, group_rows in grouped.items():
            best_by_variant: dict[tuple[str, str], tuple[ArtifactRow, tuple[float, float, float]]] = {}
            filtered_rows = [row for row in group_rows if include_fn(row)]
            for row in filtered_rows:
                score_tuple = score_fn(row)
                key = (row.engine, row.mode)
                prev = best_by_variant.get(key)
                if prev is None or score_tuple > prev[1]:
                    best_by_variant[key] = (row, score_tuple)
            if not best_by_variant:
                continue
            for variant, (row, score_tuple) in best_by_variant.items():
                raw_score = float(score_tuple[0])
                per_variant_scores.setdefault(variant, []).append(raw_score)
                per_variant_ndcg.setdefault(variant, []).append(float(row.relevance.get("ndcg@10", 0.0)))
                per_variant_mrr.setdefault(variant, []).append(float(row.relevance.get("mrr@10", 0.0)))

        leaderboard: list[tuple[str, str, float, int, float, float]] = []
        excluded_variants: list[tuple[str, str, int]] = []
        for (engine, mode), scores in per_variant_scores.items():
            groups_played = len(scores)
            if groups_played < total_groups:
                excluded_variants.append((engine, mode, groups_played))
                continue
            avg_score = sum(scores) / groups_played if groups_played else 0.0
            avg_ndcg = sum(per_variant_ndcg.get((engine, mode), [])) / max(1, len(per_variant_ndcg.get((engine, mode), [])))
            avg_mrr = sum(per_variant_mrr.get((engine, mode), [])) / max(1, len(per_variant_mrr.get((engine, mode), [])))
            leaderboard.append((engine, mode, avg_score, groups_played, avg_ndcg, avg_mrr))

        leaderboard.sort(key=lambda item: (-item[2], -item[4], -item[5], item[0].lower(), item[1].lower()))
        lines.append("## Engine + Mode Leaderboard")
        lines.append("")
        lines.append(
            "This leaderboard uses the canonical metric for each dataset family. "
            "The per-group raw score is `MRR@10` for `msmarco-passage/*`, and `NDCG@10` for `beir/*` and `hf-beir:*`. "
            "For each `(dataset_source, doc_count)` group, the table keeps one best run per engine+mode and averages those raw canonical scores globally. "
            "Only full-coverage entries are included, so engines or modes missing any group are omitted from this table. "
            "Runs with `status != ok` or `error_rate > 0` are excluded."
        )
        lines.append("")
        lines.append("Sorted by `Score` descending. Here `Score` is just the average canonical metric.")
        lines.append("")
        if not leaderboard:
            lines.append("_No eligible runs for this variant._")
            lines.append("")
            continue
        if canonical_only:
            lines.append("| Rank | Engine | Mode | Score ↓ |")
            lines.append("|---:|---|---|---:|")
            for idx, (engine, mode, score, groups_played, avg_ndcg, avg_mrr) in enumerate(leaderboard, start=1):
                lines.append(
                    f"| {idx} | {engine} | {_display_mode(mode)} | `{_bar_metric_text(score)}` |"
                )
            lines.append("")
            if excluded_variants:
                excluded_variants.sort(key=lambda item: (item[0].lower(), item[1].lower()))
                excluded_text = ", ".join(
                    f"`{engine} / {_display_mode(mode)}` ({groups_played}/{total_groups})"
                    for engine, mode, groups_played in excluded_variants
                )
                lines.append(
                    f"Excluded for incomplete coverage: {excluded_text}."
                )
                lines.append("")
        lines.append("")
    return lines


def _engine_only_leaderboard_lines(rows: list[ArtifactRow]) -> list[str]:
    eligible = [row for row in rows if _eligible_for_global_relevance(row)]
    if not eligible:
        return []

    grouped: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in eligible:
        grouped.setdefault((row.dataset_source or "-", row.doc_count), []).append(row)
    total_groups = len(grouped)
    if total_groups <= 0:
        return []

    per_engine_scores: dict[str, list[float]] = {}
    per_engine_best_modes: dict[str, dict[str, int]] = {}
    for _, group_rows in grouped.items():
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_engine[row.engine] = row
        if not best_by_engine:
            continue
        scored_rows = [(engine, row, _row_canonical_metric_value(row)) for engine, row in best_by_engine.items()]
        for engine, row, score in scored_rows:
            per_engine_scores.setdefault(engine, []).append(float(score))
            mode_counts = per_engine_best_modes.setdefault(engine, {})
            mode_counts[row.mode] = mode_counts.get(row.mode, 0) + 1

    leaderboard: list[tuple[str, str, float, int]] = []
    excluded_engines: list[tuple[str, int]] = []
    for engine, scores in per_engine_scores.items():
        groups_played = len(scores)
        if groups_played < total_groups:
            excluded_engines.append((engine, groups_played))
            continue
        avg_score = sum(scores) / groups_played if groups_played else 0.0
        mode_counts = per_engine_best_modes.get(engine, {})
        best_mode = "-"
        if mode_counts:
            best_mode = sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        leaderboard.append((engine, best_mode, avg_score, groups_played))

    leaderboard.sort(key=lambda item: (-item[2], item[0].lower()))
    if not leaderboard:
        return []

    lines = ["## Leaderboard", ""]
    lines.append(
        "This is the engine-level view of the same canonical-metric logic. "
        "For each dataset/doc-count group, each engine contributes only its best mode for that group. "
        "The table shows the average raw canonical metric across those groups. "
        "Only engines with 100% group coverage are included here."
    )
    lines.append("")
    lines.append("Sorted by `Score` descending. Here `Score` is just the average canonical metric.")
    lines.append("")
    lines.append("| Rank | Engine | Best mode | Score ↓ |")
    lines.append("|---:|---|---|---:|")
    for idx, (engine, best_mode, score, groups_played) in enumerate(leaderboard, start=1):
        lines.append(
            f"| {idx} | {engine} | {_display_mode(best_mode)} | `{_bar_metric_text(score)}` |"
        )
    lines.append("")
    if excluded_engines:
        excluded_engines.sort(key=lambda item: item[0].lower())
        excluded_text = ", ".join(
            f"`{engine}` ({groups_played}/{total_groups})"
            for engine, groups_played in excluded_engines
        )
        lines.append(f"Excluded for incomplete coverage: {excluded_text}.")
        lines.append("")
    return lines


def _canonical_grouped_rows(rows: list[ArtifactRow]) -> tuple[dict[tuple[str, int | None], list[ArtifactRow]], int]:
    eligible = [row for row in rows if _eligible_for_global_relevance(row)]
    grouped: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in eligible:
        grouped.setdefault((row.dataset_source or "-", row.doc_count), []).append(row)
    return grouped, len(grouped)


def _engine_leaderboard_data(rows: list[ArtifactRow]) -> tuple[list[tuple[str, str, float]], list[tuple[str, int]], int]:
    grouped, total_groups = _canonical_grouped_rows(rows)
    if total_groups <= 0:
        return [], [], 0
    per_engine_scores: dict[str, list[float]] = {}
    per_engine_best_modes: dict[str, dict[str, int]] = {}
    for _, group_rows in grouped.items():
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_engine[row.engine] = row
        for engine, row in best_by_engine.items():
            per_engine_scores.setdefault(engine, []).append(_row_canonical_metric_value(row))
            mode_counts = per_engine_best_modes.setdefault(engine, {})
            mode_counts[row.mode] = mode_counts.get(row.mode, 0) + 1
    leaderboard: list[tuple[str, str, float]] = []
    excluded: list[tuple[str, int]] = []
    for engine, scores in per_engine_scores.items():
        groups_played = len(scores)
        if groups_played < total_groups:
            excluded.append((engine, groups_played))
            continue
        best_mode = "-"
        mode_counts = per_engine_best_modes.get(engine, {})
        if mode_counts:
            best_mode = sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        leaderboard.append((engine, best_mode, sum(scores) / groups_played))
    leaderboard.sort(key=lambda item: (-item[2], item[0].lower()))
    excluded.sort(key=lambda item: item[0].lower())
    return leaderboard, excluded, total_groups


def _engine_mode_leaderboard_data(rows: list[ArtifactRow]) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, int]], int]:
    grouped, total_groups = _canonical_grouped_rows(rows)
    if total_groups <= 0:
        return [], [], 0
    per_variant_scores: dict[tuple[str, str], list[float]] = {}
    for _, group_rows in grouped.items():
        best_by_variant: dict[tuple[str, str], ArtifactRow] = {}
        for row in group_rows:
            key = (row.engine, row.mode)
            prev = best_by_variant.get(key)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_variant[key] = row
        for key, row in best_by_variant.items():
            per_variant_scores.setdefault(key, []).append(_row_canonical_metric_value(row))
    leaderboard: list[tuple[str, str, float]] = []
    excluded: list[tuple[str, str, int]] = []
    for (engine, mode), scores in per_variant_scores.items():
        groups_played = len(scores)
        if groups_played < total_groups:
            excluded.append((engine, mode, groups_played))
            continue
        leaderboard.append((engine, mode, sum(scores) / groups_played))
    leaderboard.sort(key=lambda item: (-item[2], item[0].lower(), item[1].lower()))
    excluded.sort(key=lambda item: (item[0].lower(), item[1].lower()))
    return leaderboard, excluded, total_groups


def _mermaid_pie(title: str, parts: list[tuple[str, float]]) -> list[str]:
    if not parts:
        return []
    lines = ["```mermaid", "pie showData", f"    title \"{title}\""]
    for label, value in parts:
        lines.append(f"    \"{label}\" : {value:.4f}")
    lines.extend(["```", ""])
    return lines


def _quickchart_horizontal_bar(title: str, parts: list[tuple[str, float]]) -> list[str]:
    if not parts:
        return []
    labels = [label for label, _ in parts]
    values = [round(float(value), 4) for _, value in parts]
    height = max(360, 100 + 48 * len(parts))
    chart_cfg = {
        "type": "horizontalBar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Score",
                    "data": values,
                    "backgroundColor": "#3b82f6",
                    "borderColor": "#1d4ed8",
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "legend": {"display": False},
            "title": {"display": True, "text": title},
            "scales": {
                "xAxes": [
                    {
                        "ticks": {
                            "beginAtZero": True,
                            "min": 0,
                            "max": 1,
                            "precision": 4,
                        }
                    }
                ],
                "yAxes": [
                    {
                        "ticks": {
                            "autoSkip": False,
                        }
                    }
                ],
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1100&height={height}&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_pie(title: str, parts: list[tuple[str, float]]) -> list[str]:
    if not parts:
        return []
    labels = [label for label, _ in parts]
    values = [round(float(value), 4) for _, value in parts]
    colors = [
        "#3b82f6",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#06b6d4",
        "#84cc16",
        "#f97316",
        "#ec4899",
        "#6366f1",
        "#14b8a6",
        "#eab308",
    ]
    chart_cfg = {
        "type": "pie",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "data": values,
                    "backgroundColor": colors[: len(values)],
                    "borderColor": "#ffffff",
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1100&height=520&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_doughnut(title: str, parts: list[tuple[str, float]]) -> list[str]:
    if not parts:
        return []
    labels = [label for label, _ in parts]
    values = [round(float(value), 4) for _, value in parts]
    colors = [
        "#3b82f6",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#06b6d4",
        "#84cc16",
        "#f97316",
        "#ec4899",
        "#6366f1",
        "#14b8a6",
        "#eab308",
    ]
    chart_cfg = {
        "type": "doughnut",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "data": values,
                    "backgroundColor": colors[: len(values)],
                    "borderColor": "#ffffff",
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "cutoutPercentage": 55,
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1100&height=520&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_polar_area(title: str, parts: list[tuple[str, float]]) -> list[str]:
    if not parts:
        return []
    labels = [label for label, _ in parts]
    values = [round(float(value), 4) for _, value in parts]
    colors = [
        "#3b82f680",
        "#10b98180",
        "#f59e0b80",
        "#ef444480",
        "#8b5cf680",
        "#06b6d480",
        "#84cc1680",
        "#f9731680",
    ]
    chart_cfg = {
        "type": "polarArea",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "data": values,
                    "backgroundColor": colors[: len(values)],
                    "borderColor": colors[: len(values)],
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "ticks": {
                    "beginAtZero": True,
                },
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1100&height=560&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_radar(title: str, labels: list[str], datasets: list[dict[str, Any]]) -> list[str]:
    if not labels or not datasets:
        return []
    chart_cfg = {
        "type": "radar",
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "scale": {
                    "ticks": {
                        "beginAtZero": True,
                        "suggestedMax": 1.0,
                    }
                },
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1200&height=700&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_multi_line(title: str, labels: list[str], datasets: list[dict[str, Any]]) -> list[str]:
    if not labels or not datasets:
        return []
    chart_cfg = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "yAxes": [
                    {
                        "ticks": {
                            "beginAtZero": True,
                            "suggestedMax": 1.0,
                        }
                    }
                ],
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1300&height=700&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_multi_bar(title: str, labels: list[str], datasets: list[dict[str, Any]]) -> list[str]:
    if not labels or not datasets:
        return []
    chart_cfg = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "yAxes": [
                    {
                        "ticks": {
                            "beginAtZero": True,
                            "suggestedMax": 1.0,
                        }
                    }
                ],
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1400&height=760&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_stacked_bar(title: str, labels: list[str], datasets: list[dict[str, Any]]) -> list[str]:
    if not labels or not datasets:
        return []
    chart_cfg = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "xAxes": [{"stacked": True}],
                "yAxes": [{"stacked": True, "ticks": {"beginAtZero": True}}],
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1300&height=720&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _quickchart_bubble(title: str, datasets: list[dict[str, Any]]) -> list[str]:
    if not datasets:
        return []
    chart_cfg = {
        "type": "bubble",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "legend": {"position": "right"},
            "title": {"display": True, "text": title},
            "scales": {
                "xAxes": [
                    {
                        "ticks": {"beginAtZero": True, "min": 0, "max": 1},
                        "scaleLabel": {"display": True, "labelString": "Avg canonical score"},
                    }
                ],
                "yAxes": [
                    {
                        "ticks": {"beginAtZero": True, "min": 0, "max": 1},
                        "scaleLabel": {"display": True, "labelString": "Simplicity proxy (higher is simpler)"},
                    }
                ],
            },
        },
    }
    encoded = quote(json.dumps(chart_cfg, separators=(",", ":")))
    url = f"https://quickchart.io/chart?width=1200&height=700&format=png&c={encoded}"
    return [f"![{title}]({url})", ""]


def _heat_color(val: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    # light yellow -> orange -> deep blue
    stops = [
        (255, 247, 188),
        (254, 196, 79),
        (217, 95, 14),
        (49, 130, 189),
        (8, 81, 156),
    ]
    pos = t * (len(stops) - 1)
    i = int(pos)
    if i >= len(stops) - 1:
        r, g, b = stops[-1]
        return f"rgb({r},{g},{b})"
    frac = pos - i
    r1, g1, b1 = stops[i]
    r2, g2, b2 = stops[i + 1]
    r = round(r1 + (r2 - r1) * frac)
    g = round(g1 + (g2 - g1) * frac)
    b = round(b1 + (b2 - b1) * frac)
    return f"rgb({r},{g},{b})"


def _engine_group_win_shares(rows: list[ArtifactRow]) -> list[tuple[str, float]]:
    grouped, _ = _canonical_grouped_rows(rows)
    wins: dict[str, float] = {}
    for _, group_rows in grouped.items():
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_engine[row.engine] = row
        if not best_by_engine:
            continue
        best_value = max(_row_canonical_metric_value(row) for row in best_by_engine.values())
        winners = [engine for engine, row in best_by_engine.items() if _row_canonical_metric_value(row) == best_value]
        share = 1.0 / len(winners)
        for engine in winners:
            wins[engine] = wins.get(engine, 0.0) + share
    return sorted(wins.items(), key=lambda item: (-item[1], item[0].lower()))


def _engine_mode_group_win_shares(rows: list[ArtifactRow]) -> list[tuple[str, float]]:
    grouped, _ = _canonical_grouped_rows(rows)
    wins: dict[str, float] = {}
    for _, group_rows in grouped.items():
        best_by_variant: dict[tuple[str, str], ArtifactRow] = {}
        for row in group_rows:
            key = (row.engine, row.mode)
            prev = best_by_variant.get(key)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_variant[key] = row
        if not best_by_variant:
            continue
        best_value = max(_row_canonical_metric_value(row) for row in best_by_variant.values())
        winners = [f"{engine} / {_display_mode(mode)}" for (engine, mode), row in best_by_variant.items() if _row_canonical_metric_value(row) == best_value]
        share = 1.0 / len(winners)
        for label in winners:
            wins[label] = wins.get(label, 0.0) + share
    return sorted(wins.items(), key=lambda item: (-item[1], item[0].lower()))


def _engine_consistency_matrix_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    group_headers = [_dataset_group_label(dataset, doc_count) for dataset, doc_count in group_keys]
    engine_to_cells: dict[str, list[str]] = {}
    for engine, _, _ in leaderboard:
        engine_to_cells[engine] = []
        for key in group_keys:
            group_rows = grouped[key]
            best_val = None
            for row in group_rows:
                if row.engine != engine:
                    continue
                val = _row_canonical_metric_value(row)
                if best_val is None or val > best_val:
                    best_val = val
            engine_to_cells[engine].append(f"`{_bar_metric_text(best_val or 0.0, width=8)}`")
    lines = [
        "### 18. Consistency Matrix",
        "",
        "This compact matrix shows the best canonical score each engine achieves in every dataset/doc-count group. It is useful for spotting consistency across groups rather than only the global average.",
        "",
    ]
    cols = ["Engine", *group_headers]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for engine, _, _ in leaderboard:
        lines.append("| " + " | ".join([engine, *engine_to_cells[engine]]) + " |")
    lines.append("")
    return lines


def _engine_consistency_radar_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    palette = [
        ("qdrant", "#3b82f6"),
        ("manticore", "#10b981"),
        ("opensearch", "#f59e0b"),
        ("elasticsearch", "#06b6d4"),
        ("meilisearch", "#ef4444"),
        ("typesense", "#8b5cf6"),
    ]
    color_map = {name: color for name, color in palette}
    datasets: list[dict[str, Any]] = []
    for engine, _, _ in leaderboard:
        data: list[float] = []
        for key in group_keys:
            best_val = 0.0
            for row in grouped[key]:
                if row.engine != engine:
                    continue
                best_val = max(best_val, _row_canonical_metric_value(row))
            data.append(round(best_val, 4))
        color = color_map.get(engine, "#6366f1")
        datasets.append(
            {
                "label": engine,
                "data": data,
                "borderColor": color,
                "backgroundColor": color + "33",
                "pointBackgroundColor": color,
                "fill": True,
            }
        )
    return _quickchart_radar("Engine Consistency Radar", labels, datasets)


def _engine_consistency_series(rows: list[ArtifactRow]) -> tuple[list[str], list[dict[str, Any]]]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return [], []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    palette = [
        ("qdrant", "#3b82f6"),
        ("manticore", "#10b981"),
        ("opensearch", "#f59e0b"),
        ("elasticsearch", "#06b6d4"),
        ("meilisearch", "#ef4444"),
        ("typesense", "#8b5cf6"),
    ]
    color_map = {name: color for name, color in palette}
    datasets: list[dict[str, Any]] = []
    for engine, _, _ in leaderboard:
        data: list[float] = []
        for key in group_keys:
            best_val = 0.0
            for row in grouped[key]:
                if row.engine != engine:
                    continue
                best_val = max(best_val, _row_canonical_metric_value(row))
            data.append(round(best_val, 4))
        color = color_map.get(engine, "#6366f1")
        datasets.append(
            {
                "label": engine,
                "data": data,
                "borderColor": color,
                "backgroundColor": color + "33",
                "pointBackgroundColor": color,
                "pointRadius": 4,
                "pointHoverRadius": 5,
                "fill": False,
                "tension": 0.2,
            }
        )
    return labels, datasets


def _engine_consistency_heatmap_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    x_labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    y_labels = [engine for engine, _, _ in leaderboard]
    cells: list[tuple[str, str, float]] = []
    for engine, _, _ in leaderboard:
        for idx, key in enumerate(group_keys):
            best_val = 0.0
            for row in grouped[key]:
                if row.engine != engine:
                    continue
                best_val = max(best_val, _row_canonical_metric_value(row))
            cells.append((x_labels[idx], engine, best_val))
    lines = [
        "### 17. Heatmap",
        "",
        "Best for spotting hot and cold spots immediately. This HTML heatmap encodes each engine's best canonical score in every dataset/doc-count group as cell intensity.",
        "",
    ]
    cell_map = {(x, y): v for x, y, v in cells}
    values = [v for _, _, v in cells]
    lo = min(values)
    hi = max(values)
    lines.append('<div style="overflow-x:auto;">')
    lines.append('<table style="border-collapse:collapse; font-size:12px;">')
    lines.append("<thead>")
    header = ['<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">Engine</th>']
    for label in x_labels:
        header.append(
            f'<th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left; min-width:110px;">{label}</th>'
        )
    header.append("</tr>")
    lines.append("".join(header))
    lines.append("</thead>")
    lines.append("<tbody>")
    for engine in y_labels:
        row_parts = [
            f'<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">{engine}</th>'
        ]
        for label in x_labels:
            val = cell_map.get((label, engine), 0.0)
            color = _heat_color(val, lo, hi)
            row_parts.append(
                '<td style="border:1px solid #ddd; padding:6px; text-align:center; '
                f'background:{color}; color:#111;">{val:.4f}</td>'
            )
        row_parts.append("</tr>")
        lines.append("".join(row_parts))
    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("</div>")
    lines.append("")
    return lines


def _dataset_family_name(dataset_source: str) -> str:
    ds = str(dataset_source or "").strip().lower()
    if ds.startswith("msmarco-passage/"):
        return "MSMARCO"
    if ds.startswith("hf-beir:"):
        return "HF-BEIR"
    if ds.startswith("beir/"):
        return "BEIR"
    return "Other"


def _best_mode_counts_chart_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    engines = [engine for engine, _, _ in leaderboard]
    counts = {engine: {"fulltext": 0, "vector": 0, "hybrid": 0} for engine in engines}
    for _, group_rows in grouped.items():
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_engine[row.engine] = row
        for engine, row in best_by_engine.items():
            if engine in counts:
                counts[engine][row.mode] = counts[engine].get(row.mode, 0) + 1
    datasets = []
    for mode in ("fulltext", "vector", "hybrid"):
        datasets.append(
            {
                "label": _display_mode(mode),
                "data": [counts[engine].get(mode, 0) for engine in engines],
                "backgroundColor": _mode_color(mode),
                "borderColor": _mode_color(mode),
                "borderWidth": 1,
            }
        )
    return _quickchart_stacked_bar("Best Mode Counts by Engine", engines, datasets)


def _coverage_matrix_lines(rows: list[ArtifactRow]) -> list[str]:
    all_groups: list[tuple[str, int | None]] = sorted(
        {(row.dataset_source or "-", row.doc_count) for row in rows if row.engine and row.mode},
        key=lambda item: ((item[0] or "").lower(), item[1] or 0),
    )
    if not all_groups:
        return []
    variants = sorted(
        {(row.engine, row.mode) for row in rows if row.engine and row.mode},
        key=lambda item: (item[0].lower(), item[1].lower()),
    )
    usable = {
        (row.engine, row.mode, row.dataset_source or "-", row.doc_count)
        for row in rows
        if _eligible_for_global_relevance(row)
    }
    lines = [
        '<div style="overflow-x:auto;">',
        '<table style="border-collapse:collapse; font-size:12px;">',
        "<thead>",
    ]
    header = ['<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">Engine / mode</th>']
    for dataset, doc_count in all_groups:
        label = f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset
        header.append(f'<th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left; min-width:110px;">{label}</th>')
    header.append("</tr>")
    lines.append("".join(header))
    lines.append("</thead>")
    lines.append("<tbody>")
    for engine, mode in variants:
        row_parts = [
            f'<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">{engine} / {_display_mode(mode)}</th>'
        ]
        for dataset, doc_count in all_groups:
            present = (engine, mode, dataset, doc_count) in usable
            bg = "#dcfce7" if present else "#fee2e2"
            text = "OK" if present else "MISSING"
            row_parts.append(
                f'<td style="border:1px solid #ddd; padding:6px; text-align:center; background:{bg}; color:#111;">{text}</td>'
            )
        row_parts.append("</tr>")
        lines.append("".join(row_parts))
    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("</div>")
    lines.append("")
    return lines


def _dataset_family_preference_chart_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    engines = [engine for engine, _, _ in leaderboard]
    family_scores: dict[str, dict[str, list[float]]] = {}
    for (dataset, _), group_rows in grouped.items():
        family = _dataset_family_name(dataset)
        best_by_engine: dict[str, ArtifactRow] = {}
        for row in group_rows:
            prev = best_by_engine.get(row.engine)
            if prev is None or _row_canonical_metric_value(row) > _row_canonical_metric_value(prev):
                best_by_engine[row.engine] = row
        for engine, row in best_by_engine.items():
            family_scores.setdefault(family, {}).setdefault(engine, []).append(_row_canonical_metric_value(row))
    family_labels = sorted(family_scores.keys())
    palette = {
        "qdrant": "#3b82f6",
        "manticore": "#10b981",
        "opensearch": "#f59e0b",
        "elasticsearch": "#06b6d4",
        "meilisearch": "#ef4444",
        "typesense": "#8b5cf6",
    }
    datasets = []
    for engine in engines:
        values = []
        for family in family_labels:
            scores = family_scores.get(family, {}).get(engine, [])
            values.append(round(sum(scores) / len(scores), 4) if scores else 0.0)
        color = palette.get(engine, "#6366f1")
        datasets.append(
            {
                "label": engine,
                "data": values,
                "backgroundColor": color,
                "borderColor": color,
                "borderWidth": 1,
            }
        )
    return _quickchart_multi_bar("Engine Preference by Dataset Family", family_labels, datasets)


def _quality_vs_simplicity_chart_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    wins = dict(_engine_group_win_shares(rows))
    palette = {
        "qdrant": "#3b82f6",
        "manticore": "#10b981",
        "opensearch": "#f59e0b",
        "elasticsearch": "#06b6d4",
        "meilisearch": "#ef4444",
        "typesense": "#8b5cf6",
    }
    datasets = []
    for engine, _, avg_score in leaderboard:
        modes_used: set[str] = set()
        for _, group_rows in grouped.items():
            best_row = None
            best_val = -1.0
            for row in group_rows:
                if row.engine != engine:
                    continue
                val = _row_canonical_metric_value(row)
                if val > best_val:
                    best_val = val
                    best_row = row
            if best_row is not None:
                modes_used.add(best_row.mode)
        simplicity = 1.0 / max(1, len(modes_used))
        radius = 8 + 6 * wins.get(engine, 0.0)
        color = palette.get(engine, "#6366f1")
        datasets.append(
            {
                "label": engine,
                "data": [{"x": round(avg_score, 4), "y": round(simplicity, 4), "r": round(radius, 2)}],
                "backgroundColor": color + "88",
                "borderColor": color,
                "borderWidth": 1,
            }
        )
    return _quickchart_bubble("Quality vs Simplicity Proxy", datasets)


def _top_engine_mode_relative_heatmap_lines(rows: list[ArtifactRow], limit: int = 5) -> list[str]:
    leaderboard, _, total_groups = _engine_mode_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    top = leaderboard[:limit]
    x_labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    y_labels = [f"{engine} / {_display_mode(mode)}" for engine, mode, _ in top]

    lines = [
        '<div style="overflow-x:auto;">',
        '<table style="border-collapse:collapse; font-size:12px;">',
        "<thead>",
    ]
    header = ['<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">Engine / mode</th>']
    for label in x_labels:
        header.append(
            f'<th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left; min-width:110px;">{label}</th>'
        )
    header.append("</tr>")
    lines.append("".join(header))
    lines.append("</thead>")
    lines.append("<tbody>")

    for engine, mode, _ in top:
        row_parts = [
            f'<tr><th style="border:1px solid #ddd; padding:6px; background:#f8f8f8; text-align:left;">{engine} / {_display_mode(mode)}</th>'
        ]
        for key in group_keys:
            group_rows = grouped[key]
            group_best = max(_row_canonical_metric_value(row) for row in group_rows) if group_rows else 0.0
            own_best = 0.0
            for row in group_rows:
                if row.engine == engine and row.mode == mode:
                    own_best = max(own_best, _row_canonical_metric_value(row))
            rel = (own_best / group_best) if group_best > 0 else 0.0
            color = _heat_color(rel, 0.0, 1.0)
            row_parts.append(
                '<td style="border:1px solid #ddd; padding:6px; text-align:center; '
                f'background:{color}; color:#111;">{rel:.3f}</td>'
            )
        row_parts.append("</tr>")
        lines.append("".join(row_parts))

    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("</div>")
    lines.append("")
    return lines


def _top_engine_mode_series(rows: list[ArtifactRow], limit: int = 5) -> tuple[list[str], list[dict[str, Any]]]:
    leaderboard, _, total_groups = _engine_mode_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return [], []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
    datasets: list[dict[str, Any]] = []
    for idx, (engine, mode, _) in enumerate(leaderboard[:limit]):
        data: list[float] = []
        for key in group_keys:
            best_val = 0.0
            for row in grouped[key]:
                if row.engine == engine and row.mode == mode:
                    best_val = max(best_val, _row_canonical_metric_value(row))
            data.append(round(best_val, 4))
        color = palette[idx % len(palette)]
        datasets.append(
            {
                "label": f"{engine} / {_display_mode(mode)}",
                "data": data,
                "borderColor": color,
                "backgroundColor": color + "33",
                "pointBackgroundColor": color,
                "pointRadius": 4,
                "showLine": True,
                "fill": False,
                "tension": 0.15,
            }
        )
    return labels, datasets


def _mode_color(mode: str) -> str:
    m = str(mode or "").lower()
    if m == "fulltext":
        return "#6b7280"
    if m == "vector":
        return "#2563eb"
    if m == "hybrid":
        return "#16a34a"
    return "#6366f1"


def _best_mode_per_engine_grouped_bar_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    lines = [
        "### 12. Best Mode Per Engine by Group",
        "",
        "Best for seeing which retrieval mode carries each engine. This variant uses one grouped bar chart per dataset group, so the mode colors stay unambiguous: gray for full-text, blue for vector, green for hybrid.",
        "",
    ]
    engine_labels = [engine for engine, _, _ in leaderboard]
    for dataset, doc_count in group_keys:
        group_label = f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset
        mode_values: dict[str, list[float | None]] = {
            "fulltext": [],
            "vector": [],
            "hybrid": [],
        }
        group_rows = grouped[(dataset, doc_count)]
        for engine in engine_labels:
            best_row = None
            best_val = -1.0
            for row in group_rows:
                if row.engine != engine:
                    continue
                val = _row_canonical_metric_value(row)
                if val > best_val:
                    best_val = val
                    best_row = row
            chosen_mode = best_row.mode if best_row else ""
            for mode in mode_values:
                mode_values[mode].append(round(max(best_val, 0.0), 4) if chosen_mode == mode else None)
        datasets: list[dict[str, Any]] = []
        for mode in ("fulltext", "vector", "hybrid"):
            datasets.append(
                {
                    "label": _display_mode(mode),
                    "data": mode_values[mode],
                    "backgroundColor": _mode_color(mode),
                    "borderColor": _mode_color(mode),
                    "borderWidth": 1,
                }
            )
        lines.append(f"#### `{group_label}`")
        lines.append("")
        lines.extend(_quickchart_multi_bar(f"Best Mode Per Engine: {group_label}", engine_labels, datasets))
    lines.append("Mode colors: `full-text = gray`, `vector = blue`, `hybrid = green`.")
    lines.append("")
    return lines


def _relative_to_winner_grouped_bar_lines(rows: list[ArtifactRow]) -> list[str]:
    leaderboard, _, total_groups = _engine_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    palette = {
        "qdrant": "#3b82f6",
        "manticore": "#10b981",
        "opensearch": "#f59e0b",
        "elasticsearch": "#06b6d4",
        "meilisearch": "#ef4444",
        "typesense": "#8b5cf6",
    }
    datasets: list[dict[str, Any]] = []
    for engine, _, _ in leaderboard:
        values: list[float] = []
        for key in group_keys:
            group_rows = grouped[key]
            best_score = max(_row_canonical_metric_value(row) for row in group_rows)
            best_for_engine = 0.0
            for row in group_rows:
                if row.engine != engine:
                    continue
                best_for_engine = max(best_for_engine, _row_canonical_metric_value(row))
            rel = (best_for_engine / best_score) if best_score > 0 else 0.0
            values.append(round(rel, 4))
        color = palette.get(engine, "#6366f1")
        datasets.append(
            {
                "label": engine,
                "data": values,
                "backgroundColor": color,
                "borderColor": color,
                "borderWidth": 1,
            }
        )
    lines = [
        "### 13. Relative-to-Winner by Group",
        "",
        "Best for fair cross-dataset comparison. In each dataset group, the winning engine gets `1.0`, and the others show how close they are to that winner. This removes a lot of the distortion caused by very different raw metric ranges across datasets.",
        "",
    ]
    lines.extend(_quickchart_multi_bar("Relative to Group Winner", labels, datasets))
    return lines


def _mode_family_grouped_bar_lines(rows: list[ArtifactRow]) -> list[str]:
    grouped, total_groups = _canonical_grouped_rows(rows)
    if total_groups <= 0:
        return []
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
    mode_order = [("fulltext", "full-text"), ("vector", "vector"), ("hybrid", "hybrid")]
    datasets: list[dict[str, Any]] = []
    for mode, label in mode_order:
        values: list[float] = []
        for key in group_keys:
            best_val = 0.0
            for row in grouped[key]:
                if row.mode != mode:
                    continue
                best_val = max(best_val, _row_canonical_metric_value(row))
            values.append(round(best_val, 4))
        color = _mode_color(mode)
        datasets.append(
            {
                "label": label,
                "data": values,
                "backgroundColor": color,
                "borderColor": color,
                "borderWidth": 1,
            }
        )
    lines = [
        "### 14. Mode Family Comparison",
        "",
        "Best for seeing whether the gains mostly come from full-text, vector, or hybrid as a family. Each bar is the best score any engine achieved in that mode family for the dataset group.",
        "",
    ]
    lines.extend(_quickchart_multi_bar("Best Score by Mode Family", labels, datasets))
    return lines


def _top_engine_mode_grouped_bar_lines(rows: list[ArtifactRow], limit: int = 6) -> list[str]:
    leaderboard, _, total_groups = _engine_mode_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    top = leaderboard[:limit]
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    lines = [
        "### 15. Top Engine + Mode Grouped Comparison",
        "",
        "Best for comparing the strongest concrete variants directly. This variant uses one chart per dataset group, and every bar is colored by its mode family, which keeps the mode colors unambiguous.",
        "",
    ]
    variant_labels = [f"{engine} / {_display_mode(mode)}" for engine, mode, _ in top]
    for dataset, doc_count in group_keys:
        group_label = f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset
        group_rows = grouped[(dataset, doc_count)]
        datasets: list[dict[str, Any]] = []
        for engine, mode, _ in top:
            own_score = 0.0
            for row in group_rows:
                if row.engine == engine and row.mode == mode:
                    own_score = max(own_score, _row_canonical_metric_value(row))
            color = _mode_color(mode)
            datasets.append(
                {
                    "label": f"{engine} / {_display_mode(mode)}",
                    "data": [round(own_score, 4)],
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            )
        lines.append(f"#### `{group_label}`")
        lines.append("")
        lines.extend(_quickchart_multi_bar(f"Top Engine + Mode: {group_label}", ["score"], datasets))
    lines.append("Mode colors: `full-text = gray`, `vector = blue`, `hybrid = green`.")
    lines.append("")
    return lines


def _engine_mode_relative_to_winner_grouped_bar_lines(rows: list[ArtifactRow], limit: int = 6) -> list[str]:
    leaderboard, _, total_groups = _engine_mode_leaderboard_data(rows)
    if not leaderboard or total_groups <= 0:
        return []
    grouped, _ = _canonical_grouped_rows(rows)
    top = leaderboard[:limit]
    group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
    lines = [
        "### 16. Engine + Mode Relative-to-Winner",
        "",
        "Best for seeing which top variants are consistently near the winner, even when they are not actually first. This variant also uses one chart per dataset group, so the mode colors stay unambiguous.",
        "",
    ]
    for dataset, doc_count in group_keys:
        group_label = f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset
        group_rows = grouped[(dataset, doc_count)]
        best_score = max(_row_canonical_metric_value(row) for row in group_rows) if group_rows else 0.0
        datasets: list[dict[str, Any]] = []
        for engine, mode, _ in top:
            own_score = 0.0
            for row in group_rows:
                if row.engine == engine and row.mode == mode:
                    own_score = max(own_score, _row_canonical_metric_value(row))
            rel = (own_score / best_score) if best_score > 0 else 0.0
            color = _mode_color(mode)
            datasets.append(
                {
                    "label": f"{engine} / {_display_mode(mode)}",
                    "data": [round(rel, 4)],
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            )
        lines.append(f"#### `{group_label}`")
        lines.append("")
        lines.extend(_quickchart_multi_bar(f"Top Engine + Mode Relative to Winner: {group_label}", ["score"], datasets))
    lines.append("Mode colors: `full-text = gray`, `vector = blue`, `hybrid = green`.")
    lines.append("")
    return lines


def _leaderboard_visualization_options_lines(rows: list[ArtifactRow]) -> list[str]:
    lines = ["## Questions Answered Visually", ""]
    engine_board, _, _ = _engine_leaderboard_data(rows)
    engine_mode_board, _, _ = _engine_mode_leaderboard_data(rows)
    line_labels, line_datasets = _engine_consistency_series(rows)
    top_labels, top_datasets = _top_engine_mode_series(rows, limit=4)
    if engine_mode_board:
        lines.append("### 1. Which engine and mode gives me the best relevance overall?")
        lines.append("")
        lines.append("Best chart: a simple engine+mode leaderboard bar chart.")
        lines.append("")
        parts = [(f"{engine} / {_display_mode(mode)}", score) for engine, mode, score in engine_mode_board]
        lines.extend(_quickchart_horizontal_bar("Engine + Mode Leaderboard Scores", parts))
    if top_labels and top_datasets:
        lines.append("### 2. Is that winner consistently good, or is it only carried by a few datasets?")
        lines.append("")
        lines.append("Best chart: a relative-to-winner heatmap for the top engine+mode variants. Values close to `1.0` mean the variant stays very close to the group winner, while colder cells show where it falls behind.")
        lines.append("")
        lines.extend(_top_engine_mode_relative_heatmap_lines(rows, limit=5))
    if line_labels and line_datasets:
        lines.append("### 3. For each dataset I care about, who wins and by how much?")
        lines.append("")
        lines.append("Best chart: grouped bars by dataset group using each engine's best score there.")
        lines.append("")
        bar_datasets = [
            {
                "label": ds["label"],
                "data": ds["data"],
                "backgroundColor": ds["borderColor"],
                "borderColor": ds["borderColor"],
                "borderWidth": 1,
            }
            for ds in line_datasets
        ]
        lines.extend(_quickchart_multi_bar("Engine Scores by Dataset Group", line_labels, bar_datasets))
    lines.append("### 4. Does hybrid actually beat vector and full-text, or is it mostly extra complexity?")
    lines.append("")
    lines.append("Best chart: mode-family comparison by dataset group.")
    lines.append("")
    grouped, total_groups = _canonical_grouped_rows(rows)
    if total_groups > 0:
        group_keys = sorted(grouped.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0))
        family_labels = [f"{dataset} ({int(doc_count):,})" if doc_count is not None else dataset for dataset, doc_count in group_keys]
        mode_datasets = []
        for mode in ("fulltext", "vector", "hybrid"):
            vals = []
            for key in group_keys:
                best_val = 0.0
                for row in grouped[key]:
                    if row.mode == mode:
                        best_val = max(best_val, _row_canonical_metric_value(row))
                vals.append(round(best_val, 4))
            mode_datasets.append(
                {
                    "label": _display_mode(mode),
                    "data": vals,
                    "backgroundColor": _mode_color(mode),
                    "borderColor": _mode_color(mode),
                    "borderWidth": 1,
                }
            )
        lines.extend(_quickchart_multi_bar("Best Score by Mode Family", family_labels, mode_datasets))
    lines.append("### 5. What is the best mode for each engine?")
    lines.append("")
    lines.append("Best chart: stacked counts of how often each engine's best result came from full-text, vector, or hybrid.")
    lines.append("")
    lines.extend(_best_mode_counts_chart_lines(rows))
    lines.append("### 6. How close are the runners-up? Are the gaps real or tiny?")
    lines.append("")
    lines.append("Best chart: relative-to-winner grouped bars, where each group winner is normalized to `1.0`.")
    lines.append("")
    rel_lines = _relative_to_winner_grouped_bar_lines(rows)
    lines.extend(rel_lines[4:] if len(rel_lines) > 4 else rel_lines)
    lines.append("### 7. Which engines or modes fail to cover all datasets, or have missing runs?")
    lines.append("")
    lines.append("Best chart: a coverage matrix showing which engine+mode combinations have usable runs for each dataset group.")
    lines.append("")
    lines.extend(_coverage_matrix_lines(rows))
    lines.append("### 8. Do different dataset families favor different engines?")
    lines.append("")
    lines.append("Best chart: grouped bars aggregated by dataset family.")
    lines.append("")
    lines.extend(_dataset_family_preference_chart_lines(rows))
    lines.append("### 9. If I ignore pure relevance for a moment, which engine is the best tradeoff between quality and operational simplicity?")
    lines.append("")
    lines.append("Best chart: a bubble chart using average canonical score on one axis and a simplicity proxy on the other. The simplicity proxy here is fewer distinct best modes needed across groups.")
    lines.append("")
    lines.extend(_quality_vs_simplicity_chart_lines(rows))
    lines.append("### 10. Are there any obvious outliers or surprising regressions on specific datasets that I should investigate before trusting the summary?")
    lines.append("")
    lines.append("Best chart: a heatmap of best engine scores by dataset group.")
    lines.append("")
    heatmap_lines = _engine_consistency_heatmap_lines(rows)
    lines.extend(heatmap_lines[4:] if len(heatmap_lines) > 4 else heatmap_lines)
    return lines


def render_combined_report(rows: list[ArtifactRow], title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    engine_lines = _engine_only_leaderboard_lines(rows)
    if engine_lines:
        lines.extend(engine_lines)
    variant_lines = _global_variant_leaderboard_lines(rows)
    if variant_lines:
        lines.extend(variant_lines)
    lines.extend(_leaderboard_visualization_options_lines(rows))
    by_dataset: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in rows:
        key = (row.dataset_source or "-", row.doc_count)
        by_dataset.setdefault(key, []).append(row)

    for dataset, doc_count in sorted(by_dataset.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0)):
        dataset_rows = by_dataset[(dataset, doc_count)]
        section_label = _dataset_group_label(dataset, doc_count)
        lines.append(f"## {section_label}")
        lines.append("")
        canonical_table_lines = _canonical_metric_table_lines(dataset_rows)
        lines.extend(canonical_table_lines)
        if canonical_table_lines:
            lines.append("")
        show_docs = len({r.doc_count for r in dataset_rows}) > 1
        rel_keys = _collect_relevance_keys(dataset_rows)
        lines.append("<details>")
        lines.append("<summary>Full Relevance Table</summary>")
        lines.append("")
        rel_cols = ["Run ID", "Profile/System/Model"]
        if show_docs:
            rel_cols.append("Docs")
        rel_cols.extend(["Run At", *[k.upper() for k in rel_keys]])
        header = "| " + " | ".join(rel_cols) + " |"
        sep = "|" + "---|" * len(rel_cols)
        lines.append(header)
        lines.append(sep)
        rel_rows = sorted(dataset_rows, key=_relevance_sort_key)
        for r in rel_rows:
            run_id_link = f"[{r.run_id}]({r.run_id}/summary.md)"
            vals = [f"{r.relevance.get(k, 0.0):.4f}" for k in rel_keys]
            row_vals = [
                run_id_link,
                _profile_system_model_label(r),
            ]
            if show_docs:
                row_vals.append(f"{int(r.doc_count):,}" if r.doc_count is not None else "-")
            row_vals.extend([r.run_at, *vals])
            lines.append(
                "| " + " | ".join(row_vals) + " |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Full Query Performance Table</summary>")
        lines.append("")
        lines.append(
            "_Performance numbers below are informational only. This benchmark is primarily focused on relevance, "
            "so latency/QPS figures may be noisy or inaccurate and should not be treated as authoritative._"
        )
        lines.append("")
        perf_cols = ["Run ID", "Profile/System/Model"]
        if show_docs:
            perf_cols.append("Docs")
        perf_cols.extend(["Run At", "Concurrency", "P50 (ms)", "P95 (ms)", "P99 (ms)", "QPS", "Error rate", "Queries"])
        lines.append("| " + " | ".join(perf_cols) + " |")
        lines.append("|" + "---|" * len(perf_cols))
        perf_rows = sorted(dataset_rows, key=_performance_sort_key)
        for r in perf_rows:
            run_id_link = f"[{r.run_id}]({r.run_id}/summary.md)"
            p = r.performance
            row_vals = [
                run_id_link,
                _profile_system_model_label(r),
            ]
            if show_docs:
                row_vals.append(f"{int(r.doc_count):,}" if r.doc_count is not None else "-")
            row_vals.extend(
                [
                    r.run_at,
                    str(r.concurrency if r.concurrency is not None else "-"),
                    f"{p.get('latency_p50_ms', 0.0):.2f}",
                    f"{p.get('latency_p95_ms', 0.0):.2f}",
                    f"{p.get('latency_p99_ms', 0.0):.2f}",
                    f"{p.get('qps', 0.0):.2f}",
                    f"{p.get('error_rate', 0.0):.4f}",
                    f"{p.get('queries_total', 0.0):.0f}",
                ]
            )
            lines.append(
                "| " + " | ".join(row_vals) + " |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(metric_definitions_markdown().strip())
    lines.append("")
    return "\n".join(lines)


def generate_combined_report(
    run_ids: list[str] | None = None,
    interactive: bool = False,
    output_path: str | None = None,
    title: str = "Benchmark Report",
) -> Path:
    all_rows = discover_artifacts()
    selected: list[ArtifactRow]
    if run_ids:
        wanted = set(run_ids)
        selected = [r for r in all_rows if r.run_id in wanted]
    elif interactive:
        selected = select_artifacts_interactive(all_rows)
    else:
        selected = all_rows
    if not selected:
        raise RuntimeError("No artifacts selected.")

    out = (
        Path(output_path)
        if output_path
        else Path("artifacts") / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_combined_report(selected, title=title), encoding="utf-8")
    return out
