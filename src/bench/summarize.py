from __future__ import annotations

import curses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys
from typing import Any

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
        blocks = max(1, int(round(val * width))) if val > 0 else 0
        bar = "█" * blocks if blocks > 0 else "·"
        lines.append(f"| {label} | `{bar} {val:.4f}` |")
    return lines


def render_combined_report(rows: list[ArtifactRow], title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    by_dataset: dict[tuple[str, int | None], list[ArtifactRow]] = {}
    for row in rows:
        key = (row.dataset_source or "-", row.doc_count)
        by_dataset.setdefault(key, []).append(row)

    for dataset, doc_count in sorted(by_dataset.keys(), key=lambda item: ((item[0] or "").lower(), item[1] or 0)):
        dataset_rows = by_dataset[(dataset, doc_count)]
        section_label = _dataset_group_label(dataset, doc_count)
        lines.append(f"## Relevance for {section_label}")
        lines.append("")
        ndcg_table_lines = _metric_table_lines(dataset_rows, "ndcg@10")
        lines.extend(ndcg_table_lines)
        if ndcg_table_lines:
            lines.append("")
        mrr_table_lines = _metric_table_lines(dataset_rows, "mrr@10")
        lines.extend(mrr_table_lines)
        if mrr_table_lines:
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
        lines.append(f"## Query Performance for {section_label}")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Full Query Performance Table</summary>")
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
