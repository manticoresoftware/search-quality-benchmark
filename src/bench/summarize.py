from __future__ import annotations

import curses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any

from .io_utils import read_json
from .reporting import metric_definitions_markdown


@dataclass
class ArtifactRow:
    run_id: str
    run_at_unix: int
    run_at: str
    path: Path
    engine: str
    mode: str
    model: str
    runtime_seconds: float
    concurrency: int | None
    relevance: dict[str, float]
    performance: dict[str, float]
    status: str


def _system_label(engine: str, mode: str) -> str:
    if engine == "manticore" and mode == "fulltext":
        return "Manticore full-text (manticore)"
    if engine == "manticore" and mode == "vector":
        return "Manticore vector (manticore)"
    if engine == "qdrant" and mode == "fulltext":
        return "Qdrant full-text (qdrant)"
    if engine == "qdrant" and mode == "vector":
        return "Qdrant vector (qdrant)"
    return f"{engine} {mode}"


def _display_mode(mode: str) -> str:
    m = mode.lower()
    if m == "fulltext":
        return "full-text"
    if m == "vector":
        return "vector"
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
                    run_at_unix=run_at_unix_val,
                    run_at=_run_at_from_unix(run_at_unix_val),
                    path=p,
                    engine=str(payload.get("engine", "")),
                    mode=mode,
                    model=model,
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
                    run_at_unix=int(p.stat().st_mtime),
                    run_at=_run_at_from_unix(int(p.stat().st_mtime)),
                    path=p,
                    engine="",
                    mode="",
                    model="n/a",
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
    return 0


def _color_pair_for_mode(mode: str) -> int:
    m = mode.lower()
    if m == "fulltext":
        return 3
    if m == "vector":
        return 4
    return 0


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _wrap_cell(text: str, width: int, max_lines: int = 2) -> list[str]:
    if max_lines <= 1:
        return [_fit(text, width)]
    src = str(text or "")
    out: list[str] = []
    while src and len(out) < max_lines:
        if len(src) <= width:
            out.append(src.ljust(width))
            src = ""
            break
        if len(out) == max_lines - 1:
            out.append(_fit(src, width))
            src = ""
            break
        out.append(src[:width].ljust(width))
        src = src[width:]
    while len(out) < max_lines:
        out.append("".ljust(width))
    return out


def _dynamic_model_lines(total_rows: int, term_height: int) -> int:
    available = max(1, term_height - 4)
    target_rows = max(1, min(total_rows, 10))
    lines = available // target_rows
    if lines < 1:
        return 1
    if lines > 3:
        return 3
    return lines


def _draw_row(
    stdscr: Any,
    y: int,
    row: ArtifactRow,
    idx: int,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    model_lines_max: int,
) -> None:
    prefix = "[x]" if selected else "[ ]"
    attrs = curses.A_REVERSE if cursor else curses.A_NORMAL
    x = 0
    stdscr.addstr(y, x, _fit(prefix, widths["sel"]), attrs)
    x += widths["sel"] + 1
    stdscr.addstr(y, x, _fit(str(idx), widths["idx"]), attrs)
    x += widths["idx"] + 1
    stdscr.addstr(y, x, _fit(row.run_id, widths["run_id"]), attrs)
    x += widths["run_id"] + 1
    stdscr.addstr(y, x, _fit(row.run_at, widths["run_at"]), attrs)
    x += widths["run_at"] + 1
    e_pair = _color_pair_for_engine(row.engine)
    stdscr.addstr(y, x, _fit(row.engine or "-", widths["engine"]), attrs | curses.color_pair(e_pair))
    x += widths["engine"] + 1
    m_pair = _color_pair_for_mode(row.mode)
    stdscr.addstr(y, x, _fit(_display_mode(row.mode or "-"), widths["mode"]), attrs | curses.color_pair(m_pair))
    x += widths["mode"] + 1
    model_lines = _wrap_cell(row.model or "n/a", widths["model"], max_lines=model_lines_max)
    stdscr.addstr(y, x, model_lines[0], attrs)
    x += widths["model"] + 1
    stdscr.addstr(y, x, _fit(row.status, widths["status"]), attrs)
    x += widths["status"] + 1
    ndcg_val = row.relevance.get("ndcg@10", 0.0)
    stdscr.addstr(y, x, _fit(f"{ndcg_val:.4f}", widths["ndcg"]), attrs)
    x += widths["ndcg"] + 1
    map_val = row.relevance.get("map@10", 0.0)
    stdscr.addstr(y, x, _fit(f"{map_val:.4f}", widths["map"]), attrs)
    x += widths["map"] + 1
    p20_val = row.relevance.get("p@20", 0.0)
    stdscr.addstr(y, x, _fit(f"{p20_val:.4f}", widths["p20"]), attrs)
    x += widths["p20"] + 1
    recall_val = row.relevance.get("recall@100", 0.0)
    stdscr.addstr(y, x, _fit(f"{recall_val:.4f}", widths["recall"]), attrs)
    x_model = (
        widths["sel"] + 1
        + widths["idx"] + 1
        + widths["run_id"] + 1
        + widths["run_at"] + 1
        + widths["engine"] + 1
        + widths["mode"] + 1
    )
    for i in range(1, model_lines_max):
        stdscr.addstr(y + i, x_model, model_lines[i], attrs)


def _draw_header(
    stdscr: Any,
    y: int,
    widths: dict[str, int],
    max_width: int,
    sort_key: str,
    sort_desc: bool,
) -> None:
    x = 0
    cols = [
        ("SEL", "sel"),
        ("#", "idx"),
        ("RUN_ID", "run_id"),
        ("RUN_AT", "run_at"),
        ("ENGINE", "engine"),
        ("MODE", "mode"),
        ("MODEL/OPTS", "model"),
        ("STATUS", "status"),
        ("NDCG@10", "ndcg"),
        ("MAP@10", "map"),
        ("P@20", "p20"),
        ("RECALL@100", "recall"),
    ]
    for title, key in cols:
        if x >= max_width:
            break
        label = title
        attrs = curses.A_UNDERLINE | curses.A_BOLD
        if key == sort_key:
            label = f"{title}{'↓' if sort_desc else '↑'}"
            attrs |= curses.A_REVERSE
        stdscr.addstr(y, x, _fit(label, widths[key]), attrs)
        x += widths[key] + 1


def _select_artifacts_curses(rows: list[ArtifactRow]) -> list[ArtifactRow]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0

    def _after_delete_reindex(selected: set[int], deleted_idx: int) -> set[int]:
        out: set[int] = set()
        for i in selected:
            if i == deleted_idx:
                continue
            out.add(i - 1 if i > deleted_idx else i)
        return out

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset, selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)     # manticore
            curses.init_pair(2, curses.COLOR_YELLOW, -1)    # qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)      # full-text
            curses.init_pair(4, curses.COLOR_RED, -1)       # vector

        widths = {
            "sel": 3,
            "idx": 3,
            "run_id": 10,
            "run_at": 19,
            "engine": 10,
            "mode": 9,
            "model": 36,
            "status": 7,
            "ndcg": 8,
            "map": 8,
            "p20": 8,
            "recall": 10,
        }
        sort_by = "run_at"
        sort_desc = True
        sort_cols = ["run_at", "engine", "mode", "model", "status", "ndcg", "map", "p20", "recall", "run_id"]
        sort_idx = sort_cols.index(sort_by)

        def _default_sort_desc(col: str) -> bool:
            return col in {"run_at", "ndcg", "map", "p20", "recall"}

        def _row_sort_key(row: ArtifactRow, col: str) -> Any:
            if col == "run_at":
                return float(row.run_at_unix or 0)
            if col == "engine":
                return (row.engine or "").lower()
            if col == "mode":
                return (row.mode or "").lower()
            if col == "model":
                return (row.model or "").lower()
            if col == "status":
                return (row.status or "").lower()
            if col == "ndcg":
                return float(row.relevance.get("ndcg@10", 0.0))
            if col == "map":
                return float(row.relevance.get("map@10", 0.0))
            if col == "p20":
                return float(row.relevance.get("p@20", 0.0))
            if col == "recall":
                return float(row.relevance.get("recall@100", 0.0))
            if col == "run_id":
                return row.run_id
            return row.run_id

        def _apply_sort(col: str, desc: bool) -> None:
            nonlocal rows, selected_idx, current, offset
            if not rows:
                return
            selected_ids = {rows[i].run_id for i in selected_idx if 0 <= i < len(rows)}
            current_id = rows[current].run_id if 0 <= current < len(rows) else None
            rows.sort(key=lambda r: _row_sort_key(r, col), reverse=desc)
            selected_idx = {i for i, r in enumerate(rows) if r.run_id in selected_ids}
            if current_id is not None:
                for i, r in enumerate(rows):
                    if r.run_id == current_id:
                        current = i
                        break
            offset = 0

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            row_h = _dynamic_model_lines(len(rows), h)
            help1 = "↑/↓ move  ←/→ sort col  SPACE toggle  a select all  1-0 sort  d delete  ENTER submit  ESC/q cancel"
            stdscr.addstr(0, 0, _fit(help1, max(1, w - 1)), curses.A_BOLD)
            _draw_header(
                stdscr,
                1,
                widths=widths,
                max_width=max(1, w - 1),
                sort_key=sort_by,
                sort_desc=sort_desc,
            )

            visible = max(1, (h - 4) // row_h)
            if current < offset:
                offset = current
            if current >= offset + visible:
                offset = current - visible + 1

            for i in range(visible):
                ridx = offset + i
                if ridx >= len(rows):
                    break
                y = 2 + i * row_h
                _draw_row(
                    stdscr=stdscr,
                    y=y,
                    row=rows[ridx],
                    idx=ridx + 1,
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    model_lines_max=row_h,
                )

            sort_hint = f"{sort_by}{' desc' if sort_desc else ' asc'}"
            summary = f"Selected: {len(selected_idx)} / {len(rows)}  |  Sort: {sort_hint}"
            stdscr.addstr(h - 1, 0, _fit(summary, max(1, w - 1)), curses.A_DIM)
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
                _apply_sort(sort_by, sort_desc)
            elif ch == curses.KEY_RIGHT:
                sort_idx = (sort_idx + 1) % len(sort_cols)
                sort_by = sort_cols[sort_idx]
                sort_desc = _default_sort_desc(sort_by)
                _apply_sort(sort_by, sort_desc)
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
            elif ch in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9"), ord("0")):
                key_to_col = {
                    ord("1"): "run_at",
                    ord("2"): "engine",
                    ord("3"): "mode",
                    ord("4"): "model",
                    ord("5"): "status",
                    ord("6"): "ndcg",
                    ord("7"): "map",
                    ord("8"): "p20",
                    ord("9"): "recall",
                    ord("0"): "run_id",
                }
                new_col = key_to_col[ch]
                if sort_by == new_col:
                    sort_desc = not sort_desc
                else:
                    sort_by = new_col
                    sort_idx = sort_cols.index(sort_by)
                    sort_desc = _default_sort_desc(new_col)
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
                if current >= offset + visible:
                    offset = max(0, current - visible + 1)
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
        print(
            f"{i:>2}.\trun_id={r.run_id}\trun_at={r.run_at}\tengine={r.engine or '-'}\tmode={_display_mode(r.mode or '-')}\t"
            f"model={r.model or 'n/a'}\tstatus={r.status}\tndcg@10={ndcg_val:.4f}\t"
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
    if "p@20" in keys:
        ordered.append("p@20")
    if "recall@100" in keys:
        ordered.append("recall@100")
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    return ordered


def _relevance_sort_key(r: ArtifactRow) -> tuple[float, float, float, float, float, str, float]:
    ndcg10 = float(r.relevance.get("ndcg@10", 0.0))
    ndcg20 = float(r.relevance.get("ndcg@20", 0.0))
    map10 = float(r.relevance.get("map@10", 0.0))
    p20 = float(r.relevance.get("p@20", 0.0))
    recall100 = float(r.relevance.get("recall@100", 0.0))
    system = _system_label(r.engine, r.mode).lower()
    run_at_unix = float(r.run_at_unix or 0.0)
    # Primary metric columns: descending (higher is better), then system asc, then run-at desc.
    return (-ndcg10, -ndcg20, -map10, -p20, -recall100, system, -run_at_unix)


def _performance_sort_key(r: ArtifactRow) -> tuple[float, float, float, float]:
    p = r.performance
    qps = float(p.get("qps", 0.0))
    p50 = float(p.get("latency_p50_ms", 0.0))
    p95 = float(p.get("latency_p95_ms", 0.0))
    p99 = float(p.get("latency_p99_ms", 0.0))
    # QPS descending, latencies ascending.
    return (-qps, p50, p95, p99)


def render_combined_report(rows: list[ArtifactRow], title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Relevance")
    lines.append("")
    rel_keys = _collect_relevance_keys(rows)
    header = "| Run ID | System | Run At | Model/Opts | Status | " + " | ".join([k.upper() for k in rel_keys]) + " |"
    sep = "|" + "---|" * (5 + len(rel_keys))
    lines.append(header)
    lines.append(sep)
    rel_rows = sorted(rows, key=_relevance_sort_key)
    for r in rel_rows:
        run_id_link = f"[{r.run_id}]({r.run_id}/summary.md)"
        vals = [f"{r.relevance.get(k, 0.0):.4f}" for k in rel_keys]
        lines.append(
            "| "
            + " | ".join(
                [
                    run_id_link,
                    _system_label(r.engine, r.mode),
                    r.run_at,
                    r.model or "n/a",
                    r.status,
                    *vals,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Query Performance")
    lines.append("")
    lines.append(
        "| Run ID | System | Run At | Model/Opts | Status | Concurrency | P50 (ms) | P95 (ms) | P99 (ms) | QPS | Error rate | Queries |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    perf_rows = sorted(rows, key=_performance_sort_key)
    for r in perf_rows:
        run_id_link = f"[{r.run_id}]({r.run_id}/summary.md)"
        p = r.performance
        lines.append(
            "| "
            + " | ".join(
                [
                    run_id_link,
                    _system_label(r.engine, r.mode),
                    r.run_at,
                    r.model or "n/a",
                    r.status,
                    str(r.concurrency if r.concurrency is not None else "-"),
                    f"{p.get('latency_p50_ms', 0.0):.2f}",
                    f"{p.get('latency_p95_ms', 0.0):.2f}",
                    f"{p.get('latency_p99_ms', 0.0):.2f}",
                    f"{p.get('qps', 0.0):.2f}",
                    f"{p.get('error_rate', 0.0):.4f}",
                    f"{p.get('queries_total', 0.0):.0f}",
                ]
            )
            + " |"
        )
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
