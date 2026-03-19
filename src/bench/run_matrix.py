from __future__ import annotations

import curses
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from .config import AppConfig, load_engine_targets
from .manticore import ManticoreClient
from .qdrant_backend import QdrantBackend
from .runbench import run_benchmark, write_run_artifacts


@dataclass
class RunVariant:
    config_path: str
    entry: str
    engine: str
    mode: str
    model: str
    default_concurrency: int
    cfg: AppConfig


@dataclass
class RunExecutionResult:
    variant: RunVariant
    run_id: str
    artifacts_dir: str


def _display_mode(mode: str) -> str:
    m = mode.lower()
    if m == "fulltext":
        return "full-text"
    if m == "vector":
        return "vector"
    return mode


def _normalize_search_options(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("option "):
        s = s[7:].strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def _manticore_vector_label(cfg: AppConfig) -> str:
    quant = (cfg.manticore.vector_quantization or "").strip().lower()
    extra: list[str] = []
    if quant:
        extra.append(f"q={quant}")
    if cfg.manticore.vector_search_oversampling is not None:
        extra.append(f"os={float(cfg.manticore.vector_search_oversampling)}")
    if cfg.manticore.vector_search_rescore is not None:
        extra.append(f"rs={1 if int(cfg.manticore.vector_search_rescore) > 0 else 0}")
    search_opts = _normalize_search_options(cfg.manticore.search_options)
    if search_opts:
        extra.append(f"search:{search_opts}")
    if extra:
        return f"{cfg.manticore.vector_model} [{','.join(extra)}]"
    return cfg.manticore.vector_model


def _qdrant_vector_label(cfg: AppConfig) -> str:
    if not bool(cfg.qdrant.dense_binary_quantization):
        return cfg.manticore.vector_model
    extra = ["bq"]
    if cfg.qdrant.dense_search_quantization_oversampling is not None:
        extra.append(f"os={float(cfg.qdrant.dense_search_quantization_oversampling)}")
    if cfg.qdrant.dense_search_quantization_rescore is not None:
        extra.append(f"rs={int(bool(cfg.qdrant.dense_search_quantization_rescore))}")
    return f"{cfg.manticore.vector_model} [{','.join(extra)}]"


def discover_config_paths() -> list[Path]:
    out: list[Path] = []
    for p in sorted(Path(".").glob("config*.yaml")):
        if p.name.endswith(".template"):
            continue
        out.append(p)
    return out


def build_variants_from_config_paths(paths: list[Path]) -> list[RunVariant]:
    variants: list[RunVariant] = []
    for p in paths:
        try:
            targets = load_engine_targets(p)
        except Exception:
            continue
        for target in targets:
            cfg = target.cfg
            engine = target.type
            modes = [target.mode_filter] if target.mode_filter in ("fulltext", "vector") else ["fulltext", "vector"]
            for mode in modes:
                if mode == "vector":
                    model_or_ranker = _manticore_vector_label(cfg) if engine == "manticore" else _qdrant_vector_label(cfg)
                elif engine == "manticore":
                    extras: list[str] = []
                    search_opts = _normalize_search_options(cfg.manticore.search_options)
                    if search_opts:
                        extras.append(f"search:{search_opts}")
                    opts = (cfg.manticore.fulltext_options or "").strip()
                    if opts:
                        extras.append(f"opts:{opts}")
                    model_or_ranker = f"[{' '.join(extras)}]" if extras else "(manticore default)"
                else:
                    model_or_ranker = "Qdrant/bm25"
                variants.append(
                    RunVariant(
                        config_path=str(p),
                        entry=target.name,
                        engine=engine,
                        mode=mode,
                        model=model_or_ranker,
                        default_concurrency=int(cfg.benchmark.concurrency),
                        cfg=cfg,
                    )
                )
    dedup: dict[tuple[str, str, str, str], RunVariant] = {}
    for v in variants:
        key = (v.entry, v.engine, v.mode, v.model)
        if key in dedup:
            continue
        dedup[key] = v

    mode_order = {"fulltext": 0, "vector": 1}
    return sorted(
        list(dedup.values()),
        key=lambda v: (
            v.entry.lower(),
            mode_order.get(v.mode, 99),
            v.model.lower(),
            v.engine.lower(),
            v.config_path,
        ),
    )


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _wrap_text_lines(text: str, width: int) -> list[str]:
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


def _draw_header(
    stdscr: Any,
    y: int,
    widths: dict[str, int],
    max_width: int,
    sort_key: str,
    sort_desc: bool,
) -> None:
    cols = [
        ("SEL", "sel"),
        ("PROFILE", "entry"),
        ("ENGINE", "engine"),
        ("MODE", "mode"),
        ("MODEL/OPTS", "model"),
    ]
    x = 0
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


def _draw_row(
    stdscr: Any,
    y: int,
    v: RunVariant,
    idx: int,
    selected: bool,
    cursor: bool,
    widths: dict[str, int],
    model_lines_max: int,
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
    entry_lines = _wrap_cell(v.entry, widths["entry"], max_lines=model_lines_max)
    model_lines = _wrap_cell(v.model, widths["model"], max_lines=model_lines_max)
    x = 0
    stdscr.addstr(y, x, _fit("[x]" if selected else "[ ]", widths["sel"]), attrs)
    x += widths["sel"] + 1
    stdscr.addstr(y, x, entry_lines[0], attrs)
    for i in range(1, model_lines_max):
        stdscr.addstr(y + i, x, entry_lines[i], attrs)
    x += widths["entry"] + 1
    stdscr.addstr(y, x, _fit(v.engine, widths["engine"]), attrs | color_engine)
    x += widths["engine"] + 1
    stdscr.addstr(y, x, _fit(_display_mode(v.mode), widths["mode"]), attrs | color_mode)
    x += widths["mode"] + 1
    stdscr.addstr(y, x, model_lines[0], attrs)
    for i in range(1, model_lines_max):
        stdscr.addstr(y + i, x, model_lines[i], attrs)


def _variant_target_name(v: RunVariant) -> str:
    if v.engine == "manticore":
        return v.cfg.manticore.fulltext_table if v.mode == "fulltext" else v.cfg.manticore.vector_table
    return v.cfg.qdrant.bm25_collection if v.mode == "fulltext" else v.cfg.qdrant.dense_collection


def _resolve_row_count(v: RunVariant) -> tuple[int | None, str | None]:
    try:
        if v.engine == "manticore":
            client = ManticoreClient(v.cfg.manticore)
            try:
                table = _variant_target_name(v)
                return client.count_rows(table), None
            finally:
                client.close()
        backend = QdrantBackend(v.cfg.qdrant)
        collection = _variant_target_name(v)
        return int(backend._count_points(collection)), None
    except Exception as exc:
        return None, str(exc)


def _details_lines(v: RunVariant, rows_line: str | None = None) -> list[str]:
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

    lines: list[str] = ["Resolved Config"]
    meta = [
        f"profile: {v.entry}",
        f"engine: {v.engine}",
        f"mode: {_display_mode(v.mode)}",
        f"target: {_variant_target_name(v)}",
    ]
    lines.extend(meta)
    flattened: list[str] = []
    visible_sections = {"dataset", "benchmark", "metrics", v.engine}
    cfg_obj = asdict(v.cfg)
    filtered_cfg = {k: val for k, val in cfg_obj.items() if k in visible_sections}
    _flatten("", filtered_cfg, flattened)
    lines.extend(flattened)
    if rows_line:
        lines.extend(["", "Live State", rows_line])
    return lines


def _draw_details_panel(
    stdscr: Any,
    v: RunVariant,
    widths: dict[str, int],
    w: int,
    h: int,
    row_count_cache: dict[tuple[str, str, str], str],
) -> None:
    table_width = sum(widths.values()) + (len(widths) - 1)
    panel_x = table_width + 3
    panel_w = w - panel_x - 1
    if panel_w < 24:
        return
    ck = (v.entry, v.engine, v.mode)
    rows_line = row_count_cache.get(ck)
    if rows_line is None:
        cnt, err = _resolve_row_count(v)
        target = _variant_target_name(v)
        if err:
            rows_line = f"rows ({target}): n/a ({err})"
        else:
            rows_line = f"rows ({target}): {int(cnt or 0):,}"
        row_count_cache[ck] = rows_line
    lines = _details_lines(v, rows_line=rows_line)
    max_rows = max(0, h - 3)
    y = 1
    for line in lines:
        if y > max_rows:
            break
        attrs = curses.A_BOLD if line in ("Resolved Config", "Live State") else curses.A_DIM
        wrapped = _wrap_text_lines(line, panel_w)
        for j, part in enumerate(wrapped):
            if y > max_rows:
                break
            part_attrs = attrs if j == 0 else curses.A_DIM
            stdscr.addstr(y, panel_x, _fit(part, panel_w), part_attrs)
            y += 1


def _select_variants_curses(variants: list[RunVariant]) -> list[RunVariant]:
    selected_idx: set[int] = set()
    current = 0
    offset = 0
    row_count_cache: dict[tuple[str, str, str], str] = {}

    def _inner(stdscr: Any) -> list[int]:
        nonlocal current, offset
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)     # engine: manticore
            curses.init_pair(2, curses.COLOR_YELLOW, -1)    # engine: qdrant
            curses.init_pair(3, curses.COLOR_BLUE, -1)      # mode: full-text
            curses.init_pair(4, curses.COLOR_WHITE, -1)     # mode: vector
        widths = {
            "sel": 3,
            "entry": 18,
            "engine": 10,
            "mode": 9,
            "model": 52,
        }
        sort_cols = ["entry", "engine", "mode", "model"]
        sort_idx = 0
        sort_key = sort_cols[sort_idx]
        sort_desc = False

        def _variant_id(v: RunVariant) -> tuple[str, str, str, str]:
            return (v.entry, v.engine, v.mode, v.model)

        def _mode_ord(mode: str) -> int:
            return 0 if mode == "fulltext" else (1 if mode == "vector" else 99)

        def _sort_value(v: RunVariant, col: str) -> Any:
            if col == "entry":
                return v.entry.lower()
            if col == "engine":
                return v.engine.lower()
            if col == "mode":
                return _mode_ord(v.mode)
            if col == "model":
                return v.model.lower()
            return v.entry.lower()

        def _apply_sort() -> None:
            nonlocal variants, selected_idx, current, offset
            selected_ids = {_variant_id(variants[i]) for i in selected_idx if 0 <= i < len(variants)}
            current_id = _variant_id(variants[current]) if 0 <= current < len(variants) else None
            variants.sort(key=lambda v: _sort_value(v, sort_key), reverse=sort_desc)
            selected_idx = {i for i, v in enumerate(variants) if _variant_id(v) in selected_ids}
            if current_id is not None:
                for i, v in enumerate(variants):
                    if _variant_id(v) == current_id:
                        current = i
                        break
            offset = 0

        _apply_sort()
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            row_h = _dynamic_model_lines(len(variants), h)
            stdscr.addstr(0, 0, _fit("↑/↓ move  ←/→ sort col  SPACE toggle  ENTER run selected  ESC/q cancel", max(1, w - 1)), curses.A_BOLD)
            _draw_header(stdscr, 1, widths=widths, max_width=max(1, w - 1), sort_key=sort_key, sort_desc=sort_desc)
            visible = max(1, (h - 4) // row_h)
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
                    y=2 + i * row_h,
                    v=variants[ridx],
                    idx=ridx + 1,
                    selected=(ridx in selected_idx),
                    cursor=(ridx == current),
                    widths=widths,
                    model_lines_max=row_h,
                )
            if variants:
                _draw_details_panel(
                    stdscr,
                    variants[current],
                    widths=widths,
                    w=w,
                    h=h,
                    row_count_cache=row_count_cache,
                )
            stdscr.addstr(h - 1, 0, _fit(f"Selected: {len(selected_idx)} / {len(variants)}", max(1, w - 1)), curses.A_DIM)
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


def select_variants_interactive(variants: list[RunVariant]) -> list[RunVariant]:
    if not variants:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _select_variants_curses(variants)
        except Exception:
            pass
    # Text fallback
    print("Available runs:")
    for i, v in enumerate(variants, start=1):
        print(
            f"{i:>2}. {v.entry}\t{v.engine}\t{_display_mode(v.mode)}\t"
            f"model={v.model}"
        )
    raw = input("Select variant numbers (e.g. 1,3-5): ").strip()
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


def run_selected_variants(
    variants: list[RunVariant],
    concurrency_override: int | None = None,
) -> list[RunExecutionResult]:
    out: list[RunExecutionResult] = []
    use_color = sys.stdout.isatty()
    c_label = "\033[1;35m" if use_color else ""
    c_cmd = "\033[36m" if use_color else ""
    c_reset = "\033[0m" if use_color else ""
    for v in variants:
        conc_part = f" --concurrency {concurrency_override}" if concurrency_override is not None else ""
        cmd = (
            f"bench --config {v.config_path} run --engine {v.engine} "
            f"--mode {v.mode} --entry {v.entry}{conc_part}"
        )
        print(
            f"{c_label}Replay command:{c_reset} {c_cmd}{cmd}{c_reset}",
            flush=True,
        )
        print(
            f"Running: profile={v.entry} engine={v.engine} mode={_display_mode(v.mode)} model={v.model}",
            flush=True,
        )

        def _progress(update: dict[str, Any]) -> None:
            completed = int(update.get("completed", 0))
            total = int(update.get("total", 0))
            success = int(update.get("success", 0))
            errors = int(update.get("errors", 0))
            elapsed = float(update.get("elapsed_seconds", 0.0))
            pct = (completed / total * 100.0) if total else 0.0
            print(
                f"  progress {completed}/{total} ({pct:5.1f}%) ok={success} err={errors} elapsed={elapsed:6.1f}s",
                flush=True,
            )

        run_id, metrics, per_query = run_benchmark(
            cfg=v.cfg,
            engine=v.engine,
            mode=v.mode,
            profile_name=v.entry,
            concurrency_override=concurrency_override,
            progress_callback=_progress,
        )
        artifacts_dir = write_run_artifacts(run_id, metrics, per_query)
        out.append(RunExecutionResult(variant=v, run_id=run_id, artifacts_dir=str(artifacts_dir)))
    return out
