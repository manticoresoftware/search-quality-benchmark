from __future__ import annotations

import argparse
import curses
from pathlib import Path
import sys
from time import perf_counter

from .cache_populate import populate_cache_from_manticore
from .config import AppConfig, load_config, load_engine_targets
from .index_matrix import build_index_variants_from_config_paths, discover_config_paths, select_index_variants_interactive
from .indexing import run_index
from .manticore import ManticoreClient
from .io_utils import generate_run_id
from .prepare import prepare_dataset
from .run_matrix import build_variants_from_config_paths, run_selected_variants, select_variants_interactive
from .runbench import run_benchmark, write_run_artifacts
from .summarize import generate_combined_report


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _select_simple_option_interactive(title: str, options: list[tuple[str, str]]) -> str | None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            def _inner(stdscr: object) -> str | None:
                current = 0
                curses.curs_set(0)
                stdscr.keypad(True)  # type: ignore[attr-defined]
                while True:
                    stdscr.erase()  # type: ignore[attr-defined]
                    h, w = stdscr.getmaxyx()  # type: ignore[attr-defined]
                    stdscr.addstr(0, 0, _fit(f"{title}: ↑/↓ move  ENTER select  ESC/q cancel", max(1, w - 1)), curses.A_BOLD)  # type: ignore[attr-defined]
                    for i, (_, label) in enumerate(options):
                        attrs = curses.A_REVERSE if i == current else curses.A_NORMAL
                        stdscr.addstr(2 + i, 2, _fit(label, max(1, w - 3)), attrs)  # type: ignore[attr-defined]
                    stdscr.refresh()  # type: ignore[attr-defined]
                    ch = stdscr.getch()  # type: ignore[attr-defined]
                    if ch in (curses.KEY_UP, ord("k")):
                        current = max(0, current - 1)
                    elif ch in (curses.KEY_DOWN, ord("j")):
                        current = min(len(options) - 1, current + 1)
                    elif ch in (10, 13, curses.KEY_ENTER):
                        return options[current][0]
                    elif ch == 27 or ch in (ord("q"),):
                        return None
            return curses.wrapper(_inner)
        except Exception:
            pass

    print(title)
    for i, (_, label) in enumerate(options, start=1):
        print(f"{i}. {label}")
    raw = input("Enter number (or empty to cancel): ").strip()
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 1 <= idx <= len(options):
        return options[idx - 1][0]
    return None


def _select_top_command_interactive() -> str | None:
    options: list[tuple[str, str, str]] = [
        ("prepare", "Prepare", "Download and prepare dataset files"),
        ("index", "Index", "Create/recreate benchmark indexes in Manticore/Qdrant"),
        ("run", "Run", "Execute benchmark queries and collect metrics"),
        ("report", "Report", "Pick artifacts and generate combined markdown report"),
    ]
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            def _inner(stdscr: object) -> str | None:
                current = 0
                curses.curs_set(0)
                stdscr.keypad(True)  # type: ignore[attr-defined]
                if curses.has_colors():
                    curses.start_color()
                    curses.use_default_colors()
                    curses.init_pair(1, curses.COLOR_WHITE, -1)   # prepare
                    curses.init_pair(2, curses.COLOR_GREEN, -1)   # index
                    curses.init_pair(3, curses.COLOR_CYAN, -1)    # run
                    curses.init_pair(4, curses.COLOR_YELLOW, -1)  # report
                while True:
                    stdscr.erase()  # type: ignore[attr-defined]
                    h, w = stdscr.getmaxyx()  # type: ignore[attr-defined]
                    stdscr.addstr(0, 0, _fit("Select command: ↑/↓ move  ENTER run  ESC/q cancel", max(1, w - 1)), curses.A_BOLD)  # type: ignore[attr-defined]
                    for i, (name, label, desc) in enumerate(options):
                        attrs = curses.A_REVERSE if i == current else curses.A_NORMAL
                        if curses.has_colors():
                            if name == "prepare":
                                attrs |= curses.color_pair(1)
                            elif name == "index":
                                attrs |= curses.color_pair(2)
                            elif name == "run":
                                attrs |= curses.color_pair(3)
                            elif name == "report":
                                attrs |= curses.color_pair(4)
                        line = f"{label:<10} {desc}"
                        stdscr.addstr(2 + i, 2, _fit(line, max(1, w - 3)), attrs)  # type: ignore[attr-defined]
                    stdscr.refresh()  # type: ignore[attr-defined]
                    ch = stdscr.getch()  # type: ignore[attr-defined]
                    if ch in (curses.KEY_UP, ord("k")):
                        current = max(0, current - 1)
                    elif ch in (curses.KEY_DOWN, ord("j")):
                        current = min(len(options) - 1, current + 1)
                    elif ch in (10, 13, curses.KEY_ENTER):
                        return options[current][0]
                    elif ch == 27 or ch in (ord("q"),):
                        return None
            return curses.wrapper(_inner)
        except Exception:
            pass

    print("Select command:")
    for i, (_, label, desc) in enumerate(options, start=1):
        print(f"{i}. {label:<10} {desc}")
    raw = input("Enter number (or empty to cancel): ").strip()
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 1 <= idx <= len(options):
        return options[idx - 1][0]
    return None


def _confirm_manticore_vector_reindex_if_needed(
    cfg: AppConfig, *, needs_vector_reindex: bool, auto_yes: bool = False
) -> bool:
    if not needs_vector_reindex:
        return True

    client = ManticoreClient(cfg.manticore)
    try:
        vec_exists = client.table_exists(cfg.manticore.vector_table)
    finally:
        client.close()

    if not vec_exists:
        return True
    if auto_yes:
        return True

    message = (
        "\nManticore vector benchmark table already exists:\n"
        f"- vector: {cfg.manticore.vector_table}\n\n"
        "Indexing in vector mode will drop and recreate this table.\n"
        "Cached vectors are reused; cache misses are auto-embedded by Manticore and then cached.\n"
        "Type 'yes' to continue: "
    )
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Refusing to recreate existing Manticore vector benchmark table in non-interactive mode. "
            "Re-run with `bench index --yes ...`."
        )
    answer = input(message).strip().lower()
    return answer == "yes"


def _confirm_manticore_vector_reindex_for_configs(
    cfgs: list[AppConfig], *, auto_yes: bool = False
) -> bool:
    existing: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for cfg in cfgs:
        key = (
            cfg.manticore.host,
            int(cfg.manticore.sql_port),
            cfg.manticore.user,
            cfg.manticore.password,
            cfg.manticore.vector_table,
        )
        if key in seen:
            continue
        seen.add(key)
        client = ManticoreClient(cfg.manticore)
        try:
            if client.table_exists(cfg.manticore.vector_table):
                existing.append((cfg.manticore.host, int(cfg.manticore.sql_port), cfg.manticore.vector_table))
        finally:
            client.close()

    if not existing:
        return True
    if auto_yes:
        return True
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Refusing to recreate existing Manticore vector benchmark table(s) in non-interactive mode. "
            "Re-run with `bench index --yes ...`."
        )
    tables_text = "\n".join(f"- {h}:{p} table={t}" for h, p, t in existing)
    message = (
        "\nManticore vector benchmark table(s) already exist:\n"
        f"{tables_text}\n\n"
        "Indexing in vector mode will drop and recreate these table(s).\n"
        "Cached vectors are reused; cache misses are auto-embedded by Manticore and then cached.\n"
        "Type 'yes' to continue: "
    )
    answer = input(message).strip().lower()
    return answer == "yes"


def cmd_prepare(cfg: AppConfig, args: argparse.Namespace) -> int:
    result = prepare_dataset(cfg)
    print(f"Prepared dataset written to {result.docs_path.parent}")
    return 0


def _resolve_entry_cfg(
    *,
    config_path: str,
    engine: str,
    entry: str | None,
    fallback_cfg: AppConfig,
) -> tuple[AppConfig, str, str | None]:
    engine_targets = [t for t in load_engine_targets(config_path) if t.type == engine]
    if entry:
        engine_targets = [t for t in engine_targets if t.name == entry]
        if not engine_targets:
            raise ValueError(
                f"No entry named '{entry}' for engine '{engine}' in {config_path}."
            )
    if engine_targets:
        t = engine_targets[0]
        return t.cfg, t.name, t.mode_filter
    return fallback_cfg, engine, None


def cmd_index(cfg: AppConfig, args: argparse.Namespace) -> int:
    is_tty = sys.stdout.isatty()
    last_by_key: dict[tuple[str, str], float] = {}
    started_by_key: dict[tuple[str, str], float] = {}

    def _progress(update: dict[str, object]) -> None:
        engine = str(update.get("engine", ""))
        mode = str(update.get("mode", ""))
        stage = str(update.get("stage", ""))
        key = (engine, mode, stage)
        now = perf_counter()
        if key not in started_by_key:
            started_by_key[key] = now
            last_by_key[key] = 0.0
        elapsed = now - started_by_key[key]
        last = last_by_key.get(key, 0.0)
        processed = int(update.get("processed", 0))
        total_obj = update.get("total")
        total = int(total_obj) if total_obj is not None else 0
        inserted = int(update.get("inserted", 0))
        retries = int(update.get("retried_batches", 0))
        failed = int(update.get("failed_batches", 0))
        last_error = str(update.get("last_error", "") or "").strip().replace("\n", " ")
        if len(last_error) > 140:
            last_error = last_error[:139] + "…"
        is_final = total > 0 and processed >= total
        if (now - last) < 0.5 and not is_final:
            return
        pct = (processed / total * 100.0) if total > 0 else 0.0
        eta = ""
        if total > 0 and processed > 0 and processed < total:
            rate = processed / elapsed if elapsed > 0 else 0.0
            if rate > 0:
                remaining = (total - processed) / rate
                eta = f" eta={remaining:6.1f}s"
        stage_part = f":{stage}" if stage else ""
        line = (
            f"Index progress [{engine}:{mode}{stage_part}] {processed}/{total if total > 0 else '?'} "
            f"({pct:5.1f}%) inserted={inserted} retries={retries} failed={failed} elapsed={elapsed:6.1f}s{eta}"
        )
        if last_error and (retries > 0 or failed > 0):
            line += f" reason={last_error}"
        if is_tty:
            end = "\n" if is_final else ""
            print(f"\r{line}", end=end, flush=True)
        else:
            print(line, flush=True)
        last_by_key[key] = now

    try:
        if args.mode is None and args.engine is None:
            variants = build_index_variants_from_config_paths(discover_config_paths())
            selected = select_index_variants_interactive(variants)
            if not selected:
                print("Index canceled: no runs selected.")
                return 0
            print("Replay commands:")
            for v in selected:
                print(
                    f"- bench --config {v.config_path} index --engine {v.engine} "
                    f"--mode {v.mode} --entry {v.entry}"
                )
            manticore_vector_cfgs = [v.cfg for v in selected if v.engine == "manticore" and v.mode == "vector"]
            if not _confirm_manticore_vector_reindex_for_configs(
                manticore_vector_cfgs, auto_yes=bool(args.yes)
            ):
                print("Index canceled: Manticore vector reindex not confirmed.")
                return 0
            for v in selected:
                started_by_key.clear()
                last_by_key.clear()
                run_id = generate_run_id()
                run_index(v.cfg, engine=v.engine, mode=v.mode, progress_callback=_progress)
                model_note = f" model={v.model}" if v.mode == "vector" else ""
                print(f"Indexed profile={v.entry} engine={v.engine} mode={v.mode}{model_note} run_id={run_id}")
            print(f"Index batch completed: {len(selected)} runs")
            return 0

        if args.mode is None:
            raise ValueError("--mode is required unless running interactive `bench index` with no index parameters.")
        engine = args.engine or "manticore"
        cfg_for_engine, entry_name, entry_mode_filter = _resolve_entry_cfg(
            config_path=args.config,
            engine=engine,
            entry=args.entry,
            fallback_cfg=cfg,
        )
        if entry_mode_filter in ("fulltext", "vector") and args.mode != entry_mode_filter:
            raise ValueError(
                f"Entry '{entry_name}' is restricted to mode '{entry_mode_filter}'. "
                f"Requested mode '{args.mode}' is not allowed."
            )
        if engine == "manticore":
            needs_vector_reindex = args.mode == "vector"
            if not _confirm_manticore_vector_reindex_if_needed(
                cfg_for_engine, needs_vector_reindex=needs_vector_reindex, auto_yes=bool(args.yes)
            ):
                print("Index canceled: Manticore vector reindex not confirmed.")
                return 0
        started_by_key.clear()
        last_by_key.clear()
        run_id = generate_run_id()
        run_index(cfg_for_engine, engine=engine, mode=args.mode, progress_callback=_progress)
        model_note = f" model={cfg_for_engine.manticore.vector_model}" if args.mode == "vector" else ""
        print(f"Indexed profile={entry_name} engine={engine} mode={args.mode}{model_note}")
        return 0
    except KeyboardInterrupt:
        if is_tty:
            print("")
        print("Index canceled by user.")
        return 130


def cmd_run(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.mode is None and args.engine is None:
        variants = build_variants_from_config_paths([Path(args.config)])
        selected = select_variants_interactive(variants)
        if not selected:
            print("Run canceled: no runs selected.")
            return 0
        try:
            results = run_selected_variants(selected, concurrency_override=args.concurrency)
        except Exception as exc:
            print(f"FATAL: {exc}")
            return 1
        print(f"Run batch completed: {len(results)} runs")
        for r in results:
            v = r.variant
            print(
                f"- profile={v.entry} engine={v.engine} mode={v.mode} model={v.model} "
                f"run_id={r.run_id} artifacts={r.artifacts_dir}"
            )
        return 0

    engine = args.engine or "manticore"
    if args.mode is None:
        raise ValueError("--mode is required unless running interactive `bench run` with no run parameters.")
    cfg_for_engine, entry_name, entry_mode_filter = _resolve_entry_cfg(
        config_path=args.config,
        engine=engine,
        entry=args.entry,
        fallback_cfg=cfg,
    )
    if entry_mode_filter in ("fulltext", "vector") and args.mode != entry_mode_filter:
        raise ValueError(
            f"Entry '{entry_name}' is restricted to mode '{entry_mode_filter}'. "
            f"Requested mode '{args.mode}' is not allowed."
        )
    is_tty = sys.stdout.isatty()

    def _progress(update: dict[str, object]) -> None:
        completed = int(update.get("completed", 0))
        total = int(update.get("total", 0))
        success = int(update.get("success", 0))
        errors = int(update.get("errors", 0))
        elapsed = float(update.get("elapsed_seconds", 0.0))
        pct = (completed / total * 100.0) if total else 0.0
        line = (
            f"Progress: {completed}/{total} ({pct:5.1f}%) "
            f"ok={success} err={errors} elapsed={elapsed:6.1f}s"
        )
        if is_tty:
            end = "\n" if completed >= total and total > 0 else ""
            print(f"\r{line}", end=end, flush=True)
        else:
            print(line, flush=True)

    try:
        run_id, metrics, per_query_rows = run_benchmark(
            cfg_for_engine,
            engine=engine,
            mode=args.mode,
            profile_name=entry_name,
            concurrency_override=args.concurrency,
            progress_callback=_progress,
        )
    except Exception as exc:
        print(f"FATAL: {exc}")
        return 1
    out_dir = write_run_artifacts(run_id, metrics, per_query_rows)
    print(f"Run completed. profile={entry_name} Artifacts: {out_dir}")
    return 0


def cmd_report(_: AppConfig, args: argparse.Namespace) -> int:
    run_ids = [x.strip() for x in (args.run_ids or "").split(",") if x.strip()] if args.run_ids else None
    interactive = False if run_ids else (not bool(args.no_interactive))
    try:
        out = generate_combined_report(
            run_ids=run_ids,
            interactive=interactive,
            output_path=args.output,
            title=args.title,
        )
    except RuntimeError as exc:
        if str(exc) == "No artifacts selected.":
            print("Report canceled: no artifacts selected.")
            return 0
        raise
    print(f"Report generated: {out}")
    return 0


def cmd_cache_populate(cfg: AppConfig, args: argparse.Namespace) -> int:
    result = populate_cache_from_manticore(
        cfg,
        source_table=args.table,
        model_name=args.model,
        similarity=args.similarity,
        vector_from=args.vector_from,
    )
    print(
        "Cache populated: "
        f"path={result.cache_path} table={result.source_table} "
        f"rows={result.seen_rows} ok={result.cached_ok} "
        f"null_from_table={result.cached_null_from_table} "
        f"null_from_missing_docs={result.cached_null_from_missing_docs}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", add_help=False)
    parser.add_argument("--config", default="config.yaml")

    sub = parser.add_subparsers(dest="command", required=False)

    p_prepare = sub.add_parser("prepare", add_help=False)

    p_index = sub.add_parser("index", add_help=False)
    p_index.add_argument("--engine", default=None, choices=["manticore", "qdrant"])
    p_index.add_argument("--mode", default=None, choices=["fulltext", "vector"])
    p_index.add_argument("--entry", default=None, help="Config entry name to run (for deterministic replay)")
    p_index.add_argument(
        "--yes",
        action="store_true",
        help="Skip Manticore vector table recreation confirmation",
    )

    p_run = sub.add_parser("run", add_help=False)
    p_run.add_argument("--engine", default=None, choices=["manticore", "qdrant"])
    p_run.add_argument("--mode", default=None, choices=["fulltext", "vector", "both"])
    p_run.add_argument("--entry", default=None, help="Config entry name to run (for deterministic replay)")
    p_run.add_argument("--concurrency", type=int, default=None)

    p_report = sub.add_parser("report", add_help=False)
    p_report.add_argument("--run-ids", default=None, help="Comma-separated run ids to include")
    p_report.add_argument("--no-interactive", action="store_true", help="Disable interactive artifact selection")
    p_report.add_argument("--output", default=None, help="Output markdown path")
    p_report.add_argument("--title", default="Benchmark Report")

    p_cache = sub.add_parser("cache-populate", add_help=False)
    p_cache.add_argument("--table", default=None, help="Source Manticore vector table")
    p_cache.add_argument("--model", default=None, help="Model name for cache keys")
    p_cache.add_argument("--similarity", default=None, help="Similarity for cache keys (e.g. cosine)")
    p_cache.add_argument("--vector-from", default=None, help="Comma-separated source text fields for cache keys")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        picked = _select_top_command_interactive()
        if picked is None:
            raise SystemExit(0)
        args = parser.parse_args(["--config", args.config, picked])
    cfg = load_config(args.config)
    if args.command == "prepare":
        raise SystemExit(cmd_prepare(cfg, args))
    if args.command == "index":
        raise SystemExit(cmd_index(cfg, args))
    if args.command == "run":
        raise SystemExit(cmd_run(cfg, args))
    if args.command == "report":
        raise SystemExit(cmd_report(cfg, args))
    if args.command == "cache-populate":
        raise SystemExit(cmd_cache_populate(cfg, args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
