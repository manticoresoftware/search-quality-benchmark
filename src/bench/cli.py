from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass
from hashlib import md5
import json
from pathlib import Path
import sys
from time import perf_counter

from .config import AppConfig, display_engine_for_profile, is_manticore_like_type, load_config, load_engine_targets
from .configmap import generate_configmap
from .discovery import discover_config_paths, resolve_config_path
from .engines import cmd_engines
from .embedding_client import resolved_embedding_model
from .index_matrix import build_index_variants_from_config_paths, select_index_variants_interactive
from .indexing import run_index
from .manticore import ManticoreClient
from .io_utils import generate_run_id
from .prepare import prepare_dataset, reset_embeddings
from .prepare_matrix import (
    PrepareSelection,
    PrepareVariant,
    build_prepare_variants_from_config_paths,
    select_prepare_variants_interactive,
)
from .run_matrix import (
    build_variants_from_config_paths,
    filter_runnable_variants,
    run_selected_variants,
    select_variants_interactive,
)
from .runbench import run_benchmark, write_run_artifacts
from .summarize import combined_html_report_path, generate_combined_report
from .vector_cache import VectorCache, cache_key_from_doc, parse_vector_from_fields


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
        ("engines", "Engines", "Start/stop Docker-backed engines and watch their status"),
        ("prepare", "Prepare", "Download dataset files and build embeddings"),
        ("index", "Index", "Create/recreate benchmark indexes in supported engines"),
        ("run", "Run", "Execute benchmark queries and collect metrics"),
        ("report", "Report", "Pick artifacts and generate combined Markdown and HTML reports"),
        ("configmap", "Draw configmap", "Generate a Mermaid graph of profile dependencies"),
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
                    curses.init_pair(1, curses.COLOR_RED, -1)     # engines
                    curses.init_pair(2, curses.COLOR_WHITE, -1)   # prepare
                    curses.init_pair(3, curses.COLOR_GREEN, -1)   # index
                    curses.init_pair(4, curses.COLOR_CYAN, -1)    # run
                    curses.init_pair(5, curses.COLOR_YELLOW, -1)  # report
                    curses.init_pair(6, curses.COLOR_MAGENTA, -1) # configmap
                while True:
                    stdscr.erase()  # type: ignore[attr-defined]
                    h, w = stdscr.getmaxyx()  # type: ignore[attr-defined]
                    stdscr.addstr(0, 0, _fit("Select command: ↑/↓ move  ENTER run  ESC/q cancel", max(1, w - 1)), curses.A_BOLD)  # type: ignore[attr-defined]
                    for i, (name, label, desc) in enumerate(options):
                        attrs = curses.A_REVERSE if i == current else curses.A_NORMAL
                        if curses.has_colors():
                            if name == "engines":
                                attrs |= curses.color_pair(1)
                            elif name == "prepare":
                                attrs |= curses.color_pair(2)
                            elif name == "index":
                                attrs |= curses.color_pair(3)
                            elif name == "run":
                                attrs |= curses.color_pair(4)
                            elif name == "report":
                                attrs |= curses.color_pair(5)
                            elif name == "configmap":
                                attrs |= curses.color_pair(6)
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
        "Cached vectors are reused; cache misses now fail and must be built during `bench prepare`.\n"
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
        "Cached vectors are reused; cache misses now fail and must be built during `bench prepare`.\n"
        "Type 'yes' to continue: "
    )
    answer = input(message).strip().lower()
    return answer == "yes"


@dataclass
class CacheSampleCheck:
    sample_size: int
    ok_count: int
    null_count: int
    missing_count: int
    missing_keys: list[str]


def _sample_docs_for_cache_check(docs: list[dict[str, object]], *, ratio: int = 100) -> list[dict[str, object]]:
    total = len(docs)
    if total <= ratio:
        return docs
    out: list[dict[str, object]] = []
    for i, doc in enumerate(docs):
        raw = str(doc.get("doc_id") or doc.get("id") or i)
        bucket = int(md5(raw.encode("utf-8")).hexdigest()[:8], 16) % ratio
        if bucket == 0:
            out.append(doc)
    if not out:
        return docs[: min(total, max(1, total // ratio))]
    return out


def _sample_docs_file_for_cache_check(
    docs_path: Path,
    *,
    ratio: int = 100,
    cap: int = 2000,
) -> list[dict[str, object]]:
    sampled: list[dict[str, object]] = []
    total = 0
    with docs_path.open("r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                doc = json.loads(line)
            except Exception:
                continue
            raw = str(doc.get("doc_id") or doc.get("id") or i)
            bucket = int(md5(raw.encode("utf-8")).hexdigest()[:8], 16) % ratio
            if bucket != 0:
                continue
            sampled.append(doc)
            if len(sampled) >= cap:
                break
    if sampled:
        return sampled

    fallback: list[dict[str, object]] = []
    fallback_cap = min(max(1, total // ratio if total > 0 else 1), cap)
    with docs_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue
            fallback.append(doc)
            if len(fallback) >= fallback_cap:
                break
    return fallback


def _check_vector_cache_sample(cfg: AppConfig) -> CacheSampleCheck | None:
    docs_path = Path(cfg.dataset.prepared_dir) / "docs.jsonl"
    if not docs_path.exists():
        return None
    sampled = _sample_docs_file_for_cache_check(docs_path)
    fields = parse_vector_from_fields(cfg.manticore.vector_from)
    cache = VectorCache(cfg.manticore.vector_cache_path)
    try:
        status_map = cache.get_status_many(
            [
                cache_key_from_doc(
                    doc,
                    fields=fields,
                    model=resolved_embedding_model(cfg),
                    similarity=cfg.manticore.vector_similarity,
                )
                for doc in sampled
            ]
        )
    finally:
        cache.close()
    ok_count = 0
    null_count = 0
    missing_count = 0
    missing_keys: list[str] = []
    for doc in sampled:
        key = cache_key_from_doc(
            doc,
            fields=fields,
            model=resolved_embedding_model(cfg),
            similarity=cfg.manticore.vector_similarity,
        )
        status = status_map.get(key, "missing")
        if status == "ok":
            ok_count += 1
        elif status == "null":
            null_count += 1
        else:
            missing_count += 1
            if len(missing_keys) < 5:
                missing_keys.append(key)
    return CacheSampleCheck(
        sample_size=len(sampled),
        ok_count=ok_count,
        null_count=null_count,
        missing_count=missing_count,
        missing_keys=missing_keys,
    )


def _confirm_vector_cache_coverage(
    *,
    cfg: AppConfig,
    engine: str,
    entry_name: str,
    auto_yes: bool = False,
) -> bool:
    check = _check_vector_cache_sample(cfg)
    if check is None:
        return True
    if check.missing_count <= 0 and check.null_count <= 0:
        return True
    if auto_yes:
        return True
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Vector cache preflight found missing entries in non-interactive mode. "
            "Re-run with `bench index --yes ...` to continue anyway."
        )
    sample_pct = (100.0 * check.missing_count / check.sample_size) if check.sample_size > 0 else 0.0
    if engine == "qdrant":
        engine_note = "Qdrant vector indexing is likely to fail if cache misses remain."
    elif engine == "meilisearch":
        engine_note = "Meilisearch vector indexing is likely to fail if cache misses remain."
    elif engine == "opensearch":
        engine_note = "OpenSearch vector indexing is likely to fail if cache misses remain."
    elif engine == "elasticsearch":
        engine_note = "Elasticsearch vector indexing is likely to fail if cache misses remain."
    elif engine == "typesense":
        engine_note = "Typesense vector indexing is likely to fail if cache misses remain."
    elif engine == "milvus":
        engine_note = "Milvus vector indexing is likely to fail if cache misses remain."
    elif engine == "mariadb":
        engine_note = "MariaDB vector indexing is likely to fail if cache misses remain."
    elif engine == "weaviate":
        engine_note = "Weaviate vector indexing is likely to fail if cache misses remain."
    elif engine == "vespa":
        engine_note = "Vespa vector indexing is likely to fail if cache misses remain."
    else:
        engine_note = "Manticore vector indexing will fail if cache misses remain. Build embeddings during `bench prepare` first."
    sample_text = ", ".join(check.missing_keys) if check.missing_keys else "-"
    prompt = (
        f"\nVector cache preflight for entry '{entry_name}':\n"
        f"- cache: {cfg.manticore.vector_cache_path}\n"
        f"- sample size: {check.sample_size}\n"
        f"- cache hits: {check.ok_count}\n"
        f"- cache nulls: {check.null_count}\n"
        f"- cache misses: {check.missing_count} ({sample_pct:.1f}%)\n"
        f"- sample missing keys: {sample_text}\n\n"
        f"{engine_note}\n"
        "Type 'yes' to continue anyway: "
    )
    return input(prompt).strip().lower() == "yes"


def cmd_prepare(cfg: AppConfig, args: argparse.Namespace) -> int:
    try:
        selected: list[PrepareVariant] = []
        action = "reset" if bool(getattr(args, "reset_embeddings", False)) else "prepare"
        raw_sources: list[str] = []
        for value in list(getattr(args, "source", []) or []):
            raw_sources.extend([x.strip() for x in str(value).split(",") if x.strip()])
        source_filter = {s.lower() for s in raw_sources}

        raw_entries: list[str] = []
        for value in list(getattr(args, "entry", []) or []):
            raw_entries.extend([x.strip() for x in str(value).split(",") if x.strip()])
        entry_filter = {s for s in raw_entries}

        variants = build_prepare_variants_from_config_paths(discover_config_paths())
        if source_filter or entry_filter:
            for v in variants:
                if source_filter and v.source.lower() not in source_filter:
                    continue
                if entry_filter:
                    entry_tokens = {x.strip() for x in v.entry.split(",")}
                    if not any(e in entry_tokens for e in entry_filter):
                        continue
                selected.append(v)
            if not selected:
                raise ValueError(
                    "No matching dataset sources/entries found. "
                    "Check `--source` / `--entry` values against config entries."
                )
        elif sys.stdin.isatty() and sys.stdout.isatty():
            selection = select_prepare_variants_interactive(variants)
            selected = selection.variants
            action = selection.action
            if not selected:
                print("Prepare canceled: no sources selected.")
                return 0
        else:
            selected = [
                PrepareVariant(
                    config_path=args.config,
                    entry="(default)",
                    source=cfg.dataset.source,
                    prepared_dir=cfg.dataset.prepared_dir,
                    cfg=cfg,
                )
            ]

        if action == "cancel":
            print("Prepare canceled.")
            return 0

        if action == "reset":
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "Refusing to reset embeddings without confirmation in non-interactive mode. "
                    "Re-run interactively."
                )
            print("About to reset embeddings for:")
            for v in selected:
                print(f"- source={v.source} output={v.cfg.dataset.prepared_dir}")
            answer = input("Type 'yes' to continue: ").strip().lower()
            if answer != "yes":
                print("Reset canceled.")
                return 0

        replay: list[str] = []
        for v in selected:
            cmd = f"bench --config {args.config} prepare --source {v.source}"
            if action == "reset":
                cmd += " --reset-embeddings"
            replay.append(cmd)
        print("Replay commands:")
        for cmd in replay:
            print(cmd)

        for v in selected:
            if action == "reset":
                print(f"Resetting embeddings: source={v.source} output={v.cfg.dataset.prepared_dir}")
                stats = reset_embeddings(v.cfg)
                print(
                    f"Reset embeddings completed: deleted={stats.deleted} "
                    f"docs={stats.doc_total} queries={stats.query_total} output={v.cfg.dataset.prepared_dir}"
                )
            else:
                print(f"Preparing: source={v.source} output={v.cfg.dataset.prepared_dir}")
                started = perf_counter()
                is_tty = sys.stdout.isatty()
                def _progress(update: dict[str, object]) -> None:
                    stage = str(update.get("stage", ""))
                    processed = int(update.get("processed", 0))
                    total = int(update.get("total", 0) or 0)
                    embedded_ok = int(update.get("embedded_ok", 0) or 0)
                    cost = float(update.get("estimated_cost_usd", 0.0) or 0.0)
                    tokens = int(update.get("input_tokens", 0) or 0)
                    pct = (100.0 * processed / total) if total > 0 else 0.0
                    elapsed = perf_counter() - started
                    line = (
                        f"Prepare progress [{stage}] {processed}/{total} ({pct:5.1f}%) "
                        f"embeddings_ok={embedded_ok} tokens={tokens} cost=${cost:.4f} elapsed={elapsed:6.1f}s"
                    )
                    if is_tty:
                        end = "\n" if total > 0 and processed >= total and stage in ("query_embed", "query_cache_scan") else ""
                        print(f"\r{line}", end=end, flush=True)
                    else:
                        print(line, flush=True)
                try:
                    result = prepare_dataset(v.cfg, progress_callback=_progress)
                except KeyboardInterrupt:
                    if is_tty:
                        print()
                    print(
                        f"Prepare interrupted: source={v.source} output={v.cfg.dataset.prepared_dir} "
                        "(partial cache and manifest state preserved)"
                    )
                    return 130
                except Exception:
                    if is_tty:
                        print()
                    raise
                if is_tty:
                    print()
                print(f"Prepared dataset written to {result.docs_path.parent}")
        return 0
    except KeyboardInterrupt:
        print("Prepare interrupted.")
        return 130
    except Exception as exc:
        print(f"FATAL: {exc}")
        return 1


def _resolve_entry_cfg(
    *,
    config_path: str,
    engine: str,
    entry: str | None,
    fallback_cfg: AppConfig,
) -> tuple[AppConfig, str, str | None]:
    engine_targets = [
        t
        for t in load_engine_targets(config_path)
        if display_engine_for_profile(t.type, t.name) == engine
    ]
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
        prepare_seconds_obj = update.get("prepare_seconds")
        send_seconds_obj = update.get("send_seconds")
        prepare_seconds = float(prepare_seconds_obj) if prepare_seconds_obj is not None else None
        send_seconds = float(send_seconds_obj) if send_seconds_obj is not None else None
        progress_percent_obj = update.get("progress_percent")
        progress_percent = (
            float(progress_percent_obj) if progress_percent_obj is not None else None
        )
        last_error = str(update.get("last_error", "") or "").strip().replace("\n", " ")
        if len(last_error) > 140:
            last_error = last_error[:139] + "…"
        error_reasons_raw = update.get("error_reasons", {}) or {}
        error_reasons: dict[str, int] = {}
        if isinstance(error_reasons_raw, dict):
            for reason, count in error_reasons_raw.items():
                try:
                    error_reasons[str(reason)] = int(count)
                except Exception:
                    continue
        is_final = (
            progress_percent >= 100.0
            if progress_percent is not None
            else total > 0 and processed >= total
        )
        if (now - last) < 0.5 and not is_final:
            return
        pct = (
            progress_percent
            if progress_percent is not None
            else (processed / total * 100.0) if total > 0 else 0.0
        )
        eta = ""
        if progress_percent is None and total > 0 and processed > 0 and processed < total:
            rate = processed / elapsed if elapsed > 0 else 0.0
            if rate > 0:
                remaining = (total - processed) / rate
                eta = f" eta={remaining:6.1f}s"
        stage_part = f":{stage}" if stage else ""
        if progress_percent is not None:
            progress_text = f"{pct:5.1f}%"
        else:
            progress_text = f"{processed}/{total if total > 0 else '?'} ({pct:5.1f}%)"
        line = (
            f"Index progress [{engine}:{mode}{stage_part}] {progress_text} "
            f"inserted={inserted} retries={retries} failed={failed} elapsed={elapsed:6.1f}s{eta}"
        )
        if prepare_seconds is not None or send_seconds is not None:
            parts: list[str] = []
            if prepare_seconds is not None:
                parts.append(f"prepare={prepare_seconds:6.1f}s")
            if send_seconds is not None:
                parts.append(f"send={send_seconds:6.1f}s")
            line += " " + " ".join(parts)
        if error_reasons and (retries > 0 or failed > 0):
            top = sorted(error_reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[:2]
            rendered: list[str] = []
            for reason, count in top:
                short = reason.strip().replace("\n", " ")
                if len(short) > 90:
                    short = short[:89] + "…"
                rendered.append(f"{count}x {short}")
            line += f" errors={'; '.join(rendered)}"
        elif last_error and (retries > 0 or failed > 0):
            line += f" reason={last_error}"
        database_state = str(update.get("database_state", "") or "").strip()
        database_stage = int(update.get("database_stage", 0) or 0)
        database_max_stage = int(update.get("database_max_stage", 0) or 0)
        stage_progress_obj = update.get("stage_progress_percent")
        if database_stage > 0 and database_max_stage > 0:
            line += f" db_stage={database_stage}/{database_max_stage}"
        if stage_progress_obj is not None and database_stage > 0:
            line += f" stage_progress={float(stage_progress_obj):.1f}%"
        if database_state:
            line += f" state={database_state!r}"
        connection_id = int(update.get("connection_id", 0) or 0)
        if connection_id > 0:
            line += f" connection={connection_id}"
        if is_tty:
            end = "\n" if is_final else ""
            print(f"\r\033[K{line}", end=end, flush=True)
        else:
            print(line, flush=True)
        last_by_key[key] = now

    def _print_index_result_details(results: list[object]) -> None:
        for result in results:
            retried = int(getattr(result, "retried_batches", 0) or 0)
            failed = int(getattr(result, "failed_batches", 0) or 0)
            reasons = dict(getattr(result, "error_reasons", {}) or {})
            if retried <= 0 and failed <= 0 and not reasons:
                continue
            mode = str(getattr(result, "mode", "") or "-")
            print(f"Index issues [{mode}]: retries={retried} failed={failed}")
            if reasons:
                for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]:
                    print(f"  {count:>4}  {reason}")
            else:
                last_error = str(getattr(result, "last_error", "") or "").strip()
                if last_error:
                    print(f"  last error: {last_error}")

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
                    f"bench --config {v.config_path} index --engine {v.engine} "
                    f"--mode {v.mode} --entry {v.entry}"
                )
            vector_variants = [v for v in selected if v.mode == "vector"]
            for v in vector_variants:
                if not _confirm_vector_cache_coverage(
                    cfg=v.cfg,
                    engine=v.engine,
                    entry_name=v.entry,
                    auto_yes=bool(args.yes),
                ):
                    print("Index canceled: vector cache coverage not confirmed.")
                    return 0
            manticore_vector_cfgs = [v.cfg for v in selected if is_manticore_like_type(v.engine) and v.mode == "vector"]
            if not _confirm_manticore_vector_reindex_for_configs(
                manticore_vector_cfgs, auto_yes=bool(args.yes)
            ):
                print("Index canceled: Manticore vector reindex not confirmed.")
                return 0
            for v in selected:
                if bool(args.resume) and not (v.engine == "vespa" and v.mode == "vector"):
                    raise ValueError("--resume is only supported for engine=vespa with mode=vector.")
                started_by_key.clear()
                last_by_key.clear()
                run_id = generate_run_id()
                results = run_index(
                    v.cfg,
                    engine=v.engine,
                    mode=v.mode,
                    resume=bool(args.resume),
                    progress_callback=_progress,
                )
                model_note = f" model={v.model}" if v.mode == "vector" else ""
                print(f"Indexed profile={v.entry} engine={v.engine} mode={v.mode}{model_note} run_id={run_id}")
                _print_index_result_details(results)
            print(f"Index batch completed: {len(selected)} runs")
            return 0

        if args.mode is None:
            raise ValueError("--mode is required unless running interactive `bench index` with no index parameters.")
        engine = args.engine or "manticore"
        if bool(args.resume) and not (engine == "vespa" and args.mode == "vector"):
            raise ValueError("--resume is only supported for engine=vespa with mode=vector.")
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
        if is_manticore_like_type(engine):
            needs_vector_reindex = args.mode == "vector"
            if needs_vector_reindex and not _confirm_vector_cache_coverage(
                cfg=cfg_for_engine,
                engine=engine,
                entry_name=entry_name,
                auto_yes=bool(args.yes),
            ):
                print("Index canceled: vector cache coverage not confirmed.")
                return 0
            if not _confirm_manticore_vector_reindex_if_needed(
                cfg_for_engine, needs_vector_reindex=needs_vector_reindex, auto_yes=bool(args.yes)
            ):
                print("Index canceled: Manticore vector reindex not confirmed.")
                return 0
        elif args.mode in ("vector", "hybrid"):
            if not _confirm_vector_cache_coverage(
                cfg=cfg_for_engine,
                engine=engine,
                entry_name=entry_name,
                auto_yes=bool(args.yes),
            ):
                print(f"Index canceled: {args.mode} cache coverage not confirmed.")
                return 0
        started_by_key.clear()
        last_by_key.clear()
        run_id = generate_run_id()
        results = run_index(
            cfg_for_engine,
            engine=engine,
            mode=args.mode,
            resume=bool(args.resume),
            progress_callback=_progress,
        )
        model_note = f" model={resolved_embedding_model(cfg_for_engine)}" if args.mode in ("vector", "hybrid") else ""
        print(f"Indexed profile={entry_name} engine={engine} mode={args.mode}{model_note}")
        _print_index_result_details(results)
        return 0
    except KeyboardInterrupt:
        if is_tty:
            print("")
        print("Index canceled by user.")
        return 130
    except Exception as exc:
        if is_tty:
            print("")
        print(f"FATAL: {exc}")
        return 1


def cmd_run(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.mode is None and args.engine is None:
        variants = build_variants_from_config_paths([Path(args.config)])
        if not variants:
            print("Run canceled: no variants available.")
            return 0
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
    if entry_mode_filter in ("fulltext", "vector", "hybrid") and args.mode != entry_mode_filter:
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
            generate_html=not bool(args.markdown_only),
        )
    except RuntimeError as exc:
        if str(exc) == "No artifacts selected.":
            print("Report canceled: no artifacts selected.")
            return 0
        raise
    print(f"Report generated: {out}")
    if not args.markdown_only:
        print(f"Interactive report generated: {combined_html_report_path(out)}")
    return 0


def cmd_configmap(_: AppConfig, args: argparse.Namespace) -> int:
    try:
        out = generate_configmap(config_path=args.config, output_path=args.output)
    except Exception as exc:
        print(f"FATAL: {exc}")
        return 1
    print(f"Config map generated: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", add_help=False)
    parser.add_argument("--config", default="config.yaml")

    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("engines", add_help=False)

    p_prepare = sub.add_parser("prepare", add_help=False)
    p_prepare.add_argument("--source", action="append", default=None, help="Dataset source(s) to prepare")
    p_prepare.add_argument("--entry", action="append", default=None, help="Entry name(s) to prepare source from")
    p_prepare.add_argument("--reset-embeddings", action="store_true", help="Clear current-model embeddings for the selected prepared dataset(s)")

    p_index = sub.add_parser("index", add_help=False)
    p_index.add_argument("--engine", default=None, choices=["manticore", "manticore_avgdl", "manticore_default_ranker", "manticore_all_default", "manticore_full_default", "columnar", "qdrant", "meilisearch", "opensearch", "elasticsearch", "typesense", "milvus", "mariadb", "weaviate", "vespa"])
    p_index.add_argument("--mode", default=None, choices=["fulltext", "vector", "hybrid"])
    p_index.add_argument("--entry", default=None, help="Config entry name to run (for deterministic replay)")
    p_index.add_argument(
        "--yes",
        action="store_true",
        help="Skip Manticore vector table recreation confirmation",
    )
    p_index.add_argument(
        "--resume",
        action="store_true",
        help="Resume Vespa vector indexing from the current document count after verifying boundary docs",
    )

    p_run = sub.add_parser("run", add_help=False)
    p_run.add_argument("--engine", default=None, choices=["manticore", "manticore_avgdl", "manticore_default_ranker", "manticore_all_default", "manticore_full_default", "columnar", "qdrant", "meilisearch", "opensearch", "elasticsearch", "typesense", "milvus", "mariadb", "weaviate", "vespa"])
    p_run.add_argument("--mode", default=None, choices=["fulltext", "vector", "hybrid", "both"])
    p_run.add_argument("--entry", default=None, help="Config entry name to run (for deterministic replay)")
    p_run.add_argument("--concurrency", type=int, default=None)

    p_report = sub.add_parser("report", add_help=False)
    p_report.add_argument("--run-ids", default=None, help="Comma-separated run ids to include")
    p_report.add_argument("--no-interactive", action="store_true", help="Disable interactive artifact selection")
    p_report.add_argument("--output", default=None, help="Output markdown path")
    p_report.add_argument("--title", default="Benchmark Report")
    p_report.add_argument("--markdown-only", action="store_true", help="Skip standalone HTML generation")

    p_configmap = sub.add_parser("configmap", add_help=False)
    p_configmap.add_argument("--output", default=None, help="Output markdown path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.config = str(resolve_config_path(args.config))
    if args.command is None:
        picked = _select_top_command_interactive()
        if picked is None:
            raise SystemExit(0)
        args = parser.parse_args(["--config", args.config, picked])
    cfg = load_config(args.config)
    if args.command == "engines":
        raise SystemExit(cmd_engines(cfg, args))
    if args.command == "prepare":
        raise SystemExit(cmd_prepare(cfg, args))
    if args.command == "index":
        raise SystemExit(cmd_index(cfg, args))
    if args.command == "run":
        raise SystemExit(cmd_run(cfg, args))
    if args.command == "report":
        raise SystemExit(cmd_report(cfg, args))
    if args.command == "configmap":
        raise SystemExit(cmd_configmap(cfg, args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
