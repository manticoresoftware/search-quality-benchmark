from __future__ import annotations

import curses
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import json
import subprocess
import sys
import threading
from time import sleep
from time import perf_counter
from time import monotonic
from typing import Any

from .config import AppConfig
from .elasticsearch_backend import ElasticsearchBackend
from .manticore import ManticoreClient
from .mariadb_backend import MariaDBBackend
from .meilisearch_backend import MeilisearchBackend
from .opensearch_backend import OpenSearchBackend
from .qdrant_backend import QdrantBackend
from .table_layout import wrap_text_lines
from .typesense_backend import TypesenseBackend
from .vespa_backend import VespaBackend
from .weaviate_backend import WeaviateBackend


@dataclass
class EngineRow:
    service: str
    label: str
    status: str
    ready: str
    uptime: str
    cpu: str
    anon_ram: str
    mmap_ram: str
    mmap_locked: str
    disk: str
    action: str
    container_name: str
    compose_state: str
    health: str
    last_error: str = ""


@dataclass
class PendingAction:
    action: str
    thread: threading.Thread
    started_at: float
    error: str = ""


@dataclass
class RefreshState:
    rows: list[EngineRow]
    error: str = ""
    running: bool = False
    last_started_at: float = 0.0
    last_finished_at: float = 0.0
    disk_by_service: dict[str, str] | None = None
    disk_last_started_at: float = 0.0
    disk_last_finished_at: float = 0.0
    disk_running: bool = False


_SERVICES: list[tuple[str, str]] = [
    ("manticore", "Manticore"),
    ("qdrant", "Qdrant"),
    ("meilisearch", "Meilisearch"),
    ("opensearch", "OpenSearch"),
    ("elasticsearch", "Elasticsearch"),
    ("typesense", "Typesense"),
    ("mariadb", "MariaDB Vector"),
    ("weaviate", "Weaviate"),
    ("vespa", "Vespa"),
]


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "compose.yaml").exists():
            return cand
    return cur


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _format_cpu_display(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "-"
    if text.endswith("%"):
        return text + "c"
    return text


def _run_compose(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_docker(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_engine_dirs(project_root: Path, service: str) -> None:
    mapping = {
        "manticore": [project_root / "data" / "manticore_storage"],
        "qdrant": [project_root / "data" / "qdrant_storage"],
        "meilisearch": [project_root / "data" / "meilisearch_data"],
        "opensearch": [project_root / "data" / "opensearch_data"],
        "elasticsearch": [project_root / "data" / "elasticsearch_data"],
        "typesense": [project_root / "data" / "typesense_data"],
        "mariadb": [project_root / "data" / "mariadb_data"],
        "weaviate": [project_root / "data" / "weaviate_data"],
        "vespa": [
            project_root / "data" / "vespa_data",
            project_root / "data" / "vespa_logs",
        ],
    }
    targets = mapping.get(service)
    if targets is None:
        return
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        if service in {"opensearch", "elasticsearch"}:
            target.chmod(0o777)
        if service == "vespa":
            target.chmod(0o755)
            try:
                os.chown(target, 1000, 1000)
            except PermissionError:
                pass
            except OSError:
                pass


def _engine_data_dir(project_root: Path, service: str) -> Path | None:
    mapping = {
        "manticore": project_root / "data" / "manticore_storage",
        "qdrant": project_root / "data" / "qdrant_storage",
        "meilisearch": project_root / "data" / "meilisearch_data",
        "opensearch": project_root / "data" / "opensearch_data",
        "elasticsearch": project_root / "data" / "elasticsearch_data",
        "typesense": project_root / "data" / "typesense_data",
        "mariadb": project_root / "data" / "mariadb_data",
        "weaviate": project_root / "data" / "weaviate_data",
        "vespa": project_root / "data" / "vespa_data",
    }
    return mapping.get(service)


def _compose_service_rows(project_root: Path) -> dict[str, dict[str, Any]]:
    result = _run_compose(["ps", "--all", "--format", "json"], project_root)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "docker compose ps failed").strip())
    text = (result.stdout or "").strip()
    if not text:
        return {}
    rows = json.loads(text)
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                service = str(row.get("Service", "") or "").strip()
                if service:
                    out[service] = row
    return out


def _docker_stats(project_root: Path, container_names: list[str]) -> dict[str, dict[str, str]]:
    if not container_names:
        return {}
    result = _run_docker(["stats", "--no-stream", "--format", "{{json .}}", *container_names], project_root)
    if result.returncode != 0:
        return {}
    out: dict[str, dict[str, str]] = {}
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        name = str(
            row.get("Name", "")
            or row.get("Container", "")
            or row.get("ContainerName", "")
            or ""
        ).strip().lstrip("/")
        if name:
            cpu = str(
                row.get("CPUPerc", "")
                or row.get("CPU %", "")
                or row.get("CpuPerc", "")
                or ""
            ).strip()
            mem = str(
                row.get("MemPerc", "")
                or row.get("Mem %", "")
                or row.get("MemUsage", "")
                or row.get("Memory", "")
                or ""
            ).strip()
            out[name] = {
                "cpu": cpu,
                "mem": mem,
            }
    return out


def _human_size(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    units = ["B", "K", "M", "G", "T"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{int(num_bytes)}B"


def _human_size_kib(num_kib: int) -> str:
    return _human_size(int(num_kib) * 1024)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except Exception:
            return 0
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except Exception:
                    continue
    except Exception:
        return 0
    return total


def _parse_proc_status_kib(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, int]:
    status_path = proc_root / str(pid) / "status"
    out: dict[str, int] = {}
    try:
        text = status_path.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if not value:
            continue
        parts = value.split()
        try:
            amount = int(parts[0])
        except Exception:
            continue
        out[key.strip()] = amount
    return out


def _linux_proc_tree_pids(pid: int, *, proc_root: Path = Path("/proc")) -> list[int]:
    if pid <= 0:
        return []
    if not (proc_root / str(pid)).exists():
        return []
    parent_to_children: dict[int, list[int]] = {}
    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return [pid]
    for child_dir in proc_entries:
        if not child_dir.is_dir():
            continue
        try:
            child_pid = int(child_dir.name)
        except Exception:
            continue
        status = _parse_proc_status_kib(child_pid, proc_root=proc_root)
        if not status:
            continue
        ppid = int(status.get("PPid", 0))
        parent_to_children.setdefault(ppid, []).append(child_pid)
    seen: set[int] = set()
    ordered: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for child_pid in reversed(sorted(parent_to_children.get(current, []))):
            stack.append(child_pid)
    return ordered


def _linux_mem_breakdown(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, str]:
    if pid <= 0:
        return {"anon": "-", "mmap": "-", "locked": "-"}
    pids = _linux_proc_tree_pids(pid, proc_root=proc_root)
    if not pids:
        return {"anon": "-", "mmap": "-", "locked": "-"}
    anon_kib = 0
    mmap_kib = 0
    locked_kib = 0
    for proc_pid in pids:
        status = _parse_proc_status_kib(proc_pid, proc_root=proc_root)
        if not status:
            continue
        anon_kib += int(status.get("RssAnon", 0))
        mmap_kib += int(status.get("RssFile", 0)) + int(status.get("RssShmem", 0))
        locked_kib += int(status.get("VmLck", 0))
    return {
        "anon": _human_size_kib(anon_kib),
        "mmap": _human_size_kib(mmap_kib),
        "locked": _human_size_kib(locked_kib),
    }


def _disk_sizes(project_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for service, _ in _SERVICES:
        data_dir = _engine_data_dir(project_root, service)
        if data_dir is None:
            out[service] = "-"
            continue
        out[service] = _human_size(_dir_size(data_dir))
    return out


def _docker_inspect(project_root: Path, container_names: list[str]) -> dict[str, dict[str, Any]]:
    if not container_names:
        return {}
    result = _run_docker(["inspect", *container_names], project_root)
    if result.returncode != 0:
        return {}
    try:
        rows = json.loads(result.stdout or "[]")
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name", "") or "").strip().lstrip("/")
        if name:
            out[name] = row
    return out


def _format_uptime(started_at: str) -> str:
    raw = str(started_at or "").strip()
    if not raw or raw.startswith("0001-01-01"):
        return "-"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        seconds = max(0, int((now - dt).total_seconds()))
    except Exception:
        return "-"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h"
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _engine_ready(cfg: AppConfig, service: str) -> tuple[bool, str]:
    try:
        if service == "manticore":
            client = ManticoreClient(cfg.manticore)
            try:
                client.query("SHOW TABLES")
            finally:
                client.close()
            return True, ""
        if service == "qdrant":
            QdrantBackend(cfg.qdrant).ensure_accessible()
            return True, ""
        if service == "meilisearch":
            MeilisearchBackend(cfg.meilisearch).ensure_accessible()
            return True, ""
        if service == "opensearch":
            OpenSearchBackend(cfg.opensearch).ensure_accessible()
            return True, ""
        if service == "elasticsearch":
            ElasticsearchBackend(cfg.elasticsearch).ensure_accessible()
            return True, ""
        if service == "typesense":
            TypesenseBackend(cfg.typesense).ensure_accessible()
            return True, ""
        if service == "mariadb":
            backend = MariaDBBackend(cfg.mariadb)
            try:
                backend.ensure_accessible()
            finally:
                backend.close()
            return True, ""
        if service == "weaviate":
            WeaviateBackend(cfg.weaviate).ensure_accessible()
            return True, ""
        if service == "vespa":
            VespaBackend(cfg.vespa).ensure_accessible()
            return True, ""
        return False, "unknown service"
    except Exception as exc:
        return False, str(exc)


def _build_rows(
    cfg: AppConfig,
    project_root: Path,
    pending: dict[str, PendingAction],
    disk_by_service: dict[str, str] | None = None,
) -> list[EngineRow]:
    compose_error = ""
    try:
        compose_rows = _compose_service_rows(project_root)
    except Exception as exc:
        compose_rows = {}
        compose_error = str(exc)
    container_names = [
        str((compose_rows.get(service, {}) or {}).get("Name", "") or "").strip()
        for service, _ in _SERVICES
        if str((compose_rows.get(service, {}) or {}).get("Name", "") or "").strip()
    ]
    stats = _docker_stats(project_root, container_names) if not compose_error else {}
    inspect = _docker_inspect(project_root, container_names) if not compose_error else {}
    disk_by_service = disk_by_service or {}
    rows: list[EngineRow] = []
    for service, label in _SERVICES:
        info = compose_rows.get(service, {})
        container_name = str(info.get("Name", "") or "").strip()
        state = str(info.get("State", "") or "").strip().lower()
        health = str(info.get("Health", "") or "").strip().lower()
        ready = False
        ready_error = ""
        if state == "running":
            ready, ready_error = _engine_ready(cfg, service)
        pending_action = pending.get(service)
        status = "stopped"
        if compose_error:
            status = "unavailable"
        elif pending_action is not None and pending_action.thread.is_alive():
            status = "starting" if pending_action.action == "start" else "stopping"
        elif state in {"created", "restarting"}:
            status = "starting"
        elif state in {"removing"}:
            status = "stopping"
        elif state == "running":
            status = "running" if ready else "starting"
        elif state in {"exited", "dead"}:
            status = "stopped"
        ready_label = "yes" if ready else ("no" if state == "running" else "-")
        uptime = "-"
        if container_name:
            started_at = (((inspect.get(container_name, {}) or {}).get("State") or {}).get("StartedAt") or "")
            uptime = _format_uptime(str(started_at))
        cpu_label = "-"
        anon_label = "-"
        mmap_label = "-"
        locked_label = "-"
        if container_name and state == "running":
            cpu = str((stats.get(container_name, {}) or {}).get("cpu", "") or "").strip()
            cpu_label = _format_cpu_display(cpu)
            pid = int((((inspect.get(container_name, {}) or {}).get("State") or {}).get("Pid") or 0) or 0)
            mem_breakdown = _linux_mem_breakdown(pid)
            anon_label = mem_breakdown["anon"]
            mmap_label = mem_breakdown["mmap"]
            locked_label = mem_breakdown["locked"]
        disk_label = str(disk_by_service.get(service, "-") or "-")
        action = "[Stop]" if status in {"running", "starting"} else "[Start]"
        last_error = ""
        if pending_action is not None and not pending_action.thread.is_alive() and pending_action.error:
            last_error = pending_action.error
        elif ready_error and status == "starting":
            last_error = ready_error
        elif compose_error:
            last_error = compose_error
        rows.append(
            EngineRow(
                service=service,
                label=label,
                status=status,
                ready=ready_label,
                uptime=uptime,
                cpu=cpu_label,
                anon_ram=anon_label,
                mmap_ram=mmap_label,
                mmap_locked=locked_label,
                disk=disk_label,
                action=action,
                container_name=container_name,
                compose_state=state or "-",
                health=health or "-",
                last_error=last_error,
            )
        )
    return rows


def _start_service(project_root: Path, service: str, pending: PendingAction) -> None:
    try:
        _ensure_engine_dirs(project_root, service)
        result = _run_compose(["up", "-d", service], project_root)
        if result.returncode != 0:
            pending.error = (result.stderr or result.stdout or "docker compose up failed").strip()
    except Exception as exc:
        pending.error = str(exc)


def _stop_service(project_root: Path, service: str, pending: PendingAction) -> None:
    try:
        result = _run_compose(["rm", "-f", "-s", "-v", service], project_root)
        if result.returncode != 0:
            pending.error = (result.stderr or result.stdout or "docker compose rm failed").strip()
    except Exception as exc:
        pending.error = str(exc)


def cmd_engines(cfg: AppConfig, _: Any) -> int:
    project_root = _find_project_root(Path.cwd())
    compose_path = project_root / "compose.yaml"
    if not compose_path.exists():
        print(f"FATAL: Missing compose file: {compose_path}")
        return 1

    pending: dict[str, PendingAction] = {}

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        try:
            rows = _build_rows(cfg, project_root, pending)
        except Exception as exc:
            print(f"FATAL: {exc}")
            return 1
        for row in rows:
            print(
                f"{row.label}: status={row.status} ready={row.ready} uptime={row.uptime} "
                f"load={row.load} container={row.container_name or '-'}"
            )
        return 0

    def _inner(stdscr: Any) -> int:
        current = 0
        rows: list[EngineRow] = []
        refresh = RefreshState(rows=[])
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.timeout(100)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # running
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # starting/stopping
            curses.init_pair(3, curses.COLOR_RED, -1)     # stopped/error
            curses.init_pair(4, curses.COLOR_CYAN, -1)    # ready

        refresh_interval = 1.0
        disk_refresh_interval = 15.0

        def _begin_refresh(force: bool = False) -> None:
            now = monotonic()
            if refresh.running:
                return
            if not force and refresh.last_started_at and (now - refresh.last_started_at) < refresh_interval:
                return

            def _worker() -> None:
                refresh.running = True
                refresh.last_started_at = monotonic()
                try:
                    refresh.rows = _build_rows(cfg, project_root, pending, refresh.disk_by_service)
                    refresh.error = ""
                except Exception as exc:
                    refresh.rows = []
                    refresh.error = str(exc)
                finally:
                    refresh.last_finished_at = monotonic()
                    refresh.running = False

            threading.Thread(target=_worker, daemon=True).start()

        def _begin_disk_refresh(force: bool = False) -> None:
            now = monotonic()
            if refresh.disk_running:
                return
            if not force and refresh.disk_last_started_at and (now - refresh.disk_last_started_at) < disk_refresh_interval:
                return

            def _worker() -> None:
                refresh.disk_running = True
                refresh.disk_last_started_at = monotonic()
                try:
                    refresh.disk_by_service = _disk_sizes(project_root)
                finally:
                    refresh.disk_last_finished_at = monotonic()
                    refresh.disk_running = False

            threading.Thread(target=_worker, daemon=True).start()

        _begin_disk_refresh(force=True)
        _begin_refresh(force=True)

        while True:
            _begin_disk_refresh()
            _begin_refresh()
            rows = refresh.rows
            if refresh.error and not rows:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                header = "Engines: refresh failed  ESC/q cancel"
                if refresh.running:
                    header = "Engines: refreshing  ESC/q cancel"
                stdscr.addstr(0, 0, _fit(header, max(1, w - 1)), curses.A_BOLD)
                stdscr.addstr(2, 0, _fit(refresh.error, max(1, w - 1)), curses.A_DIM)
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (27, ord("q")):
                    return 1
                if ch in (ord("r"), ord("R")):
                    _begin_refresh(force=True)
                continue

            current = min(max(0, current), max(0, len(rows) - 1))
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            header = "Engines: ↑/↓ move  ENTER start/stop  r refresh  ESC/q cancel"
            if refresh.running:
                header += "  [refreshing]"
            if refresh.disk_running:
                header += "  [disk]"
            stdscr.addstr(
                0,
                0,
                _fit(header, max(1, w - 1)),
                curses.A_BOLD,
            )
            headers = [("ENGINE", 13), ("STATUS", 10), ("READY", 7), ("UPTIME", 10), ("CPU*", 9), ("ANON", 8), ("MMAP", 8), ("LOCKED", 8), ("DISK", 8), ("ACTION", 8)]
            x = 0
            for title, width in headers:
                stdscr.addstr(2, x, _fit(title, width), curses.A_BOLD)
                x += width + 1
            for i, row in enumerate(rows):
                attrs = curses.A_REVERSE if i == current else curses.A_NORMAL
                status_color = 0
                ready_color = 0
                if curses.has_colors():
                    if row.status == "running":
                        status_color = curses.color_pair(1)
                    elif row.status in {"starting", "stopping"}:
                        status_color = curses.color_pair(2)
                    else:
                        status_color = curses.color_pair(3)
                    if row.ready == "yes":
                        ready_color = curses.color_pair(4)
                values = [
                    (row.label, 13, attrs),
                    (row.status, 10, attrs | status_color),
                    (row.ready, 7, attrs | ready_color),
                    (row.uptime, 10, attrs),
                    (row.cpu, 9, attrs),
                    (row.anon_ram, 8, attrs),
                    (row.mmap_ram, 8, attrs),
                    (row.mmap_locked, 8, attrs),
                    (row.disk, 8, attrs),
                    (row.action, 8, attrs),
                ]
                x = 0
                y = 4 + i
                for text, width, cell_attrs in values:
                    stdscr.addstr(y, x, _fit(text, width), cell_attrs)
                    x += width + 1

            if rows:
                row = rows[current]
                panel_y = 4 + len(rows) + 2
                lines = [
                    "Details",
                    f"service: {row.service}",
                    f"container: {row.container_name or '-'}",
                    f"compose state: {row.compose_state}",
                    f"health: {row.health}",
                    "CPU*: Docker CPU%; 100% means one fully used host CPU core",
                    "ANON: Linux RssAnon from /proc/<pid>/status",
                    "MMAP: Linux RssFile + RssShmem from /proc/<pid>/status",
                    "LOCKED: Linux VmLck from /proc/<pid>/status",
                    f"compose file: {compose_path}",
                ]
                if row.last_error:
                    lines.append(f"last error: {row.last_error}")
                yy = panel_y
                for i, line in enumerate(lines):
                    if yy >= h - 1:
                        break
                    attrs = curses.A_BOLD if i == 0 else curses.A_DIM
                    for part in wrap_text_lines(line, max(10, w - 1)):
                        if yy >= h - 1:
                            break
                        stdscr.addstr(yy, 0, _fit(part, max(1, w - 1)), attrs)
                        yy += 1
            stdscr.refresh()
            ch = stdscr.getch()
            if ch == -1 or ch in (ord("r"), ord("R")):
                if ch in (ord("r"), ord("R")):
                    _begin_disk_refresh(force=True)
                    _begin_refresh(force=True)
                continue
            if ch in (curses.KEY_UP, ord("k")):
                current = max(0, current - 1)
                continue
            if ch in (curses.KEY_DOWN, ord("j")):
                current = min(len(rows) - 1, current + 1)
                continue
            if ch in (10, 13, curses.KEY_ENTER):
                if not rows:
                    continue
                row = rows[current]
                existing = pending.get(row.service)
                if existing is not None and existing.thread.is_alive():
                    continue
                action = "stop" if row.status in {"running", "starting"} else "start"
                pending_action = PendingAction(
                    action=action,
                    thread=threading.Thread(),
                    started_at=0.0,
                )
                target_fn = _stop_service if action == "stop" else _start_service
                thread = threading.Thread(
                    target=target_fn,
                    args=(project_root, row.service, pending_action),
                    daemon=True,
                )
                pending_action.thread = thread
                pending_action.started_at = perf_counter()
                pending[row.service] = pending_action
                for i, existing_row in enumerate(refresh.rows):
                    if existing_row.service != row.service:
                        continue
                    refresh.rows[i] = EngineRow(
                        service=existing_row.service,
                        label=existing_row.label,
                        status="stopping" if action == "stop" else "starting",
                        ready=existing_row.ready,
                        uptime=existing_row.uptime,
                        cpu=existing_row.cpu,
                        anon_ram=existing_row.anon_ram,
                        mmap_ram=existing_row.mmap_ram,
                        mmap_locked=existing_row.mmap_locked,
                        disk=existing_row.disk,
                        action="[Start]" if action == "stop" else "[Stop]",
                        container_name=existing_row.container_name,
                        compose_state=existing_row.compose_state,
                        health=existing_row.health,
                        last_error="",
                    )
                    break
                thread.start()
                _begin_disk_refresh(force=True)
                _begin_refresh(force=True)
                continue
            if ch in (27, ord("q")):
                return 0
        return 0

    try:
        return curses.wrapper(_inner)
    except KeyboardInterrupt:
        print("Engines canceled.")
        return 130
