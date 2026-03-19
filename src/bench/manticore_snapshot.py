from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import perf_counter

from .config import AppConfig
from .io_utils import ensure_dir


@dataclass
class SnapshotResult:
    action: str
    output_path: str
    tables: list[str]


def _tables_for_mode(cfg: AppConfig, mode: str) -> list[str]:
    if mode == "fulltext":
        return [cfg.manticore.fulltext_table]
    if mode == "vector":
        return [cfg.manticore.vector_table]
    if mode == "both":
        return [cfg.manticore.fulltext_table, cfg.manticore.vector_table]
    raise ValueError("mode must be one of: fulltext, vector, both")


def _mysql_client_base(cfg: AppConfig) -> list[str]:
    base = ["-h", cfg.manticore.host, "-P", str(cfg.manticore.sql_port)]
    if cfg.manticore.user:
        base.extend(["-u", cfg.manticore.user])
    return base


def _with_password_env(cfg: AppConfig) -> dict[str, str]:
    env = os.environ.copy()
    if cfg.manticore.password:
        env["MYSQL_PWD"] = cfg.manticore.password
    return env


def _resolved_vector_api_key(cfg: AppConfig) -> str:
    key = (cfg.manticore.vector_api_key or "").strip()
    if key:
        return key
    key_file = (cfg.manticore.vector_api_key_file or "").strip()
    if not key_file:
        return ""
    p = Path(key_file)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def _vector_api_key_source(cfg: AppConfig) -> str:
    key = (cfg.manticore.vector_api_key or "").strip()
    if key:
        return "manticore.vector_api_key (inline value)"
    key_file = (cfg.manticore.vector_api_key_file or "").strip()
    if not key_file:
        return "none (no manticore.vector_api_key or manticore.vector_api_key_file configured)"
    p = Path(key_file)
    if not p.exists():
        return f"manticore.vector_api_key_file={key_file} (file not found)"
    return f"manticore.vector_api_key_file={key_file} (file loaded)"


def dump_tables(cfg: AppConfig, mode: str, output_path: str) -> SnapshotResult:
    tables = _tables_for_mode(cfg, mode)
    out = Path(output_path)
    ensure_dir(out.parent)
    cmd = [
        "mysqldump",
        *_mysql_client_base(cfg),
        "--no-tablespaces",
        "--skip-comments",
        "Manticore",
        *tables,
    ]
    env = _with_password_env(cfg)
    with out.open("wb") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"mysqldump failed: {stderr.strip()}") from exc
    return SnapshotResult(action="dump", output_path=str(out), tables=tables)


def restore_dump(cfg: AppConfig, input_path: str) -> SnapshotResult:
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    print(f"Snapshot restore: vector API key source: {_vector_api_key_source(cfg)}", flush=True)

    sql_source = inp
    temp_path: Path | None = None
    api_key = _resolved_vector_api_key(cfg)
    if api_key:
        safe_key = api_key.replace("'", "''")
        pattern = re.compile(r"API_KEY='[^']*'")
        inject_pattern = re.compile(r"(?i)\bFROM\s*=")
        tf = tempfile.NamedTemporaryFile(prefix="manticore_restore_", suffix=".sql", delete=False)
        tf.close()
        temp_path = Path(tf.name)
        replaced = 0
        injected = 0
        started = perf_counter()
        total_bytes = inp.stat().st_size
        processed = 0
        last_print = 0.0
        print(
            "Snapshot restore: preparing SQL stream (replacing embedded API_KEY and injecting missing API_KEY into vector DDL)...",
            flush=True,
        )
        with inp.open("r", encoding="utf-8", errors="ignore") as src, temp_path.open("w", encoding="utf-8") as dst:
            for line in src:
                processed += len(line.encode("utf-8", errors="ignore"))
                newline, n = pattern.subn(f"API_KEY='{safe_key}'", line)
                replaced += n
                if (
                    n == 0
                    and "float_vector" in newline.lower()
                    and "model_name" in newline.lower()
                    and "api_key" not in newline.lower()
                ):
                    newer, k = inject_pattern.subn(f" API_KEY='{safe_key}' FROM=", newline, count=1)
                    if k > 0:
                        newline = newer
                        injected += 1
                dst.write(newline)
                now = perf_counter()
                if now - last_print >= 1.0:
                    pct = (processed / total_bytes * 100.0) if total_bytes > 0 else 0.0
                    print(
                        f"Snapshot restore: prepare progress {processed}/{total_bytes} bytes ({pct:5.1f}%) elapsed={now-started:6.1f}s",
                        flush=True,
                    )
                    last_print = now
        print(
            f"Snapshot restore: prepare done, API_KEY replacements={replaced}, API_KEY injections={injected}",
            flush=True,
        )
        if replaced > 0 or injected > 0:
            sql_source = temp_path
        else:
            temp_path.unlink(missing_ok=True)
            temp_path = None

    cmd = ["mysql", *_mysql_client_base(cfg)]
    env = _with_password_env(cfg)
    print(f"Snapshot restore: importing SQL into Manticore from {sql_source} ...", flush=True)
    with sql_source.open("rb") as f:
        try:
            subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"mysql restore failed: {stderr.strip()}") from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    print("Snapshot restore: import completed.", flush=True)
    return SnapshotResult(action="restore", output_path=str(inp), tables=[])
