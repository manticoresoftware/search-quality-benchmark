from __future__ import annotations

from pathlib import Path


def resolve_config_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    cwd = Path.cwd()
    for base in (cwd, *cwd.parents):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return raw


def discover_config_paths() -> list[Path]:
    for base in (Path.cwd(), *Path.cwd().parents):
        out: list[Path] = []
        for p in sorted(base.glob("config*.yaml")):
            if p.name.endswith(".template"):
                continue
            out.append(p)
        if out:
            return out
    return []
