from __future__ import annotations

from pathlib import Path


def resolve_config_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return raw
    if raw.exists():
        return raw.resolve()
    cwd = Path.cwd().resolve()
    for cand_dir in [cwd, *cwd.parents]:
        cand = cand_dir / raw
        if cand.exists():
            return cand.resolve()
    return raw.resolve()


def discover_config_paths() -> list[Path]:
    out: list[Path] = []
    root = None
    cwd = Path.cwd().resolve()
    for cand_dir in [cwd, *cwd.parents]:
        matches = [p for p in sorted(cand_dir.glob("config*.yaml")) if not p.name.endswith(".template")]
        if matches:
            root = cand_dir
            break
    if root is None:
        root = cwd
    for p in sorted(root.glob("config*.yaml")):
        if p.name.endswith(".template"):
            continue
        out.append(p.resolve())
    return out
