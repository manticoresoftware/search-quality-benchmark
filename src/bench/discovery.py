from __future__ import annotations

from pathlib import Path


def discover_config_paths() -> list[Path]:
    out: list[Path] = []
    for p in sorted(Path(".").glob("config*.yaml")):
        if p.name.endswith(".template"):
            continue
        out.append(p)
    return out
