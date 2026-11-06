from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
import yaml

from .config import load_engine_targets


def _node_id(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "").strip())
    if not safe:
        safe = "node"
    if safe[0].isdigit():
        safe = f"n_{safe}"
    return safe


def _engine_color(engine: str) -> str:
    return {
        "manticore": "#d9f2d9",
        "qdrant": "#fff1cc",
        "meilisearch": "#d9f0ff",
        "opensearch": "#f7d9ff",
        "milvus": "#cceeff",
        "mariadb": "#ffe0b2",
        "serenedb": "#ffd7c2",
    }.get(engine, "#eeeeee")


def _mode_label(mode: str | None) -> str:
    if mode == "fulltext":
        return "full-text"
    if mode == "vector":
        return "vector"
    if mode == "hybrid":
        return "hybrid"
    return "base"


def _raw_entry_defs(config_path: str | Path) -> dict[str, dict]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _flatten_entry_items(payload: dict, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in payload.items():
        if key == "base":
            continue
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.extend(_flatten_entry_items(value, prefix=full_key))
            continue
        rendered = str(value)
        rendered = re.sub(r"\s+", " ", rendered).strip()
        if len(rendered) > 42:
            rendered = rendered[:41] + "…"
        out.append((full_key, rendered))
    return out


def _node_label(target: object, entry_defs: dict[str, dict], *, max_items: int = 6) -> str:
    name = str(target.name)
    raw_items = _flatten_entry_items(entry_defs.get(name, {}))
    lines = [f"<b>{name}</b>"]
    if not raw_items:
        lines.append(_mode_label(getattr(target, "mode_filter", None)))
    else:
        shown = raw_items[:max_items]
        for key, value in shown:
            lines.append(f"{key}={value}")
        remaining = len(raw_items) - len(shown)
        if remaining > 0:
            lines.append(f"+{remaining} more")
    return "<br/>".join(lines)


def render_configmap_markdown(config_path: str | Path) -> str:
    targets = load_engine_targets(config_path)
    entry_defs = _raw_entry_defs(config_path)
    groups: dict[str, list[object]] = defaultdict(list)
    for target in targets:
        groups[str(target.type)].append(target)

    lines: list[str] = []
    lines.append("# Config Map")
    lines.append("")
    lines.append(f"- Config: `{Path(config_path)}`")
    lines.append(f"- Profiles: `{len(targets)}`")
    lines.append("")
    for engine in ("manticore", "qdrant", "meilisearch", "opensearch", "milvus", "mariadb", "serenedb"):
        entries = groups.get(engine, [])
        if not entries:
            continue
        lines.append(f"## {engine}")
        lines.append("")
        lines.append("```mermaid")
        lines.append("flowchart TD")
        lines.append(f"    classDef {engine} fill:{_engine_color(engine)},stroke:#666,color:#111;")
        lines.append("")
        for target in entries:
            nid = _node_id(target.name)
            label = _node_label(target, entry_defs)
            lines.append(f'    {nid}["{label}"]')
        lines.append("")
        for target in entries:
            if target.base_name:
                lines.append(f"    {_node_id(target.base_name)} --> {_node_id(target.name)}")
        lines.append("")
        for target in entries:
            lines.append(f"    class {_node_id(target.name)} {target.type};")
        lines.append("```")
        lines.append("")
    lines.append("Legend:")
    lines.append("- Arrow means `base -> derived profile`.")
    lines.append("- Each node shows the profile name plus the YAML keys explicitly defined on that entry.")
    lines.append("- If an entry defines nothing beyond inheritance, the node shows its resolved mode instead.")
    return "\n".join(lines) + "\n"


def generate_configmap(config_path: str | Path, output_path: str | Path | None = None) -> Path:
    out = (
        Path(output_path)
        if output_path
        else Path("artifacts") / f"configmap-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_configmap_markdown(config_path), encoding="utf-8")
    return out
