from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json


def _display_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m == "fulltext":
        return "full-text"
    if m == "vector":
        return "vector"
    if m == "hybrid":
        return "hybrid"
    return mode


def metric_definitions_markdown() -> str:
    return (
        "## Metric Definitions\n\n"
        "- `ndcg@k`: ranking quality at top-k with graded relevance (higher is better).\n"
        "- `map@k`: mean average precision at top-k, emphasizing early relevant hits (higher is better).\n"
        "- `mrr@k`: mean reciprocal rank at top-k, emphasizing the rank of the first relevant hit (higher is better).\n"
        "- `p@k`: precision at top-k (fraction of top-k hits that are relevant; higher is better).\n"
        "- `recall@k`: fraction of all relevant docs retrieved in top-k (higher is better).\n"
        "- `latency_p50_ms/p95_ms/p99_ms`: median/tail query latency in milliseconds (lower is better).\n"
        "- `qps`: successful queries per second for the run workload (higher is better).\n"
        "- `error_rate`: failed queries divided by total queries (lower is better).\n"
    )


def build_summary_markdown(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark Summary ({metrics.get('run_id', 'unknown')})")
    lines.append("")
    profile = str(metrics.get("profile", "") or "").strip()
    if profile:
        lines.append(f"- Profile: `{profile}`")
    engine = str(metrics.get("engine", "") or "").strip()
    if engine:
        lines.append(f"- Engine: `{engine}`")
    engine_version = str(metrics.get("engine_version", "") or "").strip()
    if engine_version:
        lines.append(f"- Engine version: `{engine_version}`")
    engine_image = str(metrics.get("engine_image", "") or "").strip()
    if engine_image:
        lines.append(f"- Engine image: `{engine_image}`")
    engine_image_digest = str(metrics.get("engine_image_digest", "") or "").strip()
    if engine_image_digest:
        lines.append(f"- Engine image digest: `{engine_image_digest}`")
    engine_image_created_at = str(metrics.get("engine_image_created_at", "") or "").strip()
    if engine_image_created_at:
        lines.append(f"- Engine image created: `{engine_image_created_at}`")
    dataset_source = str(metrics.get("dataset_source", "") or "").strip()
    if dataset_source:
        lines.append(f"- Dataset source: `{dataset_source}`")
    lines.append(f"- Mode: `{_display_mode(str(metrics.get('mode', '')))}`")
    lines.append(f"- Runtime (sec): `{metrics.get('runtime_seconds', 0):.3f}`")
    bcfg = metrics.get("benchmark_settings", {}) or {}
    if bcfg:
        lines.append(
            f"- Query settings: concurrency=`{bcfg.get('concurrency', 0)}` "
            f"limit=`{bcfg.get('limit', 0)}` warmup=`{bcfg.get('warmup_queries', 0)}` repeats=`{bcfg.get('repeats', 0)}`"
        )
    resolved_cfg = metrics.get("resolved_config_lines", []) or []
    if resolved_cfg:
        lines.append("")
        lines.append("## Resolved Config")
        lines.append("")
        for line in resolved_cfg:
            lines.append(f"- `{line}`")
    live_state = metrics.get("live_state_lines", []) or []
    if live_state:
        lines.append("")
        lines.append("## Live State")
        lines.append("")
        for line in live_state:
            lines.append(f"- `{line}`")
    lines.append("")
    lines.append("## Relevance")
    rel = metrics.get("relevance", {})
    if not rel:
        lines.append("- No relevance metrics available.")
    for mode, vals in rel.items():
        lines.append(f"- `{_display_mode(mode)}`:")
        for k, v in vals.items():
            lines.append(f"  - {k}: `{v:.4f}`")
    lines.append("")
    lines.append("## Performance")
    perf = metrics.get("performance", {})
    if not perf:
        lines.append("- No performance metrics available.")
    for mode, vals in perf.items():
        lines.append(f"- `{_display_mode(mode)}`:")
        for key in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "qps", "error_rate", "queries_total"):
            lines.append(f"  - {key}: `{vals.get(key, 0):.4f}`")
    lines.append("")
    lines.append(metric_definitions_markdown().strip())
    return "\n".join(lines) + "\n"


def compare_summaries(current: dict[str, Any], other: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Comparison: {current.get('run_id')} vs {other.get('run_id')}")
    lines.append("")
    for mode in sorted(set(current.get("performance", {}).keys()) | set(other.get("performance", {}).keys())):
        c_perf = current.get("performance", {}).get(mode, {})
        o_perf = other.get("performance", {}).get(mode, {})
        lines.append(f"## {_display_mode(mode)}")
        for key in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "qps", "error_rate"):
            c_val = float(c_perf.get(key, 0.0))
            o_val = float(o_perf.get(key, 0.0))
            delta = c_val - o_val
            lines.append(f"- {key}: current `{c_val:.4f}` | compare `{o_val:.4f}` | delta `{delta:.4f}`")
        c_rel = current.get("relevance", {}).get(mode, {})
        o_rel = other.get("relevance", {}).get(mode, {})
        for key in sorted(set(c_rel.keys()) | set(o_rel.keys())):
            c_val = float(c_rel.get(key, 0.0))
            o_val = float(o_rel.get(key, 0.0))
            delta = c_val - o_val
            lines.append(f"- {key}: current `{c_val:.4f}` | compare `{o_val:.4f}` | delta `{delta:.4f}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_report(run_id: str, compare_id: str | None = None) -> Path:
    run_dir = Path("artifacts") / run_id
    metrics = read_json(run_dir / "metrics.json")
    summary = build_summary_markdown(metrics)
    if compare_id:
        compare_metrics = read_json(Path("artifacts") / compare_id / "metrics.json")
        summary += "\n" + compare_summaries(metrics, compare_metrics)
    out_path = run_dir / "summary.md"
    out_path.write_text(summary, encoding="utf-8")
    return out_path
