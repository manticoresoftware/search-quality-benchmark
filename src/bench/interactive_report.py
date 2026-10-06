#!/usr/bin/env python3
"""Build the standalone search benchmark dashboard.

The generated HTML has no network or runtime dependencies. Data is read from
individual artifact directories; combined Markdown reports are intentionally
not used.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parent


def locate_repository() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, SOURCE_DIR, *SOURCE_DIR.parents]
    for candidate in candidates:
        if (candidate / "artifacts").is_dir() and (candidate / "src" / "bench").is_dir():
            return candidate
    raise RuntimeError("Could not locate the benchmark repository containing artifacts/")


ROOT = locate_repository()
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_PATH = ROOT / "reports" / "search-benchmark" / "index.html"

SLOW_QUERIES_PER_RUN = 8
UNKNOWN_VERSION_VALUES = {"", "unknown", "n/a", "na", "none", "null", "latest"}

DISPLAY_NAMES = {
    "columnar": "Manticore",
    "elasticsearch": "Elasticsearch",
    "mariadb": "MariaDB",
    "meilisearch": "Meilisearch",
    "milvus": "Milvus",
    "opensearch": "OpenSearch",
    "qdrant": "Qdrant",
    "typesense": "Typesense",
    "vespa": "Vespa",
    "weaviate": "Weaviate",
}

DATASET_NAMES = {
    "beir/arguana": "Arguana",
    "beir/cqadupstack/webmasters": "CQADupStack / webmasters",
    "beir/fiqa/test": "FiQA",
    "beir/nfcorpus/test": "NFCorpus",
    "beir/scidocs": "SCIDOCS",
    "beir/scifact/test": "SciFact",
    "beir/trec-covid": "TREC-COVID",
    "dl-typo": "DL-Typo",
    "hf-beir:theatticusproject/acord:test": "ACORD",
    "msmarco-passage/trec-dl-2019/judged": "TREC-DL 2019",
    "msmarco-passage/trec-dl-2020/judged": "TREC-DL 2020",
    "wands": "WANDS",
}


def extract_doc_count(payload: dict[str, Any]) -> int | None:
    for line in payload.get("live_state_lines", []) or []:
        match = re.search(r"rows \([^)]+\):\s*([0-9][0-9,]*)", str(line))
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def canonical_metric(dataset: str) -> str:
    return "mrr@10" if dataset.lower().startswith("msmarco-passage/") else "ndcg@10"


def dataset_label(dataset: str, docs: int | None) -> str:
    if dataset == "msmarco-passage/dev/small":
        base = "MS MARCO Dev (250K corpus)" if docs is not None and docs < 1_000_000 else "MS MARCO Dev (full corpus)"
    else:
        base = DATASET_NAMES.get(dataset, dataset)
    return f"{base} · {docs:,} docs" if docs is not None else base


def safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def known_engine_version(payload: dict[str, Any]) -> str | None:
    version = str(payload.get("engine_version") or "").strip()
    return version if version.casefold() not in UNKNOWN_VERSION_VALUES else None


def slow_queries(csv_path: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    heap: list[tuple[float, int, dict[str, Any]]] = []
    sequence = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            success = str(row.get("success", "")).strip().lower()
            if success not in {"true", "1", "yes"}:
                continue
            latency = safe_number(row.get("latency_ms"))
            item = {
                "runId": run["runId"],
                "datasetKey": run["datasetKey"],
                "engine": run["engine"],
                "version": run["version"],
                "imageCreatedAt": run["imageCreatedAt"],
                "mode": run["mode"],
                "topicId": str(row.get("topic_id", "")),
                "query": str(row.get("query", "")),
                "latency": latency,
            }
            entry = (latency, sequence, item)
            sequence += 1
            if len(heap) < SLOW_QUERIES_PER_RUN:
                heapq.heappush(heap, entry)
            elif latency > heap[0][0]:
                heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, reverse=True)]


def collect_data(default_run_ids: set[str] | None = None) -> dict[str, Any]:
    systems: dict[str, dict[str, str]] = {
        engine: {"name": name} for engine, name in DISPLAY_NAMES.items()
    }

    runs: list[dict[str, Any]] = []
    artifact_dirs: dict[str, Path] = {}
    excluded_manticore = 0
    invalid = 0

    for metrics_path in sorted(ARTIFACTS_DIR.glob("*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        version = known_engine_version(payload)
        if version is None:
            continue
        run_at = int(payload.get("run_at_unix") or 0)
        engine = str(payload.get("engine", "")).strip().lower()
        if engine == "manticore":
            excluded_manticore += 1
            continue
        if engine not in systems:
            continue
        mode = str(payload.get("mode", "")).strip().lower()
        dataset = str(payload.get("dataset_source", "")).strip()
        docs = extract_doc_count(payload)
        dataset_key = f"{dataset}|{docs if docs is not None else 'unknown'}"
        relevance = payload.get("relevance", {}) or {}
        performance = payload.get("performance", {}) or {}
        rel = relevance.get(mode, {}) if isinstance(relevance, dict) else {}
        perf = performance.get(mode, {}) if isinstance(performance, dict) else {}
        metric = canonical_metric(dataset)
        run_id = str(payload.get("run_id") or metrics_path.parent.name)
        settings = payload.get("benchmark_settings", {}) or {}
        run = {
            "runId": run_id,
            "runAt": run_at,
            "date": datetime.fromtimestamp(run_at, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "engine": engine,
            "version": version,
            "image": str(payload.get("engine_image") or "").strip() or "unknown",
            "imageCreatedAt": str(payload.get("engine_image_created_at") or "").strip(),
            "mode": mode,
            "dataset": dataset,
            "datasetKey": dataset_key,
            "docs": docs,
            "metric": metric,
            "quality": safe_number(rel.get(metric)),
            "ndcg10": safe_number(rel.get("ndcg@10")),
            "ndcg20": safe_number(rel.get("ndcg@20")),
            "mrr10": safe_number(rel.get("mrr@10")),
            "map10": safe_number(rel.get("map@10")),
            "p20": safe_number(rel.get("p@20")),
            "recall100": safe_number(rel.get("recall@100")),
            "qps": safe_number(perf.get("qps")),
            "p50": safe_number(perf.get("latency_p50_ms")),
            "p95": safe_number(perf.get("latency_p95_ms")),
            "p99": safe_number(perf.get("latency_p99_ms")),
            "queries": int(safe_number(perf.get("queries_total"))),
            "runtime": safe_number(payload.get("runtime_seconds")),
            "concurrency": settings.get("concurrency"),
            "limit": settings.get("limit"),
            "warmup": settings.get("warmup_queries"),
            "repeats": settings.get("repeats"),
            "model": str(payload.get("model") or ("n/a" if mode == "fulltext" else "unknown")),
            "profile": str(payload.get("profile") or ""),
        }
        runs.append(run)
        artifact_dirs[run_id] = metrics_path.parent

    runs.sort(key=lambda item: (item["runAt"], item["runId"]))
    for engine, system in systems.items():
        engine_runs = [run for run in runs if run["engine"] == engine]
        versions = sorted({run["version"] for run in engine_runs if run["version"] != "unknown"})
        images = sorted({run["image"] for run in engine_runs if run["image"] != "unknown"})
        image_dates = sorted({run["imageCreatedAt"] for run in engine_runs if run["imageCreatedAt"]})
        system["version"] = ", ".join(versions) or "unknown"
        system["image"] = ", ".join(images) or "unknown"
        system["imageCreatedAt"] = image_dates[0] if len(image_dates) == 1 else ""

    slowest: list[dict[str, Any]] = []
    for run in runs:
        slowest.extend(slow_queries(artifact_dirs[run["runId"]] / "per_query.csv", run))
    slowest.sort(key=lambda item: item["latency"], reverse=True)

    dataset_map: dict[str, dict[str, Any]] = {}
    for run in runs:
        dataset_map[run["datasetKey"]] = {
            "key": run["datasetKey"],
            "source": run["dataset"],
            "docs": run["docs"],
            "label": dataset_label(run["dataset"], run["docs"]),
            "metric": run["metric"],
        }

    datasets = sorted(dataset_map.values(), key=lambda item: item["label"].lower())
    default_runs = [run for run in runs if not default_run_ids or run["runId"] in default_run_ids]
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return {
        "meta": {
            "title": "Search benchmark",
            "generatedAt": generated_at,
            "runs": len(runs),
            "excludedManticoreRuns": excluded_manticore,
            "invalidArtifacts": invalid,
            "slowQueriesPerRun": SLOW_QUERIES_PER_RUN,
            "source": "Individual artifacts/*/metrics.json and artifacts/*/per_query.csv files",
            "defaultDatasets": sorted({run["datasetKey"] for run in default_runs}),
            "defaultEngines": sorted({run["engine"] for run in default_runs}),
            "defaultModes": sorted({run["mode"] for run in default_runs}),
        },
        "systems": systems,
        "datasets": datasets,
        "runs": runs,
        "slowQueries": slowest,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <title>Search Quality Benchmark</title>
  <style>
    :root {
      --ink: #e8eef4;
      --muted: #93a4b5;
      --faint: #667789;
      --page: #071014;
      --panel: #0d181d;
      --panel-2: #111f25;
      --line: #20343d;
      --line-2: #2c4652;
      --accent: #2bd4ad;
      --accent-2: #67a6ff;
      --warm: #ffb65c;
      --danger: #ff6f7f;
      --shadow: 0 20px 60px rgba(0, 0, 0, .24);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% -10%, rgba(43, 212, 173, .14), transparent 32rem),
        radial-gradient(circle at 92% 0%, rgba(103, 166, 255, .12), transparent 34rem),
        var(--page);
      font: 14px/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: #8fc5ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    button, select { font: inherit; }
    .shell { width: min(1560px, calc(100% - 32px)); margin: 0 auto; padding-bottom: 64px; }
    header { padding: 52px 2px 26px; display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }
    h1 { margin: 0 0 7px; font-size: clamp(32px, 5vw, 64px); line-height: .98; letter-spacing: -.045em; }
    .lede { max-width: 820px; color: var(--muted); font-size: 16px; }
    .stamp { color: var(--muted); text-align: right; font-size: 12px; }
    .stamp strong { color: var(--ink); display: block; font-size: 14px; }
    .filter-bar {
      position: sticky; top: 10px; z-index: 10; display: grid;
      grid-template-columns: minmax(220px, 1.5fr) minmax(180px, 1fr) minmax(165px, .9fr) minmax(240px, 1.4fr) auto;
      gap: 10px; align-items: end; padding: 13px; margin-bottom: 18px;
      border: 1px solid rgba(75, 108, 121, .55); border-radius: 16px;
      background: rgba(9, 20, 25, .90); backdrop-filter: blur(18px); box-shadow: var(--shadow);
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .filter-field { position: relative; display: grid; gap: 5px; min-width: 0; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .multi-select { position: relative; min-width: 0; }
    .multi-select summary { min-height: 39px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 10px; color: var(--ink); background: #132329; border: 1px solid var(--line-2); border-radius: 10px; cursor: pointer; list-style: none; text-transform: none; letter-spacing: 0; font-weight: 500; white-space: nowrap; overflow: hidden; }
    .multi-select summary::-webkit-details-marker { display: none; }
    .multi-select summary::after { content: '▾'; flex: none; color: var(--muted); }
    .multi-select[open] summary { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(43,212,173,.12); }
    .multi-summary { overflow: hidden; text-overflow: ellipsis; }
    .multi-menu { position: absolute; z-index: 30; top: calc(100% + 6px); left: 0; width: min(390px, 88vw); padding: 9px; color: var(--ink); background: #101f25; border: 1px solid var(--line-2); border-radius: 12px; box-shadow: 0 18px 50px rgba(0,0,0,.48); text-transform: none; letter-spacing: 0; font-weight: 500; }
    .multi-actions { display: flex; gap: 6px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
    .multi-actions button { min-height: 30px; padding: 4px 9px; font-size: 11px; }
    .multi-options { display: grid; gap: 2px; max-height: 330px; overflow: auto; padding-top: 7px; }
    label.multi-option { display: flex; grid-template-columns: none; flex-direction: row; align-items: flex-start; gap: 8px; padding: 6px 5px; color: var(--ink); border-radius: 6px; font-size: 11px; font-weight: 500; text-transform: none; letter-spacing: 0; cursor: pointer; }
    label.multi-option:hover { background: rgba(103,166,255,.08); }
    .multi-option input { margin: 2px 0 0; accent-color: var(--accent); }
    .multi-option small { display: block; color: var(--faint); }
    select, button {
      min-height: 39px; color: var(--ink); background: #132329; border: 1px solid var(--line-2);
      border-radius: 10px; padding: 8px 10px; outline: none;
    }
    select:focus, button:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(43, 212, 173, .12); }
    button { cursor: pointer; font-weight: 700; }
    button:hover { border-color: #4f7180; background: #172a32; }
    .button-row { display: flex; gap: 8px; }
    .report-tabs { display: flex; gap: 5px; margin: 2px 0 18px; border-bottom: 1px solid var(--line); }
    .report-tabs button, .report-tabs a { min-height: 42px; display: inline-flex; align-items: center; padding: 9px 16px; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; color: var(--muted); font-weight: 700; }
    .report-tabs button.active { color: var(--accent); border-bottom-color: var(--accent); }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
    .panel { grid-column: span 12; border: 1px solid var(--line); border-radius: var(--radius); background: rgba(13,24,29,.94); box-shadow: var(--shadow); overflow: hidden; }
    .panel.half { grid-column: span 6; }
    .panel.third { grid-column: span 4; }
    .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; padding: 20px 22px 13px; }
    .panel-head h2 { margin: 0; font-size: 19px; letter-spacing: -.015em; }
    .panel-head p { margin: 4px 0 0; color: var(--muted); max-width: 860px; font-size: 13px; }
    .panel-controls { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 10px; align-items: end; }
    .panel-controls label { min-width: 170px; }
    .reference-note { margin-top: 10px; color: var(--faint); font-size: 11px; }
    .fixed-gradient { width: 150px; height: 7px; border-radius: 6px; background: linear-gradient(90deg, #7b2d3b 0%, #9a5b2f 28%, #9b8636 50%, #366f5c 72%, #2bd4ad 100%); }
    .uplift-gradient { width: 150px; height: 7px; border-radius: 6px; background: linear-gradient(90deg, #a23b4a 0%, #273a42 50%, #2bd4ad 100%); }
    .panel-body { padding: 0 22px 22px; }
    .tag { display: inline-flex; align-items: center; white-space: nowrap; padding: 4px 8px; border: 1px solid var(--line-2); border-radius: 999px; color: var(--muted); font-size: 11px; }
    .tag.accent { border-color: rgba(43,212,173,.45); color: var(--accent); background: rgba(43,212,173,.07); }
    .callout { padding: 14px 16px; border-left: 3px solid var(--warm); background: rgba(255,182,92,.07); color: #d8c5a8; border-radius: 5px 12px 12px 5px; }
    abbr.metric-name { color: inherit; text-decoration: underline dotted rgba(147,164,181,.65); text-underline-offset: 3px; cursor: help; }
    abbr.metric-name::after { content: ' ⓘ'; color: var(--accent-2); font-size: 1.08em; font-weight: 800; text-decoration: none; }
    abbr.metric-name:focus-visible { outline: 2px solid var(--accent-2); outline-offset: 3px; border-radius: 2px; }
    .info-tooltip-popover {
      position: fixed; z-index: 10000; max-width: min(360px, calc(100vw - 24px)); padding: 8px 10px;
      border: 1px solid var(--line-2); border-radius: 5px; background: var(--panel-2); color: var(--ink);
      box-shadow: 0 8px 24px rgba(0,0,0,.28); font: 11px/1.45 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      text-align: left; white-space: normal; pointer-events: auto;
    }
    .info-tooltip-popover[hidden] { display: none; }
    .info-tooltip-popover::before {
      content: ''; position: absolute; width: 8px; height: 8px; background: var(--panel-2);
      border-left: 1px solid var(--line-2); border-top: 1px solid var(--line-2); transform: rotate(45deg);
      top: -5px; left: var(--tooltip-arrow-left, 20px);
    }
    .info-tooltip-popover.above::before { top: auto; bottom: -5px; transform: rotate(225deg); }
    .chart { min-height: 340px; width: 100%; overflow: auto; }
    .chart svg { width: 100%; min-width: 680px; height: auto; display: block; }
    .compact-chart svg { width: min(100%, 820px); margin-inline: auto; }
    .axis { stroke: #35505c; stroke-width: 1; }
    .gridline { stroke: #1d3038; stroke-width: 1; stroke-dasharray: 3 5; }
    .axis-label { fill: #8396a5; font-size: 11px; }
    .point-label { fill: #c9d6df; font-size: 10px; }
    .empty { padding: 44px 20px; color: var(--muted); text-align: center; border: 1px dashed var(--line-2); border-radius: 12px; }
    .table-wrap { width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th { position: sticky; top: 0; z-index: 2; color: #a9bac6; background: #14242b; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
    th, td { padding: 9px 11px; border-bottom: 1px solid #1b2c34; white-space: nowrap; }
    tbody tr:hover { background: rgba(103,166,255,.045); }
    td.num, th.num { text-align: right; }
    td.subtle { color: var(--muted); }
    td.query { width: 38%; min-width: 300px; max-width: 560px; white-space: normal; vertical-align: top; }
    .query-text { display: -webkit-box; overflow: hidden; overflow-wrap: anywhere; white-space: pre-wrap; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; line-height: 1.42; }
    td.query.expanded .query-text { display: block; overflow: visible; -webkit-line-clamp: unset; line-clamp: unset; }
    .query-toggle { min-height: 0; margin-top: 3px; padding: 0; color: #8fc5ff; background: transparent; border: 0; border-radius: 3px; font-size: 11px; font-weight: 700; }
    .query-toggle:hover { color: var(--accent); background: transparent; border-color: transparent; text-decoration: underline; }
    .engine-cell b { display: block; }
    .engine-cell small { color: var(--faint); }
    .coverage { display: flex; align-items: center; gap: 8px; min-width: 110px; }
    .coverage-track { width: 62px; height: 5px; background: #23363e; border-radius: 4px; overflow: hidden; }
    .coverage-track i { display: block; height: 100%; background: var(--accent); }
    .best { color: var(--accent); font-weight: 800; }
    .frontier { color: var(--accent); }
    .heatmap th:first-child, .heatmap td:first-child { position: sticky; left: 0; z-index: 3; background: #122229; }
    .heatmap th:first-child { z-index: 4; }
    .heat-cell { min-width: 68px; text-align: center; font-size: 11px; border-left: 1px solid rgba(32,52,61,.7); }
    .heat-header { max-width: 95px; min-width: 75px; white-space: normal; line-height: 1.2; text-align: center; }
    .missing-cell { min-height: 45px; display: grid; place-items: center; border: 1px solid rgba(255,111,127,.25); border-radius: 7px; color: #ffc0c8; background: repeating-linear-gradient(135deg,rgba(255,111,127,.05),rgba(255,111,127,.05) 6px,rgba(255,111,127,.13) 6px,rgba(255,111,127,.13) 10px); font-size: 9px; font-weight: 800; letter-spacing: .03em; }
    .matrix-sort-button { min-height: 0; max-width: 100%; padding: 0; border: 0; border-radius: 3px; color: inherit; background: transparent; font: inherit; font-weight: 750; text-align: inherit; white-space: normal; }
    .matrix-sort-button:hover { color: var(--accent); background: transparent; border-color: transparent; text-decoration: underline; }
    .matrix-row-actions { display: flex; align-items: baseline; gap: 7px; }
    .matrix-axis-resets { display: flex; align-items: center; gap: 5px; white-space: nowrap; }
    .matrix-axis-resets button { min-height: 0; padding: 0; color: #a9bac6; background: transparent; border: 0; border-radius: 2px; font: inherit; text-transform: uppercase; letter-spacing: .08em; }
    .matrix-axis-resets button:hover { color: var(--accent); background: transparent; text-decoration: underline; }
    .matrix-axis-resets span { color: var(--faint); }
    .matrix-explore { min-height: 0; padding: 0; border: 0; color: var(--muted); background: transparent; font-size: 10px; }
    .matrix-explore:hover { color: var(--accent-2); background: transparent; }
    .legend { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 11px; }
    .gradient { width: 100px; height: 7px; border-radius: 6px; background: linear-gradient(90deg, #17262d, #155e59, #2bd4ad); }
    .footer { color: var(--faint); padding: 26px 2px; text-align: center; font-size: 12px; }
    .sort-button { min-height: 0; border: 0; padding: 0; border-radius: 0; color: inherit; background: none; text-transform: inherit; letter-spacing: inherit; font-size: inherit; }
    .sort-button:hover { background: none; text-decoration: underline; }
    .leaderboard-sort-button { min-height: 0; border: 0; padding: 0; border-radius: 0; color: inherit; background: none; font: inherit; font-weight: inherit; text-transform: inherit; letter-spacing: inherit; white-space: nowrap; }
    .leaderboard-sort-button:hover { color: var(--accent); background: none; text-decoration: underline; }
    .leaderboard-sort-button.active { color: var(--ink); }
    .leaderboard-sort-button .sort-indicator { display: inline-block; min-width: 1.1em; color: var(--accent); }
    .matrix-cell-button { width: 100%; min-height: 45px; padding: 5px; border: 1px solid transparent; border-radius: 7px; background: var(--cell-bg); color: var(--cell-fg, var(--ink)); }
    .matrix-cell-button:hover { border-color: var(--accent); filter: brightness(1.08); }
    .matrix-cell-button strong { display: block; font-variant-numeric: tabular-nums; }
    .matrix-cell-button small { display: flex; justify-content: center; gap: 5px; color: inherit; opacity: .76; }
    .mode-badge { display: inline-block; padding: 0 4px; border: 1px solid currentColor; border-radius: 4px; font-size: 9px; font-weight: 850; text-transform: uppercase; }
    .mode-fulltext { color: #ffcf66; }
    .mode-vector { color: #74b7ff; }
    .mode-hybrid { color: #62e0bd; }
    .dataset-link { min-height: 0; padding: 0; border: 0; background: transparent; text-align: left; color: var(--ink); font-weight: 750; }
    .dataset-link:hover { color: var(--accent); background: transparent; text-decoration: underline; }
    dialog { width: min(690px, calc(100% - 24px)); max-height: min(860px, calc(100vh - 30px)); padding: 0; color: var(--ink); background: var(--panel); border: 1px solid var(--line-2); border-radius: 18px; box-shadow: 0 35px 100px rgba(0,0,0,.6); }
    dialog::backdrop { background: rgba(2,8,11,.72); backdrop-filter: blur(4px); }
    .dialog-head { position: sticky; top: 0; z-index: 2; display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 18px 20px; background: var(--panel); border-bottom: 1px solid var(--line); }
    .dialog-head h2 { margin: 0; font-size: 19px; }
    .dialog-close { min-height: 32px; width: 32px; padding: 0; font-size: 20px; }
    .dialog-body { padding: 20px; }
    .detail-hero { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; margin-bottom: 18px; }
    .detail-score { padding: 18px; border: 1px solid rgba(43,212,173,.35); border-radius: 13px; background: rgba(43,212,173,.08); }
    .detail-score strong { display: block; margin-top: 4px; color: var(--accent); font-size: 34px; }
    .detail-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 15px 0; }
    .detail-tile { padding: 11px; border: 1px solid var(--line); border-radius: 9px; background: var(--panel-2); }
    .detail-tile span { color: var(--muted); font-size: 10px; text-transform: uppercase; }
    .detail-tile strong { display: block; margin-top: 5px; font-variant-numeric: tabular-nums; }
    .detail-kv { display: grid; grid-template-columns: 120px 1fr; gap: 7px 12px; margin: 16px 0; }
    .detail-kv dt { color: var(--muted); }
    .detail-kv dd { margin: 0; overflow-wrap: anywhere; }
    .run-tools { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
    .run-tools input { flex: 1; min-height: 39px; padding: 8px 11px; color: var(--ink); background: #132329; border: 1px solid var(--line-2); border-radius: 10px; }
    .pagination { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding-top: 12px; color: var(--muted); }
    @media (max-width: 1500px) {
      .filter-bar { position: static; grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .filter-bar > :first-child { grid-column: span 2; }
    }

    .performance-disclaimer {
      margin-top: 10px;
      padding: 8px 10px;
      border-left: 2px solid var(--warm);
      background: rgba(255, 182, 92, .06);
      color: var(--mut);
      font-size: 10.5px;
      line-height: 1.45;
    }
    .performance-disclaimer strong { color: var(--ink); }
    .panel-details { margin: 0; }
    .panel-details > summary {
      list-style: none;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding: 13px 14px 11px;
      border-bottom: 0;
      user-select: none;
    }
    .panel-details > summary::-webkit-details-marker { display: none; }
    .panel-details > summary::after {
      content: '▸';
      flex: 0 0 auto;
      margin-top: 1px;
      color: var(--mut);
      font: 700 13px/1 var(--mono);
      transition: transform .12s ease;
    }
    .panel-details[open] > summary { border-bottom: .5px solid var(--line); }
    .panel-details[open] > summary::after { transform: rotate(90deg); }
    .panel-details > summary h2 { margin: 0 0 3px; font-size: 19px; letter-spacing: -.015em; }
    .panel-details > summary p { margin: 0; color: var(--muted); max-width: 860px; font-size: 13px; }
    .panel-summary-main { min-width: 0; }
    .panel-summary-side { display: flex; align-items: center; gap: 10px; }
    @media (max-width: 980px) {
      .filter-bar { position: static; grid-template-columns: 1fr 1fr; }
      .filter-bar > :first-child { grid-column: span 2; }
      .panel.half, .panel.third { grid-column: span 12; }
      .panel-head { flex-direction: column; }
      .panel-controls { width: 100%; justify-content: flex-start; }
      header { grid-template-columns: 1fr; }
      .stamp { text-align: left; }
    }
    @media (max-width: 560px) {
      .shell { width: min(100% - 18px, 1560px); }
      header { padding-top: 32px; }
      .filter-bar { grid-template-columns: 1fr; }
      .filter-bar > :first-child { grid-column: auto; }
      .panel-head, .panel-body { padding-left: 14px; padding-right: 14px; }
    }
    @media print {
      body { background: white; color: #17242a; }
      .shell { width: 100%; }
      .filter-bar, .button-row { display: none; }
      .panel { box-shadow: none; break-inside: avoid; background: white; border-color: #ccd6db; }
      .panel.half { grid-column: span 12; }
      .muted, .panel-head p { color: #53666f; }
    }
  

    /* ------------------------------------------------------------------
       Benchmark UI restyle — compact, technical, flat.
       Inspired by the visual language of SereneDB SearchBench and ClickBench,
       without copying either brand verbatim.
       ------------------------------------------------------------------ */
    :root, :root[data-theme="dark"] {
      --ink: #f4f4f5;
      --muted: #b9b9bf;
      --faint: #7f8088;
      --page: #1c1c1c;
      --panel: #212121;
      --panel-2: #181818;
      --head: #262626;
      --inset: #151515;
      --stripe: rgba(255,255,255,.022);
      --line: rgba(255,255,255,.13);
      --line-2: rgba(255,255,255,.20);
      --accent: #895af8;
      --accent-2: #3386ff;
      --link: #80beff;
      --warm: #ffb454;
      --danger: #e06c9f;
      --positive: #5ed29a;
      --heat-green-fg: #081c13;
      --heat-cyan-fg: #071819;
      --heat-yellow-fg: #1d1803;
      --heat-orange-fg: #251402;
      --heat-red-fg: #fff1f6;
      --uplift-positive-fg: #d9f5e6;
      --uplift-negative-fg: #fff0f6;
      --shadow: none;
      --radius: 2px;
      --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
    }
    :root[data-theme="light"] {
      --ink: #11121d;
      --muted: #5b6373;
      --faint: #8c93a0;
      --page: #ffffff;
      --panel: #ffffff;
      --panel-2: #f3f4f6;
      --head: #f7f7f8;
      --inset: #f0f1f3;
      --stripe: #fafafa;
      --line: #e1e2e6;
      --line-2: #c9ccd3;
      --accent: #7450d8;
      --accent-2: #166de2;
      --link: #176bb5;
      --warm: #c8781a;
      --danger: #b84570;
      --positive: #167a50;
      --heat-green-fg: #0b2a1b;
      --heat-cyan-fg: #0b2427;
      --heat-yellow-fg: #2c2506;
      --heat-orange-fg: #382105;
      --heat-red-fg: #3a0e1f;
      --uplift-positive-fg: #0e4d32;
      --uplift-negative-fg: #6b1738;
      color-scheme: light;
    }
    html { background: var(--page); }
    body {
      background: var(--page);
      color: var(--ink);
      font: 13px/1.42 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-synthesis: none;
    }
    body, button, select, input { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    *::-webkit-scrollbar { width: 10px; height: 10px; }
    *::-webkit-scrollbar-thumb { background: var(--line-2); border: 3px solid transparent; background-clip: padding-box; }
    *::-webkit-scrollbar-track { background: transparent; }
    a { color: var(--link); }
    a:hover { color: var(--accent); text-decoration: none; }

    .shell { width: min(1540px, calc(100% - 28px)); padding-bottom: 28px; }
    .site-header {
      padding: 22px 0 14px;
      margin: 0;
      border-bottom: .5px solid var(--line);
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
    }
    .positioning {
      margin-top: 7px;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.35;
      font-weight: 700;
      letter-spacing: .005em;
    }
    h1 {
      margin: 0;
      font-size: clamp(25px, 3vw, 38px);
      line-height: 1.05;
      letter-spacing: -.035em;
      font-weight: 760;
    }
    .byline {
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      white-space: nowrap;
    }
    .byline:hover { color: var(--accent); }

    .header-links { display:flex; align-items:center; gap:6px; white-space:nowrap; }
    .header-link-sep { color:var(--faint); font:10px/1 var(--mono); }
    .utility-link { color:var(--link); font-size:10px; font-weight:700; white-space:nowrap; }
    .utility-link:hover { color:var(--accent); }
    .lede { max-width: 850px; margin-top: 7px; color: var(--muted); font-size: 12.5px; }
    .header-meta { margin-top: 7px; color: var(--faint); font: 10px/1.3 var(--mono); }
    .header-actions { display: flex; align-items: center; gap: 12px; padding: 0 44px 1px 0; }
    .stamp { color: var(--faint); text-align: right; font: 10px/1.35 var(--mono); white-space: nowrap; }
    .theme-toggle {
      position: fixed;
      top: 12px;
      right: 12px;
      z-index: 80;
      display: inline-grid;
      place-items: center;
      width: 34px;
      height: 34px;
      padding: 0;
      border: .5px solid var(--line-2);
      border-radius: 6px;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      color: var(--muted);
      box-shadow: 0 1px 5px rgba(0,0,0,.14);
      backdrop-filter: blur(10px);
      cursor: pointer;
    }
    .theme-toggle:hover { color: var(--ink); border-color: var(--accent); background: var(--head); }
    .theme-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .theme-toggle svg { width: 16px; height: 16px; display: block; }
    .theme-icon-moon { display: none !important; }
    html[data-theme="light"] .theme-icon-sun { display: none !important; }
    html[data-theme="light"] .theme-icon-moon { display: block !important; }

    .filter-bar {
      position: sticky;
      top: 0;
      z-index: 20;
      grid-template-columns: auto minmax(210px, 1.35fr) minmax(165px, 1fr) minmax(150px, .9fr) auto;
      gap: 7px;
      align-items: end;
      padding: 7px 0;
      margin: 0 0 8px;
      border: 0;
      border-bottom: .5px solid var(--line);
      border-radius: 0;
      background: color-mix(in srgb, var(--page) 94%, transparent);
      backdrop-filter: blur(12px);
      box-shadow: none;
    }
    .filter-caption {
      align-self: center;
      padding-right: 3px;
      color: var(--accent);
      font: 700 9.5px/1 var(--mono);
      letter-spacing: .08em;
    }
    label, .filter-field {
      gap: 3px;
      color: var(--muted);
      font-size: 9px;
      line-height: 1.1;
      letter-spacing: .075em;
      font-weight: 750;
    }
    select, button, .multi-select summary, .run-tools input {
      min-height: 30px;
      border: .5px solid var(--line-2);
      border-radius: 3px;
      background: var(--inset);
      color: var(--ink);
      padding: 5px 8px;
      outline: none;
      box-shadow: none;
    }
    select { padding-right: 25px; }
    button { font-weight: 680; }
    button:hover, select:hover, .multi-select summary:hover { border-color: var(--accent); background: var(--head); }
    select:focus, button:focus, .multi-select[open] summary, .run-tools input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 55%, transparent);
    }
    .multi-select summary { font-size: 11px; font-weight: 560; }
    .multi-select summary::after { color: var(--accent); font-size: 10px; }
    .multi-menu {
      top: calc(100% + 4px);
      padding: 6px;
      color: var(--ink);
      background: var(--panel);
      border: .5px solid var(--line-2);
      border-radius: 2px;
      box-shadow: 8px 10px 0 color-mix(in srgb, var(--ink) 9%, transparent);
    }
    .multi-actions { gap: 4px; padding-bottom: 5px; border-color: var(--line); }
    .multi-actions button { min-height: 25px; padding: 3px 7px; font-size: 9.5px; }
    .multi-options { gap: 0; padding-top: 4px; }
    label.multi-option { padding: 4px 5px; border-radius: 0; font-size: 10.5px; }
    label.multi-option:hover { background: var(--inset); }
    .multi-option input { accent-color: var(--accent); }
    .multi-option small { color: var(--faint); font-family: var(--mono); font-size: 9px; }
    .button-row { gap: 5px; }

    .report-tabs {
      gap: 2px;
      margin: 0 0 10px;
      padding: 0 0 7px;
      border-bottom: .5px solid var(--line);
    }
    .report-tabs button, .report-tabs a {
      min-height: 29px;
      padding: 5px 10px;
      border: .5px solid transparent;
      border-radius: 3px;
      color: var(--muted);
      background: transparent;
      font-size: 10.5px;
      font-weight: 700;
    }
    .report-tabs button:hover, .report-tabs a:hover { color: var(--ink); background: var(--inset); }
    .report-tabs button.active {
      color: #fff;
      background: var(--accent);
      border-color: var(--accent);
    }
    :root[data-theme="light"] .report-tabs button.active { color: #fff; }
    .report-tabs a { margin-left: auto; color: var(--link); }

    .grid { gap: 10px; }
    .panel {
      position: relative;
      border: .5px solid var(--line);
      border-radius: 0;
      background: var(--panel);
      box-shadow: none;
      overflow: visible;
    }
    .panel::after {
      content: "";
      position: absolute;
      z-index: -1;
      left: 4px; top: 4px; right: -4px; bottom: -4px;
      background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--ink) 7%, transparent) 0 1px, transparent 1px 4px);
      pointer-events: none;
    }
    .panel-head {
      align-items: flex-start;
      padding: 7px 9px;
      border-bottom: .5px solid var(--line);
      background: var(--head);
    }
    .panel-head h2 {
      margin: 0;
      color: var(--muted);
      font-size: 10.5px;
      line-height: 1.25;
      font-weight: 780;
      letter-spacing: .085em;
      text-transform: uppercase;
    }
    .panel-head h2::before { content: "●"; margin-right: 6px; color: var(--accent); font-size: 8px; vertical-align: 1px; }
    .panel-head p { margin: 3px 0 0; color: var(--faint); max-width: 980px; font-size: 10.5px; line-height: 1.42; }
    .panel-body { padding: 8px 9px 9px; }
    .panel-controls { gap: 7px; }
    .panel-controls label { min-width: 135px; }
    .panel-controls select { font-size: 10.5px; }
    .tag {
      padding: 2px 6px;
      border: .5px solid var(--line-2);
      border-radius: 2px;
      color: var(--muted);
      background: var(--inset);
      font: 9.5px/1.4 var(--mono);
    }
    .tag.accent { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); background: color-mix(in srgb, var(--accent) 8%, transparent); }
    .reference-note, .matrix-note {
      margin-top: 6px;
      padding: 5px 7px;
      border-left: 2px solid var(--accent);
      border-radius: 0;
      background: var(--inset);
      color: var(--faint);
      font-size: 9.5px;
      line-height: 1.4;
    }
    .callout { border-radius: 0; background: var(--inset); border-left-color: var(--warm); color: var(--muted); }

    .table-wrap { border: .5px solid var(--line); border-radius: 0; background: var(--panel); }
    table { font-size: 11px; line-height: 1.3; }
    th {
      background: var(--head);
      color: var(--muted);
      font-size: 9.5px;
      line-height: 1.25;
      font-weight: 760;
      letter-spacing: .065em;
    }
    th, td { padding: 5px 7px; border-bottom: .5px solid var(--line); }
    tbody tr:nth-child(even) { background: var(--stripe); }
    tbody tr:hover { background: color-mix(in srgb, var(--accent) 6%, var(--panel)); }
    td.num, th.num, .detail-tile strong, .detail-score strong, .coverage, .matrix-cell-button strong, .matrix-cell-button small {
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }
    td.subtle, .subtle { color: var(--faint); }
    .engine-cell b { font-weight: 740; }
    .engine-cell small { font: 9px/1.3 var(--mono); color: var(--faint); }
    .best { color: var(--positive); }
    .frontier { color: var(--accent); }
    .coverage-track { height: 4px; background: var(--inset); border-radius: 0; }
    .coverage-track i { background: var(--accent); }

    .heatmap th:first-child, .heatmap td:first-child { background: var(--head); }
    .heatmap td:first-child { background: var(--panel); }
    .heat-cell { min-width: 67px; border-left: .5px solid var(--line); font-size: 10px; }
    .heat-header { min-width: 78px; max-width: 104px; }
    .matrix-cell-button {
      min-height: 38px;
      border-radius: 0;
      border: .5px solid color-mix(in srgb, var(--line) 70%, transparent);
      padding: 4px;
    }
    .matrix-cell-button:hover { border-color: var(--accent); filter: none; box-shadow: inset 0 0 0 1px var(--accent); }
    .matrix-cell-button.matrix-winner {
      border: 1.5px solid var(--accent);
      box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 32%, transparent);
    }
    .matrix-cell-button.matrix-winner:hover {
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .matrix-cell-button strong { font-size: 10.5px; }
    .matrix-cell-button small { font-size: 8.5px; }
    .matrix-sort-button, .dataset-link { font-size: 10.5px; font-weight: 680; }
    .matrix-explore { color: var(--link); font-size: 9px; }
    .matrix-axis-resets button { color: var(--muted); font-size: 9px; }
    .missing-cell {
      min-height: 38px;
      border-radius: 0;
      border-color: color-mix(in srgb, var(--danger) 35%, transparent);
      color: var(--danger);
      background: repeating-linear-gradient(135deg, transparent, transparent 5px, color-mix(in srgb, var(--danger) 9%, transparent) 5px, color-mix(in srgb, var(--danger) 9%, transparent) 8px);
      font: 700 8.5px/1 var(--mono);
    }
    .mode-badge {
      border: 0;
      border-radius: 1px;
      padding: 0 3px;
      background: color-mix(in srgb, currentColor 13%, transparent);
      font: 800 8px/1.35 var(--mono);
    }
    .mode-fulltext { color: #ffb454; }
    .mode-vector { color: #80beff; }
    .mode-hybrid { color: #5ed29a; }
    .fixed-gradient { height: 6px; border-radius: 0; background: linear-gradient(90deg, #b84570 0%, #c8781a 30%, #b4a03d 52%, #3f8f71 76%, #5ed29a 100%); }
    .uplift-gradient { height: 6px; border-radius: 0; background: linear-gradient(90deg, #e06c9f 0%, var(--inset) 50%, #5ed29a 100%); }

    .chart { min-height: 310px; }
    .axis { stroke: var(--line-2); stroke-width: .8; }
    .gridline { stroke: var(--line); stroke-width: .7; stroke-dasharray: 2 4; }
    .axis-label { fill: var(--faint); font-family: var(--mono); font-size: 9.5px; }
    .point-label { fill: var(--muted); font-family: var(--mono); font-size: 8.5px; }
    .legend { color: var(--faint); font-size: 9.5px; }

    .run-tools { gap: 6px; }
    .run-tools input { background: var(--inset); font-size: 10.5px; }
    .pagination { color: var(--faint); font: 9.5px/1.4 var(--mono); }
    .query-toggle { color: var(--link); font-size: 9.5px; }

    dialog {
      border: .5px solid var(--line-2);
      border-radius: 2px;
      background: var(--panel);
      box-shadow: 10px 12px 0 color-mix(in srgb, #000 35%, transparent);
    }
    dialog::backdrop { background: rgba(0,0,0,.66); backdrop-filter: none; }
    .dialog-head { padding: 9px 11px; border-bottom: .5px solid var(--line); background: var(--head); }
    .dialog-head h2 { font-size: 13px; letter-spacing: 0; }
    .dialog-close { min-height: 26px; width: 26px; font-size: 16px; }
    .dialog-body { padding: 10px; }
    .detail-hero { gap: 7px; margin-bottom: 9px; }
    .detail-score { padding: 10px; border: .5px solid var(--line-2); border-radius: 0; background: var(--inset); }
    .detail-score strong { margin-top: 2px; color: var(--accent); font-size: 25px; }
    .detail-grid { gap: 5px; margin: 8px 0; }
    .detail-tile { padding: 7px; border: .5px solid var(--line); border-radius: 0; background: var(--panel-2); }
    .detail-tile span { font-size: 8.5px; }
    .detail-kv { gap: 5px 10px; margin: 10px 0; font-size: 10.5px; }

    .footer { padding: 16px 0 4px; color: var(--faint); border-top: .5px solid var(--line); text-align: left; font: 9.5px/1.4 var(--mono); }

    @media (max-width: 1500px) {
      .filter-bar { position: static; grid-template-columns: auto repeat(2, minmax(0, 1fr)); }
      .filter-caption { grid-row: span 2; align-self: start; padding-top: 13px; }
      .filter-bar > :nth-child(2) { grid-column: span 2; }
    }
    @media (max-width: 980px) {
      .site-header { grid-template-columns: 1fr; gap: 9px; }
      .header-actions { justify-content: space-between; padding-right: 0; }
      .stamp { text-align: left; }
      .filter-bar { grid-template-columns: 1fr 1fr; }
      .filter-caption { display: none; }
      .filter-bar > :nth-child(2) { grid-column: span 2; }
      .report-tabs a { margin-left: 0; }
    }
    @media (max-width: 560px) {
      .shell { width: calc(100% - 14px); }
      .site-header { padding-top: 14px; padding-right: 34px; }
      .theme-toggle { top: 9px; right: 9px; width: 32px; height: 32px; }
      .filter-bar { grid-template-columns: 1fr; }
      .filter-bar > :nth-child(2) { grid-column: auto; }
      .report-tabs { overflow-x: auto; white-space: nowrap; }
      .panel-head, .panel-body { padding-left: 7px; padding-right: 7px; }
    }
    @media print {
      :root, :root[data-theme="dark"] { --page:#fff; --panel:#fff; --head:#f7f7f7; --ink:#111; --muted:#555; --faint:#777; --line:#ccc; --line-2:#aaa; }
      .theme-toggle, .filter-bar, .button-row { display:none !important; }
      .panel::after { display:none; }
    }



    .quality-toolbar {
      display:flex; align-items:end; justify-content:space-between; gap:16px;
      margin:0 0 9px; padding:9px 10px;
      border:.5px solid var(--line); background:var(--panel);
    }
    .quality-toolbar-copy { display:flex; align-items:baseline; gap:9px; min-width:0; }
    .quality-toolbar-copy > span { color:var(--accent); font:700 9px/1 var(--mono); letter-spacing:.09em; }
    .quality-toolbar-copy > strong { color:var(--ink); font-size:11px; font-weight:720; }
    .quality-toolbar-copy > small { color:var(--faint); font-size:10px; }
    .quality-toolbar label { min-width:260px; }
    .quality-toolbar select { width:100%; }
    @media (max-width: 720px) {
      .quality-toolbar { align-items:stretch; flex-direction:column; gap:7px; }
      .quality-toolbar-copy { align-items:flex-start; flex-wrap:wrap; }
      .quality-toolbar label { min-width:0; width:100%; }
    }

    /* ---- search mode comparison ---- */
    .mode-legend { display:flex; align-items:center; gap:13px; flex-wrap:wrap; color:var(--muted); font-size:10px; }
    .mode-legend span { display:inline-flex; align-items:center; gap:5px; }
    .mode-symbol { width:9px; height:9px; display:inline-block; border:1.5px solid var(--muted); background:var(--panel); }
    .mode-symbol.ft { border-radius:50%; }
    .mode-symbol.v { transform:rotate(45deg); }
    .mode-symbol.h { border-radius:1px; background:var(--muted); }
    .mode-comparison-chart { min-height:280px; width:100%; overflow:auto; }
    .mode-comparison-chart svg { width:100%; min-width:760px; height:auto; display:block; }


    /* ---- overall quality consistency ---- */
    .quality-floor { min-width:108px; }
    .quality-floor strong { display:block; font:800 11px/1.15 var(--mono); }
    .quality-floor small { display:block; margin-top:2px; color:var(--faint); font-size:9px; white-space:normal; line-height:1.25; }
    .consistency-chart { width:100%; min-height:260px; overflow:auto; }
    .consistency-chart svg { width:100%; min-width:760px; height:auto; display:block; }
    .consistency-legend { display:flex; align-items:center; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:10px; }
    .consistency-legend span { display:inline-flex; align-items:center; gap:6px; }
    .consistency-dot, .consistency-floor, .consistency-mean { display:inline-block; width:9px; height:9px; }
    .consistency-dot { border-radius:50%; background:var(--muted); opacity:.7; }
    .consistency-floor { border-radius:50%; border:1.5px solid var(--accent); background:transparent; }
    .consistency-mean { transform:rotate(45deg); background:var(--accent-2); }



    /* Dataset explorer: persistent dataset navigation */
    .dataset-switcher {
      position: sticky;
      top: var(--dataset-switcher-top, 0px);
      z-index: 19;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      min-height: 52px;
      padding: 7px 0 8px;
      margin: 0 0 12px;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--page) 96%, transparent);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }
    .dataset-switcher[hidden] { display: none; }
    .dataset-switcher-copy { display: flex; align-items: baseline; gap: 9px; min-width: 0; }
    .dataset-switcher-copy span {
      color: var(--faint);
      font: 700 9px/1.2 var(--mono);
      letter-spacing: .12em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .dataset-switcher-copy strong { color: var(--ink); font-size: 12px; white-space: nowrap; }
    .dataset-switcher-controls { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .dataset-picker { display: block; min-width: 300px; width: min(46vw, 520px); }
    .dataset-picker select { width: 100%; min-height: 34px; padding-top: 5px; padding-bottom: 5px; }
    .dataset-nav-button {
      display: inline-grid;
      place-items: center;
      width: 34px;
      min-width: 34px;
      height: 34px;
      min-height: 34px;
      padding: 0;
      border-radius: 7px;
      font: 800 18px/1 var(--mono);
    }
    .dataset-nav-button:disabled { opacity: .32; cursor: default; border-color: var(--line); background: transparent; }
    .dataset-nav-button:disabled:hover { border-color: var(--line); background: transparent; }
    .dataset-position {
      min-width: 45px;
      color: var(--faint);
      font: 700 10px/1 var(--mono);
      text-align: right;
      white-space: nowrap;
    }
    @media (max-width: 760px) {
      .dataset-switcher { align-items: stretch; gap: 7px; }
      .dataset-switcher-copy strong { display: none; }
      .dataset-switcher-copy { align-items: center; }
      .dataset-switcher-controls { flex: 1; }
      .dataset-picker { flex: 1; width: auto; min-width: 0; }
      .dataset-position { display: none; }
    }

  </style>
</head>
<body>
  <main class="shell">
    <header class="site-header">
      <div class="title-block">
        <h1>Search Quality Benchmark</h1>
        <div class="positioning">Open-source search engines. Public datasets. Quality first.</div>
        <div class="lede">Compare full-text, vector, and hybrid search across open-source engines and public datasets. Relevance is the primary measure; latency and throughput provide additional context.</div>
        <div id="header-meta" class="header-meta"></div>
      </div>
      <div class="header-actions">
        <div class="header-links">
          <a class="byline" href="https://manticoresearch.com/" target="_blank" rel="noreferrer">by Manticore ↗</a>
          <span class="header-link-sep">·</span>
          <a class="utility-link" href="https://github.com/manticoresoftware/search-bench" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
        <div class="stamp"><span id="generated-at"></span></div>
      </div>
    </header>

    <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Switch to light theme" title="Switch to light theme">
      <svg class="theme-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="3.6"></circle>
        <path d="M12 2.2v2M12 19.8v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2.2 12h2M19.8 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>
      </svg>
      <svg class="theme-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20.3 15.2A8.4 8.4 0 0 1 8.8 3.7 8.7 8.7 0 1 0 20.3 15.2Z"></path>
      </svg>
    </button>

    <section class="filter-bar" aria-label="Report filters"><div class="filter-caption">FILTERS</div>
      <div class="filter-field">Datasets<details class="multi-select" id="dataset-filter"><summary><span id="dataset-summary" class="multi-summary">All datasets</span></summary><div class="multi-menu"><div class="multi-actions"><button type="button" data-multi="datasets" data-choice="all">All</button><button type="button" data-multi="datasets" data-choice="none">Clear</button></div><div id="dataset-options" class="multi-options"></div></div></details></div>
      <div class="filter-field">Engines<details class="multi-select" id="engine-filter"><summary><span id="engine-summary" class="multi-summary">All engines</span></summary><div class="multi-menu"><div class="multi-actions"><button type="button" data-multi="engines" data-choice="all">All</button><button type="button" data-multi="engines" data-choice="none">Clear</button></div><div id="engine-options" class="multi-options"></div></div></details></div>
      <div class="filter-field">Search modes<details class="multi-select" id="mode-filter"><summary><span id="mode-summary" class="multi-summary">All modes</span></summary><div class="multi-menu"><div class="multi-actions"><button type="button" data-multi="modes" data-choice="all">All</button><button type="button" data-multi="modes" data-choice="none">Clear</button></div><div id="mode-options" class="multi-options"></div></div></details></div>
      <select id="metric-filter" hidden aria-hidden="true"><option value="canonical">Recommended for each dataset</option><option value="ndcg10">Ranking quality · top 10 (NDCG@10)</option><option value="ndcg20">Ranking quality · top 20 (NDCG@20)</option><option value="mrr10">First useful result · top 10 (MRR@10)</option><option value="map10">Relevant results near top (MAP@10)</option><option value="p20">Useful share of top 20 (P@20)</option><option value="recall100">Relevant-result coverage (Recall@100)</option></select>
      <div class="button-row"><button id="reset-button" type="button">Reset</button><button id="export-button" type="button">Export CSV</button></div>
    </section>

    <nav class="report-tabs" aria-label="Report views">
      <button type="button" data-tab="overview" class="active">Overview</button>
      <button type="button" data-tab="dataset">Dataset explorer</button>
      <button type="button" data-tab="runs">All runs</button>
      <a href="https://github.com/manticoresoftware/search-bench#benchmark-report-methodology" target="_blank" rel="noreferrer">Methodology ↗</a>
    </nav>

    <section class="dataset-switcher" data-view="dataset" hidden aria-label="Dataset navigation">
      <div class="dataset-switcher-copy"><span>DATASET EXPLORER</span></div>
      <div class="dataset-switcher-controls">
        <button id="focus-prev" class="dataset-nav-button" type="button" aria-label="Previous dataset" title="Previous dataset">←</button>
        <label class="dataset-picker" aria-label="Focused dataset"><select id="focus-dataset"></select></label>
        <button id="focus-next" class="dataset-nav-button" type="button" aria-label="Next dataset" title="Next dataset">→</button>
        <span id="focus-position" class="dataset-position" aria-live="polite"></span>
      </div>
    </section>

    <section class="quality-toolbar" data-view="overview" aria-label="Quality controls">
      <div class="quality-toolbar-copy"><span>QUALITY</span><strong>Comparison metric</strong><small>Applies to all quality views below.</small></div>
      <label><abbr class="metric-name" tabindex="0" role="button" aria-label="Quality metric: Recommended for each dataset uses that dataset’s canonical quality metric: NDCG@10 for most datasets and MRR@10 for MS MARCO and TREC-DL. Choosing another option applies that metric across all selected datasets." title="Recommended for each dataset uses that dataset’s canonical quality metric: NDCG@10 for most datasets and MRR@10 for MS MARCO and TREC-DL. Choosing another option applies that metric across all selected datasets.">Quality metric</abbr><select class="metric-filter-sync" data-metric-filter><option value="canonical">Recommended for each dataset</option><option value="ndcg10">Ranking quality · top 10 (NDCG@10)</option><option value="ndcg20">Ranking quality · top 20 (NDCG@20)</option><option value="mrr10">First useful result · top 10 (MRR@10)</option><option value="map10">Relevant results near top (MAP@10)</option><option value="p20">Useful share of top 20 (P@20)</option><option value="recall100">Relevant-result coverage (Recall@100)</option></select></label>
    </section>

    <div class="grid">
      <section class="panel" data-view="overview">
        <div class="panel-head"><div><h2>Overall quality</h2><p>Mean relative shows average quality across the selected datasets. Quality floor shows the engine’s worst relative result: how close it stays to the best observed result even on its weakest dataset. Engines missing a selected dataset remain visible but are not ranked.</p></div><span id="leaderboard-scope" class="tag"></span></div>
        <div class="panel-body"><div id="leaderboard"></div><div class="reference-note">Reference values are computed from all benchmarked engines for the selected datasets and modes, including engines currently hidden by the engine filter. Higher is better for both Mean relative and Quality floor.</div></div>
      </section>

      <section class="panel" data-view="overview">
        <div class="panel-head"><div><h2>Quality consistency across datasets</h2><p>Each small dot is one selected dataset, normalized to the best observed result on that dataset. The ring marks the engine’s weakest dataset (Quality floor); the diamond marks Mean relative. A tight cluster near 100% means consistently strong quality without large workload-specific drops. Engines with partial coverage are labeled with their dataset count and use a hollow mean marker.</p></div><div class="consistency-legend"><span><i class="consistency-dot"></i> Dataset</span><span><i class="consistency-floor"></i> Quality floor</span><span><i class="consistency-mean"></i> Mean relative</span></div></div>
        <div class="panel-body"><div id="consistency-chart" class="consistency-chart"></div></div>
      </section>

      <section id="quality-latency-panel" class="panel" data-view="dataset" hidden>
        <div class="panel-head"><div><h2>Quality × latency</h2><p>Use latency to understand the cost of search quality on this dataset. Performance is supporting context only and does not affect benchmark ranking.</p></div><div class="panel-controls"><label><abbr class="metric-name" tabindex="0" role="button" aria-label="Quality metric: Recommended for each dataset uses that dataset’s canonical quality metric: NDCG@10 for most datasets and MRR@10 for MS MARCO and TREC-DL. Choosing another option applies that metric across all selected datasets." title="Recommended for each dataset uses that dataset’s canonical quality metric: NDCG@10 for most datasets and MRR@10 for MS MARCO and TREC-DL. Choosing another option applies that metric across all selected datasets.">Quality metric</abbr><select class="metric-filter-sync" data-metric-filter><option value="canonical">Recommended for each dataset</option><option value="ndcg10">Ranking quality · top 10 (NDCG@10)</option><option value="ndcg20">Ranking quality · top 20 (NDCG@20)</option><option value="mrr10">First useful result · top 10 (MRR@10)</option><option value="map10">Relevant results near top (MAP@10)</option><option value="p20">Useful share of top 20 (P@20)</option><option value="recall100">Relevant-result coverage (Recall@100)</option></select></label><label>Points<select id="scatter-mode-filter"><option value="all">All modes</option><option value="best" selected>Best mode per engine</option></select></label><label>Context metric<select id="performance-filter"><option value="p50">P50 latency</option><option value="p99">P99 latency</option><option value="qps">QPS · informational</option></select></label></div></div>
        <div class="panel-body"><div id="scatter" class="chart"></div><div class="performance-disclaimer"><strong>Performance note.</strong> This benchmark is designed primarily to compare search quality. Latency and QPS were measured with a lightweight methodology and may be noisy; use them only as directional context, not as an authoritative performance comparison.</div><div class="reference-note">QPS is the observed query rate at concurrency 1 and should not be read as maximum throughput.</div></div>
      </section>

      <section class="panel" data-view="overview">
        <div class="panel-head"><div><h2>Dataset × engine quality matrix</h2><p>Each cell uses the best result among the selected search modes. “Per-dataset relative” is the engine score divided by the best observed score in that dataset. Color uses a fixed scale, so the same percentage always has the same color across filters and screenshots.</p></div><div class="panel-controls"><label>Matrix values<select id="value-filter"><option value="relative">Per-dataset relative</option><option value="absolute">Absolute metric</option></select></label><div class="legend"><span>&lt;90%</span><i class="fixed-gradient"></i><span>100%</span></div></div></div>
        <div class="panel-body"><div id="heatmap"></div></div>
      </section>

      <section class="panel" data-view="overview">
        <div class="panel-head"><div><h2>Search mode comparison</h2><p>Compares Full-text, Vector, and Hybrid on one normalized cross-dataset scale. Each point is that engine/mode’s mean per-dataset quality relative to the best observed result in the same dataset. Lines connect the modes in the order Full-text → Vector → Hybrid.</p></div><div class="mode-legend"><span><i class="mode-symbol ft"></i> Full-text</span><span><i class="mode-symbol v"></i> Vector</span><span><i class="mode-symbol h"></i> Hybrid</span></div></div>
        <div class="panel-body"><div id="mode-profile" class="mode-comparison-chart"></div><div class="reference-note">Filled markers have complete coverage for the selected datasets. Hollow markers are partial and include their coverage in the tooltip.</div></div>
      </section>

      <section class="panel" data-view="dataset" hidden>
        <div class="panel-head"><div><h2>Latency shape</h2><p>The three points show typical latency (P50), slower requests (P95), and tail latency (P99). A short line means consistent response times. The horizontal scale is logarithmic; lower is better.</p></div></div>
        <div class="panel-body"><div id="latency" class="chart latency-chart"></div><div class="performance-disclaimer"><strong>Performance note.</strong> This benchmark is designed primarily to compare search quality. Latency and QPS were measured with a lightweight methodology and may be noisy; use them only as directional context, not as an authoritative performance comparison.</div></div>
      </section>

      <section class="panel" data-view="dataset" hidden>
        <details class="panel-details">
          <summary>
            <div class="panel-summary-main"><h2>Slow-query inspection</h2><p>Eight independent slowest successful query samples from every run matching the focused dataset, engine and search-mode filters, ordered by latency.</p></div>
            <div class="panel-summary-side"><span id="query-count" class="tag"></span></div>
          </summary>
          <div class="panel-body"><div class="performance-disclaimer"><strong>Performance note.</strong> This benchmark is designed primarily to compare search quality. Latency and QPS were measured with a lightweight methodology and may be noisy; use them only as directional context, not as an authoritative performance comparison.</div><div id="slow-queries"></div></div>
        </details>
      </section>

      <section class="panel" data-view="runs" hidden>
        <div class="panel-head"><div><h2>Run explorer</h2><p>Inspect the underlying measured runs. Quality is the benchmark target; latency and QPS are included as supporting context. The filters above remain active.</p></div><span id="run-count" class="tag"></span></div>
        <div class="panel-body"><div class="performance-disclaimer"><strong>Performance note.</strong> This benchmark is designed primarily to compare search quality. Latency and QPS were measured with a lightweight methodology and may be noisy; use them only as directional context, not as an authoritative performance comparison.</div><div class="run-tools"><input id="run-search" type="search" placeholder="Search engine, dataset, profile or model"><button id="export-json-button" type="button">Export JSON</button></div><div id="run-table"></div></div>
      </section>
    </div>
  </main>

  <dialog id="run-dialog"><div id="run-dialog-content"></div></dialog>

  <script id="report-data" type="application/json">__DATA_JSON__</script>
  <script>
  (() => {
    'use strict';
    const DATA = JSON.parse(document.getElementById('report-data').textContent);
    const $ = (selector) => document.querySelector(selector);
    const COLORS = {
      columnar: '#895af8', elasticsearch: '#ffb454', mariadb: '#e06c9f', meilisearch: '#b48ead',
      milvus: '#5ed29a', opensearch: '#7fd7e0', qdrant: '#3386ff', typesense: '#d0c05a',
      vespa: '#9aa7ff', weaviate: '#ff7b6b'
    };
    const ALL_DATASET_KEYS = DATA.datasets.map(d => d.key);
    const ALL_ENGINE_KEYS = Object.keys(DATA.systems);
    const ALL_MODE_KEYS = ['fulltext', 'vector', 'hybrid'];
    const MODE_NAMES = { fulltext: 'Full text', vector: 'Vector', hybrid: 'Hybrid' };
    const ALL_FILTER_VALUES = { datasets: ALL_DATASET_KEYS, engines: ALL_ENGINE_KEYS, modes: ALL_MODE_KEYS };
    const validDefaults = (values, allowed) => (values || []).filter(value => allowed.includes(value));
    const DEFAULT_DATASET_KEYS = validDefaults(DATA.meta.defaultDatasets, ALL_DATASET_KEYS);
    const DEFAULT_ENGINE_KEYS = validDefaults(DATA.meta.defaultEngines, ALL_ENGINE_KEYS);
    const DEFAULT_MODE_KEYS = validDefaults(DATA.meta.defaultModes, ALL_MODE_KEYS);
    if (!DEFAULT_DATASET_KEYS.length) DEFAULT_DATASET_KEYS.push(...ALL_DATASET_KEYS);
    if (!DEFAULT_ENGINE_KEYS.length) DEFAULT_ENGINE_KEYS.push(...ALL_ENGINE_KEYS);
    if (!DEFAULT_MODE_KEYS.length) DEFAULT_MODE_KEYS.push(...ALL_MODE_KEYS);
    const DEFAULT_FOCUS = DATA.datasets.find(d => DEFAULT_DATASET_KEYS.includes(d.key) && d.source === 'wands')?.key || DEFAULT_DATASET_KEYS[0] || DATA.datasets[0]?.key;
    const state = { datasets: [...DEFAULT_DATASET_KEYS], engines: [...DEFAULT_ENGINE_KEYS], modes: [...DEFAULT_MODE_KEYS], metric: 'canonical', value: 'relative', performance: 'p50', scatterMode: 'best', focus: DEFAULT_FOCUS, tab: 'overview', search: '', page: 1, sort: 'quality', direction: 'desc', leaderboardSort: 'relative', leaderboardDirection: 'desc' };
    let matrixSort = { axis: null, key: null, direction: 'desc' };
    let matrixOrder = { datasets: [...ALL_DATASET_KEYS], engines: [...ALL_ENGINE_KEYS] };
    const datasetByKey = Object.fromEntries(DATA.datasets.map(d => [d.key, d]));
    const fmt = new Intl.NumberFormat('en-US');
    const dec = (value, digits = 3) => Number(value || 0).toFixed(digits);
    const ms = value => `${Number(value || 0).toFixed(value >= 100 ? 0 : 1)} ms`;
    const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    const systemName = engine => DATA.systems[engine]?.name || engine;
    const systemVersion = engine => DATA.systems[engine]?.version || 'unknown';
    const runVersion = run => run?.version || 'unknown';
    const imageDate = value => {
      if (!value) return '';
      const parsed = new Date(value);
      return Number.isNaN(parsed.getTime()) ? '' : new Intl.DateTimeFormat('en-US', {month:'short', day:'numeric', year:'numeric', timeZone:'UTC'}).format(parsed);
    };
    const versionWithDate = (version, createdAt) => `${version || 'unknown'}${imageDate(createdAt) ? ` (${imageDate(createdAt)})` : ''}`;
    const systemVersionLabel = engine => versionWithDate(systemVersion(engine), DATA.systems[engine]?.imageCreatedAt);
    const runVersionLabel = run => versionWithDate(runVersion(run), run?.imageCreatedAt);
    const modeBadge = mode => `<span class="mode-badge mode-${escapeHtml(mode)}">${escapeHtml(mode === 'fulltext' ? 'FT' : mode[0])}</span>`;
    const variantName = item => `${systemName(item.engine)} / ${item.mode}`;
    const METRIC_LABELS = { canonical: 'Canonical', ndcg10: 'NDCG@10', ndcg20: 'NDCG@20', mrr10: 'MRR@10', map10: 'MAP@10', p20: 'P@20', recall100: 'Recall@100' };
    const METRIC_DETAILS = {
      'Coverage': ['Dataset coverage', 'Completed results for this engine out of all datasets selected in the filter. Missing any selected dataset makes the engine incomplete and therefore unranked.'],
      'Mean relative': ['Average relative score', 'Arithmetic mean of engine score ÷ best observed score for every completed selected dataset. The best reference includes hidden engines, so filtering engines does not move it.'],
      'Mean raw': ['Average metric value', 'Arithmetic mean of the displayed metric values. With “Recommended for each dataset”, this can combine NDCG@10 and MRR@10, so use it only as a quick orientation.'],
      'Dataset wins': ['Best results', 'Number of selected datasets where this engine matches the best observed score. Ties count as wins.'],
      'Chosen modes': ['Modes behind the scores', 'Counts of the modes supplying each displayed result: FT = full text, V = vector, H = hybrid. The highest-quality result among the selected search modes is chosen separately for every engine and dataset.'],
      'NDCG@10': ['Ranking quality · top 10', 'Rewards highly relevant documents near the top of the first 10 results and respects graded relevance.'],
      'NDCG@20': ['Ranking quality · top 20', 'The same ranking-quality measure over the first 20 results. Higher is better.'],
      'MRR@10': ['First useful result · top 10', 'Measures how quickly the first relevant result appears. Best when users usually need one answer.'],
      'MAP@10': ['Relevant results near the top', 'Rewards retrieving multiple relevant documents early in the first 10 results.'],
      'P@20': ['Useful share of top 20', 'The fraction of the first 20 results judged relevant. Higher is better.'],
      'Recall@100': ['Relevant-result coverage', 'The fraction of all known relevant documents found in the first 100 results.'],
      'P50': ['Typical latency', 'Half of measured queries completed within this many milliseconds. Lower is better.'],
      'P95': ['Slow-query latency', '95% of measured queries completed within this time. Highlights slower requests.'],
      'P99': ['Tail latency', '99% of measured queries completed within this time. Exposes rare but severe delays.'],
      'QPS': ['Observed query rate', 'Successful queries per second at concurrency 1. This is not maximum throughput.'],
    };
    const metricValue = run => state.metric === 'canonical' ? run.quality : Number(run[state.metric] || 0);
    const metricLabel = run => state.metric === 'canonical' ? run.metric.toUpperCase() : METRIC_LABELS[state.metric];
    const metricDescription = name => METRIC_DETAILS[name]?.[1] || '';
    const metricNameHtml = name => {
      const description = metricDescription(name);
      return `<abbr class="metric-name" tabindex="0" role="button" aria-label="${escapeHtml(name)}: ${escapeHtml(description)}" title="${escapeHtml(description)}">${escapeHtml(name)}</abbr>`;
    };
    const friendlyMetric = name => METRIC_DETAILS[name] ? `${METRIC_DETAILS[name][0]} (${name})` : name;

    function mean(values) { return values.length ? values.reduce((a,b) => a + b, 0) / values.length : 0; }

    function visibleRuns({ ignoreMode = false } = {}) {
      return DATA.runs.filter(run =>
        state.datasets.includes(run.datasetKey) &&
        state.engines.includes(run.engine) &&
        (ignoreMode || state.modes.includes(run.mode))
      );
    }

    function selectedDatasetKeys() {
      return state.datasets;
    }

    function bestBy(items, keyFn, scoreFn = metricValue) {
      const out = new Map();
      for (const item of items) {
        const key = keyFn(item);
        if (!out.has(key) || scoreFn(item) > scoreFn(out.get(key))) out.set(key, item);
      }
      return [...out.values()];
    }

    function variantAggregates() {
      const groups = selectedDatasetKeys();
      const bestRuns = bestBy(visibleRuns(), run => `${run.engine}|${run.mode}|${run.datasetKey}`);
      const variants = new Map();
      for (const run of bestRuns) {
        const key = `${run.engine}|${run.mode}`;
        if (!variants.has(key)) variants.set(key, { engine: run.engine, mode: run.mode, runs: [] });
        variants.get(key).runs.push(run);
      }
      return [...variants.values()].map(item => ({
        engine: item.engine,
        mode: item.mode,
        covered: new Set(item.runs.map(run => run.datasetKey)).size,
        total: groups.length,
        quality: mean(item.runs.map(metricValue)),
        qps: mean(item.runs.map(run => run.qps)),
        p50: mean(item.runs.map(run => run.p50)),
        p95: mean(item.runs.map(run => run.p95)),
        p99: mean(item.runs.map(run => run.p99)),
        runs: item.runs,
      })).sort((a,b) => b.covered - a.covered || b.quality - a.quality || variantName(a).localeCompare(variantName(b)));
    }

    function runsForReference() {
      return DATA.runs.filter(run =>
        selectedDatasetKeys().includes(run.datasetKey) &&
        state.modes.includes(run.mode)
      );
    }

    function selectedRunsByEngineDataset({ respectEngineFilter = true } = {}) {
      const runs = (respectEngineFilter ? visibleRuns() : runsForReference());
      return bestBy(runs, run => `${run.engine}|${run.datasetKey}`);
    }

    function referenceWinners() {
      return new Map(bestBy(runsForReference(), run => run.datasetKey).map(run => [run.datasetKey, metricValue(run)]));
    }

    function engineAggregates() {
      const total = selectedDatasetKeys().length;
      const winners = referenceWinners();
      const grouped = new Map();
      for (const run of selectedRunsByEngineDataset()) {
        if (!grouped.has(run.engine)) grouped.set(run.engine, []);
        grouped.get(run.engine).push(run);
      }
      const result = [...grouped].map(([engine, runs]) => {
        const relatives = runs.map(run => metricValue(run) / Math.max(winners.get(run.datasetKey) || 0, 1e-12));
        const counts = { fulltext: 0, vector: 0, hybrid: 0 };
        runs.forEach(run => counts[run.mode]++);
        const complete = runs.length === total;
        const relativeByRun = runs.map((run, index) => ({ run, value: relatives[index] * 100 }));
        const floorEntry = relativeByRun.reduce((worst, entry) => !worst || entry.value < worst.value ? entry : worst, null);
        return {
          engine, runs, counts, complete, covered: runs.length, total,
          relative: mean(relatives) * 100,
          relatives: relativeByRun,
          floor: floorEntry ? floorEntry.value : 0,
          floorRun: floorEntry ? floorEntry.run : null,
          raw: mean(runs.map(metricValue)),
          wins: relatives.filter(value => Math.abs(value - 1) < 1e-10).length,
        };
      });
      const sortValue = item => state.leaderboardSort === 'floor' ? item.floor : state.leaderboardSort === 'wins' ? item.wins : item.relative;
      const direction = state.leaderboardDirection === 'asc' ? 1 : -1;
      result.sort((a,b) => {
        const primary = (sortValue(a) - sortValue(b)) * direction;
        if (Math.abs(primary) > 1e-12) return primary;
        if (state.leaderboardSort !== 'relative' && Math.abs(b.relative - a.relative) > 1e-12) return b.relative - a.relative;
        if (state.leaderboardSort !== 'floor' && Math.abs(b.floor - a.floor) > 1e-12) return b.floor - a.floor;
        if (state.leaderboardSort !== 'wins' && b.wins !== a.wins) return b.wins - a.wins;
        return b.covered-a.covered || systemName(a.engine).localeCompare(systemName(b.engine));
      });
      return result;
    }

    function populateControls() {
      $('#dataset-options').innerHTML = DATA.datasets.map(d => `<label class="multi-option"><input type="checkbox" data-multi-option="datasets" value="${escapeHtml(d.key)}"><span>${escapeHtml(d.label)}</span></label>`).join('');
      $('#engine-options').innerHTML = Object.entries(DATA.systems).sort((a,b) => a[1].name.localeCompare(b[1].name)).map(([key, info]) => `<label class="multi-option"><input type="checkbox" data-multi-option="engines" value="${key}"><span>${escapeHtml(info.name)}<small>${escapeHtml(systemVersionLabel(key))}</small></span></label>`).join('');
      $('#mode-options').innerHTML = ALL_MODE_KEYS.map(mode => `<label class="multi-option"><input type="checkbox" data-multi-option="modes" value="${mode}"><span>${modeBadge(mode)} ${escapeHtml(MODE_NAMES[mode])}</span></label>`).join('');
      const params = new URLSearchParams(location.hash.slice(1));
      const datasetParams = params.getAll('dataset');
      const engineParams = params.getAll('engine');
      const modeParams = params.getAll('mode');
      if (datasetParams.length) state.datasets = datasetParams.includes('none') ? [] : datasetParams.includes('all') ? [...ALL_DATASET_KEYS] : datasetParams.filter(key => ALL_DATASET_KEYS.includes(key));
      if (engineParams.length) state.engines = engineParams.includes('none') ? [] : engineParams.includes('all') ? [...ALL_ENGINE_KEYS] : engineParams.filter(key => ALL_ENGINE_KEYS.includes(key));
      if (modeParams.length) state.modes = modeParams.includes('none') ? [] : modeParams.includes('all') || modeParams.includes('best') ? [...ALL_MODE_KEYS] : modeParams.filter(key => ALL_MODE_KEYS.includes(key));
      for (const key of ['metric','value','performance','scatterMode','focus']) {
        const value = params.get(key);
        const node = key === 'focus' ? $('#focus-dataset') : key === 'scatterMode' ? $('#scatter-mode-filter') : $(`#${key}-filter`);
        if (key === 'focus') { if (value && ALL_DATASET_KEYS.includes(value)) state.focus = value; continue; }
        if (value && node && [...node.options].some(option => option.value === value)) state[key] = value;
        if (node) node.value = state[key];
      }
      if (['overview','dataset','runs'].includes(params.get('tab'))) state.tab = params.get('tab');
      if (['relative','floor','wins'].includes(params.get('qualitySort'))) state.leaderboardSort = params.get('qualitySort');
      if (['asc','desc'].includes(params.get('qualityDirection'))) state.leaderboardDirection = params.get('qualityDirection');
      document.querySelectorAll('[data-metric-filter]').forEach(node => node.value = state.metric);
      updateMultiFilterUi();
      updateFocusOptions();
    }

    function updateMultiFilterUi() {
      document.querySelectorAll('[data-multi-option="datasets"]').forEach(input => input.checked = state.datasets.includes(input.value));
      document.querySelectorAll('[data-multi-option="engines"]').forEach(input => input.checked = state.engines.includes(input.value));
      document.querySelectorAll('[data-multi-option="modes"]').forEach(input => input.checked = state.modes.includes(input.value));
      const datasetNames = state.datasets.map(key => datasetByKey[key]?.label).filter(Boolean);
      const engineNames = state.engines.map(systemName);
      const modeNames = state.modes.map(mode => MODE_NAMES[mode]);
      const datasetText = state.datasets.length === ALL_DATASET_KEYS.length ? `All ${ALL_DATASET_KEYS.length} datasets` : state.datasets.length === 0 ? 'No datasets selected' : state.datasets.length === 1 ? datasetNames[0] : `${state.datasets.length} of ${ALL_DATASET_KEYS.length} datasets`;
      const engineText = state.engines.length === ALL_ENGINE_KEYS.length ? `All ${ALL_ENGINE_KEYS.length} engines` : state.engines.length === 0 ? 'No engines selected' : state.engines.length === 1 ? engineNames[0] : `${state.engines.length} of ${ALL_ENGINE_KEYS.length} engines`;
      const modeText = state.modes.length === ALL_MODE_KEYS.length ? 'All modes · best available' : state.modes.length === 0 ? 'No modes selected' : state.modes.length === 1 ? modeNames[0] : `${state.modes.length} of ${ALL_MODE_KEYS.length} modes · best`;
      $('#dataset-summary').textContent = datasetText; $('#dataset-summary').title = datasetNames.join('\n');
      $('#engine-summary').textContent = engineText; $('#engine-summary').title = engineNames.join(', ');
      $('#mode-summary').textContent = modeText; $('#mode-summary').title = modeNames.join(', ');
    }

    function updateFocusOptions() {
      const datasets = DATA.datasets.filter(d => state.datasets.includes(d.key));
      if (datasets.length && !state.datasets.includes(state.focus)) state.focus = datasets[0].key;
      if (!datasets.length) state.focus = '';
      const select = $('#focus-dataset');
      select.innerHTML = datasets.length ? datasets.map(d => `<option value="${escapeHtml(d.key)}">${escapeHtml(d.label)}</option>`).join('') : '<option value="">No datasets selected</option>';
      select.disabled = datasets.length === 0;
      if (datasets.length) select.value = state.focus;
      const index = datasets.findIndex(d => d.key === state.focus);
      const prev = $('#focus-prev'), next = $('#focus-next'), position = $('#focus-position');
      prev.disabled = index <= 0;
      next.disabled = index < 0 || index >= datasets.length - 1;
      position.textContent = datasets.length ? `${index + 1} / ${datasets.length}` : '0 / 0';
      const prevDataset = index > 0 ? datasets[index - 1] : null;
      const nextDataset = index >= 0 && index < datasets.length - 1 ? datasets[index + 1] : null;
      prev.title = prevDataset ? `Previous: ${prevDataset.label}` : 'No previous dataset';
      next.title = nextDataset ? `Next: ${nextDataset.label}` : 'No next dataset';
      prev.setAttribute('aria-label', prevDataset ? `Previous dataset: ${prevDataset.label}` : 'No previous dataset');
      next.setAttribute('aria-label', nextDataset ? `Next dataset: ${nextDataset.label}` : 'No next dataset');
    }

    function stepFocusedDataset(direction) {
      const datasets = DATA.datasets.filter(d => state.datasets.includes(d.key));
      if (!datasets.length) return;
      const index = Math.max(0, datasets.findIndex(d => d.key === state.focus));
      const nextIndex = Math.max(0, Math.min(datasets.length - 1, index + direction));
      if (nextIndex === index) return;
      state.focus = datasets[nextIndex].key;
      updateFocusOptions();
      renderAll();
    }

    function updateDatasetSwitcherOffset() {
      const filterBar = $('.filter-bar');
      if (!filterBar) return;
      const style = getComputedStyle(filterBar);
      const sticky = style.position === 'sticky';
      const top = sticky ? (parseFloat(style.top) || 0) + filterBar.getBoundingClientRect().height : 0;
      document.documentElement.style.setProperty('--dataset-switcher-top', `${Math.ceil(top)}px`);
    }

    const HISTORY_MARKER = 'searchQualityBenchmark';
    let restoringHistory = false;
    let scrollHistoryTimer = null;

    function stateSnapshot() {
      return {
        datasets: [...state.datasets], engines: [...state.engines], modes: [...state.modes],
        metric: state.metric, value: state.value, performance: state.performance,
        scatterMode: state.scatterMode, focus: state.focus, tab: state.tab,
        search: state.search, page: state.page, sort: state.sort, direction: state.direction,
        leaderboardSort: state.leaderboardSort, leaderboardDirection: state.leaderboardDirection,
      };
    }

    function restoreStateSnapshot(saved) {
      if (!saved || typeof saved !== 'object') return false;
      state.datasets = Array.isArray(saved.datasets) ? saved.datasets.filter(key => ALL_DATASET_KEYS.includes(key)) : [...DEFAULT_DATASET_KEYS];
      state.engines = Array.isArray(saved.engines) ? saved.engines.filter(key => ALL_ENGINE_KEYS.includes(key)) : [...DEFAULT_ENGINE_KEYS];
      state.modes = Array.isArray(saved.modes) ? saved.modes.filter(key => ALL_MODE_KEYS.includes(key)) : [...DEFAULT_MODE_KEYS];
      const metricOptions = [...$('#metric-filter').options].map(option => option.value);
      const valueOptions = [...$('#value-filter').options].map(option => option.value);
      const performanceOptions = [...$('#performance-filter').options].map(option => option.value);
      const scatterOptions = [...$('#scatter-mode-filter').options].map(option => option.value);
      state.metric = metricOptions.includes(saved.metric) ? saved.metric : 'canonical';
      state.value = valueOptions.includes(saved.value) ? saved.value : 'relative';
      state.performance = performanceOptions.includes(saved.performance) ? saved.performance : 'p50';
      state.scatterMode = scatterOptions.includes(saved.scatterMode) ? saved.scatterMode : 'best';
      state.focus = ALL_DATASET_KEYS.includes(saved.focus) ? saved.focus : DEFAULT_FOCUS;
      state.tab = ['overview','dataset','runs'].includes(saved.tab) ? saved.tab : 'overview';
      state.search = typeof saved.search === 'string' ? saved.search : '';
      state.page = Number.isFinite(Number(saved.page)) && Number(saved.page) > 0 ? Number(saved.page) : 1;
      state.sort = typeof saved.sort === 'string' ? saved.sort : 'quality';
      state.direction = saved.direction === 'asc' ? 'asc' : 'desc';
      state.leaderboardSort = ['relative','floor','wins'].includes(saved.leaderboardSort) ? saved.leaderboardSort : 'relative';
      state.leaderboardDirection = saved.leaderboardDirection === 'asc' ? 'asc' : 'desc';
      $('#metric-filter').value = state.metric;
      document.querySelectorAll('[data-metric-filter]').forEach(node => node.value = state.metric);
      $('#value-filter').value = state.value;
      $('#performance-filter').value = state.performance;
      $('#scatter-mode-filter').value = state.scatterMode;
      $('#run-search').value = state.search;
      updateMultiFilterUi();
      updateFocusOptions();
      return true;
    }

    function hashForState() {
      const params = new URLSearchParams();
      if (state.datasets.join('|') !== DEFAULT_DATASET_KEYS.join('|')) (state.datasets.length ? state.datasets : ['none']).forEach(key => params.append('dataset', key));
      if (state.engines.join('|') !== DEFAULT_ENGINE_KEYS.join('|')) (state.engines.length ? state.engines : ['none']).forEach(key => params.append('engine', key));
      if (state.modes.join('|') !== DEFAULT_MODE_KEYS.join('|')) (state.modes.length ? state.modes : ['none']).forEach(key => params.append('mode', key));
      const defaults = { metric:'canonical', value:'relative', performance:'p50', scatterMode:'best', focus:DEFAULT_FOCUS, tab:'overview' };
      for (const key of Object.keys(defaults)) if (state[key] !== defaults[key]) params.set(key, state[key]);
      if (state.leaderboardSort !== 'relative') params.set('qualitySort', state.leaderboardSort);
      if (state.leaderboardDirection !== 'desc') params.set('qualityDirection', state.leaderboardDirection);
      return params.toString();
    }

    function stateUrl() {
      const nextHash = hashForState();
      const base = location.href.split('#')[0];
      return nextHash ? `${base}#${nextHash}` : base;
    }

    function historyPayload(scrollY = window.scrollY) {
      return {
        [HISTORY_MARKER]: true,
        ui: stateSnapshot(),
        scrollY: Math.max(0, Math.round(Number(scrollY) || 0)),
      };
    }

    function replaceHistoryEntry(scrollY = window.scrollY) {
      if (restoringHistory) return;
      try {
        history.replaceState(historyPayload(scrollY), '', stateUrl());
      } catch (_) {
        const nextHash = hashForState();
        if (location.hash.slice(1) !== nextHash) location.hash = nextHash;
      }
    }

    function pushHistoryEntry(scrollY = window.scrollY) {
      try {
        history.pushState(historyPayload(scrollY), '', stateUrl());
        return true;
      } catch (_) {
        const nextHash = hashForState();
        if (location.hash.slice(1) !== nextHash) location.hash = nextHash;
        return false;
      }
    }

    function syncHash() {
      replaceHistoryEntry(window.scrollY);
    }

    function navigateWithHistory(mutator, afterRender) {
      // Save the page we are leaving, including its exact scroll position.
      replaceHistoryEntry(window.scrollY);
      mutator();
      pushHistoryEntry(window.scrollY);
      renderAll({ syncHistory: false });
      if (typeof afterRender === 'function') afterRender();
    }

    function scheduleScrollHistoryUpdate() {
      if (restoringHistory || !history.state?.[HISTORY_MARKER]) return;
      clearTimeout(scrollHistoryTimer);
      scrollHistoryTimer = setTimeout(() => {
        if (!restoringHistory) replaceHistoryEntry(window.scrollY);
      }, 120);
    }

    function renderLeaderboard() {
      const engines = engineAggregates();
      $('#leaderboard-scope').textContent = `${selectedDatasetKeys().length} dataset${selectedDatasetKeys().length === 1 ? '' : 's'}`;
      if (!engines.length) { $('#leaderboard').innerHTML = '<div class="empty">No engines match these filters.</div>'; return; }
      let rank = 0;
      const completeEngines = engines.filter(item => item.complete);
      const bestRelative = Math.max(...completeEngines.map(item => item.relative), -Infinity);
      const bestFloor = Math.max(...completeEngines.map(item => item.floor), -Infinity);
      const bestWins = Math.max(...completeEngines.map(item => item.wins), -Infinity);
      const sortHeader = (key, label, help) => {
        const active = state.leaderboardSort === key;
        const arrow = active ? (state.leaderboardDirection === 'asc' ? '↑' : '↓') : '';
        return `<button type="button" class="leaderboard-sort-button${active ? ' active' : ''}" data-leaderboard-sort="${key}" title="${escapeHtml(help)}">${escapeHtml(label)} <span class="sort-indicator">${arrow}</span></button>`;
      };
      const rows = engines.map(item => {
        if (item.complete) rank++;
        return `<tr>
        <td class="num ${rank === 1 && item.complete ? 'best' : ''}">${item.complete ? rank : '—'}</td>
        <td class="engine-cell"><b style="color:${COLORS[item.engine]}">${escapeHtml(systemName(item.engine))}</b><small>${escapeHtml(systemVersionLabel(item.engine))}</small></td>
        <td><div class="coverage"><span>${item.covered}/${item.total}</span><i class="coverage-track"><i style="width:${100 * item.covered / item.total}%"></i></i></div></td>
        <td class="num ${item.complete && Math.abs(item.relative - bestRelative) < 1e-10 ? 'best' : ''}">${dec(item.relative, 2)}%</td>
        <td class="num quality-floor ${item.complete && Math.abs(item.floor - bestFloor) < 1e-10 ? 'best' : ''}"><strong>${dec(item.floor, 2)}%</strong><small>${escapeHtml(item.floorRun ? (datasetByKey[item.floorRun.datasetKey]?.label || item.floorRun.dataset).split(' · ')[0] : '—')}</small></td>
        ${state.metric === 'canonical' ? '' : `<td class="num">${dec(item.raw, 4)}</td>`}<td class="num ${item.complete && item.wins === bestWins ? 'best' : ''}">${item.wins}</td>
        <td class="subtle">${modeBadge('fulltext')} ${item.counts.fulltext} · ${modeBadge('vector')} ${item.counts.vector} · ${modeBadge('hybrid')} ${item.counts.hybrid}</td>
      </tr>`;
      }).join('');
      const rawHeader = state.metric === 'canonical' ? '' : `<th class="num">${metricNameHtml('Mean raw')}</th>`;
      $('#leaderboard').innerHTML = `<div class="table-wrap"><table><thead><tr><th class="num">Rank</th><th>Engine</th><th>${metricNameHtml('Coverage')}</th><th class="num">${sortHeader('relative','Mean relative',metricDescription('Mean relative'))}</th><th class="num">${sortHeader('floor','Quality floor','Lowest per-dataset quality relative to the best observed result on that dataset. Higher is better.')}</th>${rawHeader}<th class="num">${sortHeader('wins','Dataset wins',metricDescription('Dataset wins'))}</th><th>${metricNameHtml('Chosen modes')}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
      document.querySelectorAll('#leaderboard [data-leaderboard-sort]').forEach(button => button.addEventListener('click', () => {
        const next = button.dataset.leaderboardSort;
        if (state.leaderboardSort === next) state.leaderboardDirection = state.leaderboardDirection === 'asc' ? 'desc' : 'asc';
        else { state.leaderboardSort = next; state.leaderboardDirection = 'desc'; }
        renderAll();
      }));
    }

    function renderConsistency() {
      const target = $('#consistency-chart');
      const engines = engineAggregates();
      if (!target) return;
      if (!engines.length) { target.innerHTML = '<div class="empty">No engines match these filters.</div>'; return; }
      const allValues = engines.flatMap(item => item.relatives.map(entry => entry.value));
      if (!allValues.length) { target.innerHTML = '<div class="empty">No comparable quality results.</div>'; return; }
      const minObserved = Math.min(...allValues);
      const domainMin = Math.max(0, Math.min(90, Math.floor(minObserved / 5) * 5));
      const domainMax = 100;
      const width = 1120, left = 165, right = 44, top = 38, rowH = 42, bottom = 42;
      const plotW = width - left - right;
      const height = top + engines.length * rowH + bottom;
      const x = value => left + ((Math.max(domainMin, Math.min(domainMax, value)) - domainMin) / Math.max(1e-9, domainMax - domainMin)) * plotW;
      const ticks = [];
      for (let t = Math.ceil(domainMin / 5) * 5; t <= 100; t += 5) ticks.push(t);
      let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Quality consistency across datasets">`;
      for (const tick of ticks) {
        const tx = x(tick);
        svg += `<line class="gridline" x1="${tx}" y1="${top-10}" x2="${tx}" y2="${height-bottom+6}"></line>`;
        svg += `<text class="axis-label" x="${tx}" y="${height-12}" text-anchor="middle">${tick}%</text>`;
      }
      engines.forEach((item, index) => {
        const y = top + index * rowH + rowH/2;
        const color = COLORS[item.engine] || 'var(--accent-2)';
        const maxVal = Math.max(...item.relatives.map(entry => entry.value));
        svg += `<text class="point-label" x="${left-12}" y="${y+4}" text-anchor="end" style="fill:${color};font-weight:750">${escapeHtml(systemName(item.engine))}${item.complete ? '' : ` (${item.covered}/${item.total})`}</text>`;
        svg += `<line x1="${x(item.floor)}" y1="${y}" x2="${x(maxVal)}" y2="${y}" stroke="${color}" stroke-opacity=".28" stroke-width="2"></line>`;
        item.relatives.forEach(entry => {
          const datasetName = (datasetByKey[entry.run.datasetKey]?.label || entry.run.dataset).split(' · ')[0];
          svg += `<circle cx="${x(entry.value)}" cy="${y}" r="3.2" fill="${color}" fill-opacity=".55"><title>${escapeHtml(datasetName)} · ${dec(entry.value,2)}%</title></circle>`;
        });
        const floorName = item.floorRun ? (datasetByKey[item.floorRun.datasetKey]?.label || item.floorRun.dataset).split(' · ')[0] : '—';
        svg += `<circle cx="${x(item.floor)}" cy="${y}" r="6.2" fill="var(--panel)" stroke="${color}" stroke-width="2"><title>Quality floor · ${dec(item.floor,2)}% · ${escapeHtml(floorName)}</title></circle>`;
        const mx = x(item.relative), d=6;
        svg += `<polygon points="${mx},${y-d} ${mx+d},${y} ${mx},${y+d} ${mx-d},${y}" fill="${item.complete ? color : 'var(--panel)'}" stroke="${color}" stroke-width="1.6"><title>Mean relative · ${dec(item.relative,2)}% · coverage ${item.covered}/${item.total}</title></polygon>`;
        svg += `<text class="axis-label" x="${Math.min(width-right-2, mx+10)}" y="${y-8}" text-anchor="start" style="fill:${color}">${dec(item.relative,1)}%</text>`;
      });
      svg += `<text class="axis-label" x="${left+plotW/2}" y="14" text-anchor="middle">Per-dataset quality relative to the best observed result · higher is better</text></svg>`;
      target.innerHTML = svg;
    }

    function paretoFrontier(items, perfKey) {
      const higherIsBetter = perfKey === 'qps';
      return new Set(items.filter(a => !items.some(b => {
        if (a === b) return false;
        const perfNoWorse = higherIsBetter ? b[perfKey] >= a[perfKey] : b[perfKey] <= a[perfKey];
        const perfBetter = higherIsBetter ? b[perfKey] > a[perfKey] : b[perfKey] < a[perfKey];
        return metricValue(b) >= metricValue(a) && perfNoWorse && (metricValue(b) > metricValue(a) || perfBetter);
      })).map(item => item.runId));
    }

    function renderScatter() {
      if (!state.datasets.includes(state.focus)) { $('#scatter').innerHTML = '<div class="empty">Select at least one dataset to open the dataset explorer.</div>'; return; }
      const scoped = DATA.runs.filter(run =>
        run.datasetKey === state.focus &&
        state.engines.includes(run.engine) &&
        state.modes.includes(run.mode)
      );
      const items = bestBy(scoped, run => state.scatterMode === 'all' ? `${run.engine}|${run.mode}` : run.engine).filter(run => metricValue(run) > 0 && run[state.performance] > 0);
      if (!items.length) { $('#scatter').innerHTML = '<div class="empty">No chartable runs match this dataset and filter selection.</div>'; return; }
      const width = 1160, height = 440, left = 76, right = 30, top = 28, bottom = 58;
      const plotW = width - left - right, plotH = height - top - bottom;
      const qualities = items.map(metricValue), perfs = items.map(v => v[state.performance]);
      let xMin = Math.min(...qualities), xMax = Math.max(...qualities);
      if (xMin === xMax) { xMin *= .95; xMax *= 1.05; }
      const xPad = (xMax - xMin) * .08; xMin = Math.max(0, xMin - xPad); xMax += xPad;
      let yMin = Math.min(...perfs), yMax = Math.max(...perfs);
      if (yMin === yMax) { yMin *= .8; yMax *= 1.2; }
      yMin = Math.max(.001, yMin * .78); yMax *= 1.28;
      const lyMin = Math.log10(yMin), lyMax = Math.log10(yMax);
      const sx = value => left + (value - xMin) / (xMax - xMin) * plotW;
      const sy = value => top + plotH - (Math.log10(value) - lyMin) / (lyMax - lyMin) * plotH;
      const frontier = paretoFrontier(items, state.performance);
      let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Quality and performance scatter plot">`;
      for (let i=0; i<=5; i++) {
        const value = xMin + (xMax - xMin) * i / 5, x = sx(value);
        svg += `<line class="gridline" x1="${x}" y1="${top}" x2="${x}" y2="${top + plotH}"/><text class="axis-label" x="${x}" y="${height - 29}" text-anchor="middle">${dec(value,3)}</text>`;
      }
      for (let i=0; i<=4; i++) {
        const logValue = lyMin + (lyMax - lyMin) * i / 4, value = 10 ** logValue, y = sy(value);
        svg += `<line class="gridline" x1="${left}" y1="${y}" x2="${left + plotW}" y2="${y}"/><text class="axis-label" x="${left - 10}" y="${y + 4}" text-anchor="end">${value >= 100 ? value.toFixed(0) : value.toFixed(1)}</text>`;
      }
      svg += `<line class="axis" x1="${left}" y1="${top + plotH}" x2="${left + plotW}" y2="${top + plotH}"/><line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${top + plotH}"/>`;
      const yLabel = state.performance === 'qps' ? `${friendlyMetric('QPS')} · higher is better` : `${friendlyMetric(state.performance.toUpperCase())} (ms) · lower is better`;
      svg += `<text class="axis-label" x="${left + plotW/2}" y="${height - 7}" text-anchor="middle">${escapeHtml(friendlyMetric(state.metric === 'canonical' ? datasetByKey[state.focus].metric.toUpperCase() : METRIC_LABELS[state.metric]))} · higher is better</text><text class="axis-label" transform="translate(16 ${top + plotH/2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
      for (const item of items) {
        const x = sx(metricValue(item)), y = sy(item[state.performance]);
        const isFrontier = frontier.has(item.runId), radius = isFrontier ? 8 : 6;
        const tooltip = `${variantName(item)} · ${runVersionLabel(item)}\n${metricLabel(item)} ${dec(metricValue(item),4)}\n${state.performance.toUpperCase()} ${dec(item[state.performance],2)}\n${isFrontier ? 'Pareto frontier\n' : ''}Click for run details`;
        const stroke = isFrontier ? '#e9fff9' : '#071014', strokeWidth = isFrontier ? 2.5 : 1, color = COLORS[item.engine];
        let mark = '';
        if (item.mode === 'fulltext') mark = `<rect x="${x-radius}" y="${y-radius}" width="${radius*2}" height="${radius*2}" rx="1.5" fill="${color}" fill-opacity=".88" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
        else if (item.mode === 'hybrid') mark = `<path d="M ${x} ${y-radius-1} L ${x+radius+1} ${y} L ${x} ${y+radius+1} L ${x-radius-1} ${y} Z" fill="${color}" fill-opacity=".88" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
        else mark = `<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" fill-opacity=".88" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
        svg += `<g data-run-id="${escapeHtml(item.runId)}" tabindex="0" style="cursor:pointer"><title>${escapeHtml(tooltip)}</title>${mark}<text class="point-label" x="${x + 10}" y="${y + 3}">${escapeHtml(systemName(item.engine))} · ${escapeHtml(item.mode === 'fulltext' ? 'FT' : item.mode === 'vector' ? 'V' : 'H')}</text></g>`;
      }
      svg += '</svg>';
      $('#scatter').innerHTML = svg;
      document.querySelectorAll('#scatter [data-run-id]').forEach(node => {
        node.addEventListener('click', () => showRun(node.dataset.runId));
        node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') showRun(node.dataset.runId); });
      });
    }

    function qualityHeatColor(ratio) {
      if (ratio == null || !Number.isFinite(ratio)) return { bg: 'transparent', fg: '#bfd0d8' };
      const p = Math.max(0, Math.min(1, ratio));
      if (p >= .99) return { bg: 'rgba(94,210,154,.72)', fg: 'var(--heat-green-fg)' };
      if (p >= .97) return { bg: 'rgba(127,215,224,.58)', fg: 'var(--heat-cyan-fg)' };
      if (p >= .95) return { bg: 'rgba(208,192,90,.56)', fg: 'var(--heat-yellow-fg)' };
      if (p >= .90) return { bg: 'rgba(255,180,84,.52)', fg: 'var(--heat-orange-fg)' };
      return { bg: 'rgba(224,108,159,.48)', fg: 'var(--heat-red-fg)' };
    }

    function renderHeatmap() {
      let datasets = matrixOrder.datasets.map(key => datasetByKey[key]).filter(d => d && state.datasets.includes(d.key));
      let engines = matrixOrder.engines.filter(engine => state.engines.includes(engine));
      if (!datasets.length || !engines.length) { $('#heatmap').innerHTML = '<div class="empty">Select at least one dataset and one engine to build the comparison matrix.</div>'; return; }
      const best = new Map(selectedRunsByEngineDataset().map(run => [`${run.engine}|${run.datasetKey}`, run]));
      const winners = referenceWinners();
      const relativeScore = (engine,datasetKey) => {
        const run=best.get(`${engine}|${datasetKey}`), reference=winners.get(datasetKey) || 0;
        return run && reference > 0 ? metricValue(run) / reference : null;
      };
      const sortValues = (items,valueFn,labelFn) => [...items].sort((a,b)=>{
        const av=valueFn(a), bv=valueFn(b);
        if(av==null && bv==null)return labelFn(a).localeCompare(labelFn(b));
        if(av==null)return 1;
        if(bv==null)return -1;
        const delta=av-bv;
        return delta ? delta*(matrixSort.direction==='asc'?1:-1) : labelFn(a).localeCompare(labelFn(b));
      });
      if(matrixSort.axis==='datasets' && engines.includes(matrixSort.key)) {
        datasets=sortValues(datasets,d=>relativeScore(matrixSort.key,d.key),d=>d.label);
        matrixOrder.datasets=[...datasets.map(d=>d.key),...matrixOrder.datasets.filter(key=>!state.datasets.includes(key))];
      }
      if(matrixSort.axis==='engines' && datasets.some(d=>d.key===matrixSort.key)) {
        engines=sortValues(engines,e=>relativeScore(e,matrixSort.key),systemName);
        matrixOrder.engines=[...engines,...matrixOrder.engines.filter(engine=>!state.engines.includes(engine))];
      }
      const arrow=(axis,key)=>matrixSort.axis===axis&&matrixSort.key===key?(matrixSort.direction==='asc'?' ↑':' ↓'):' ↕';
      let rows = '';
      for (const dataset of datasets) {
        const cells = engines.map(engine => {
          const run = best.get(`${engine}|${dataset.key}`);
          if (!run) return `<td class="heat-cell"><div class="missing-cell" title="No completed run for this engine, dataset, and selected search modes">NO RUN</div></td>`;
          const reference = winners.get(dataset.key) || 0;
          const ratio = reference > 0 ? metricValue(run) / reference : 0;
          const color = qualityHeatColor(ratio);
          const bg = color.bg;
          const title = `${systemName(engine)} ${runVersionLabel(run)}\n${dataset.label}\n${MODE_NAMES[run.mode]}: ${dec(metricValue(run),4)}\n${dec(ratio*100,1)}% of the best observed result in this dataset`;
          const main = state.value === 'relative' ? `${dec(ratio*100,1)}%` : dec(metricValue(run),4);
          const sub = state.value === 'relative' ? dec(metricValue(run),4) : `${dec(ratio*100,1)}%`;
          const winnerClass = ratio >= 0.9995 ? ' matrix-winner' : '';
          return `<td class="heat-cell"><button type="button" class="matrix-cell-button${winnerClass}" data-run-id="${escapeHtml(run.runId)}" style="--cell-bg:${bg};--cell-fg:${color.fg}" title="${escapeHtml(title)}"><strong>${main}</strong><small>${modeBadge(run.mode)}${sub}</small></button></td>`;
        }).join('');
        rows += `<tr><td><div class="matrix-row-actions"><button class="matrix-sort-button" type="button" data-sort-dataset="${escapeHtml(dataset.key)}" title="Sort engines by this dataset">${escapeHtml(dataset.label.replace(/ · .*/, ''))}${arrow('engines',dataset.key)}</button><button class="matrix-explore" type="button" data-dataset-key="${escapeHtml(dataset.key)}" title="Open Dataset explorer">Explore →</button></div><span class="subtle">${dataset.docs == null ? '' : fmt.format(dataset.docs) + ' docs · '}${metricNameHtml(state.metric === 'canonical' ? dataset.metric.toUpperCase() : METRIC_LABELS[state.metric])}</span></td>${cells}</tr>`;
      }
      const headers = engines.map(engine => `<th class="heat-header"><button class="matrix-sort-button" type="button" data-sort-engine="${escapeHtml(engine)}" title="Sort datasets by this engine"><span style="color:${COLORS[engine]}">${escapeHtml(systemName(engine))}${arrow('datasets',engine)}</span><br><span class="subtle">${escapeHtml(systemVersionLabel(engine))}</span></button></th>`).join('');
      const corner = `<div class="matrix-axis-resets"><button type="button" data-alpha-order="datasets" title="Restore alphabetical dataset order">Dataset ↕</button><span>/</span><button type="button" data-alpha-order="engines" title="Restore alphabetical engine order">Engine ↕</button></div>`;
      $('#heatmap').innerHTML = `<div class="table-wrap"><table class="heatmap"><thead><tr><th>${corner}</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
      document.querySelectorAll('#heatmap [data-run-id]').forEach(node => node.addEventListener('click', () => showRun(node.dataset.runId)));
      document.querySelectorAll('#heatmap [data-sort-engine]').forEach(node=>node.addEventListener('click',()=>{const key=node.dataset.sortEngine;matrixSort=matrixSort.axis==='datasets'&&matrixSort.key===key?{...matrixSort,direction:matrixSort.direction==='desc'?'asc':'desc'}:{axis:'datasets',key,direction:'desc'};renderHeatmap();}));
      document.querySelectorAll('#heatmap [data-sort-dataset]').forEach(node=>node.addEventListener('click',()=>{const key=node.dataset.sortDataset;matrixSort=matrixSort.axis==='engines'&&matrixSort.key===key?{...matrixSort,direction:matrixSort.direction==='desc'?'asc':'desc'}:{axis:'engines',key,direction:'desc'};renderHeatmap();}));
      document.querySelectorAll('#heatmap [data-alpha-order]').forEach(node=>node.addEventListener('click',()=>{
        if(node.dataset.alphaOrder==='datasets') matrixOrder.datasets=[...ALL_DATASET_KEYS].sort((a,b)=>datasetByKey[a].label.localeCompare(datasetByKey[b].label));
        else matrixOrder.engines=[...ALL_ENGINE_KEYS].sort((a,b)=>systemName(a).localeCompare(systemName(b)));
        matrixSort={axis:null,key:null,direction:'desc'};renderHeatmap();
      }));
      document.querySelectorAll('#heatmap [data-dataset-key]').forEach(node => node.addEventListener('click', () => {
        navigateWithHistory(() => {
          state.focus = node.dataset.datasetKey;
          state.tab = 'dataset';
          $('#focus-dataset').value = state.focus;
        }, () => requestAnimationFrame(() => {
          const target = document.getElementById('quality-latency-panel');
          if (!target) return;
          const switcher = document.querySelector('.dataset-switcher');
          const filterBar = document.querySelector('.filter-bar');
          const stickyOffset = (switcher?.getBoundingClientRect().height || 0) +
            (filterBar && getComputedStyle(filterBar).position === 'sticky' ? filterBar.getBoundingClientRect().height : 0) + 16;
          const top = target.getBoundingClientRect().top + window.scrollY - stickyOffset;
          window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
        }));
      }));
    }

    function horizontalBars(items, options) {
      if (!items.length) return '<div class="empty">No comparable data for this selection.</div>';
      const width = 720, rowH = 31, top = 22, bottom = 28, left = 185, right = 58;
      const height = top + bottom + items.length * rowH, plotW = width - left - right;
      const maxAbs = Math.max(...items.map(item => Math.abs(item.value)), options.minimum || 0.01);
      const min = options.diverging ? -maxAbs : 0, max = maxAbs;
      const sx = value => left + (value - min) / (max - min) * plotW;
      let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
      if (options.diverging) svg += `<line class="axis" x1="${sx(0)}" y1="${top-5}" x2="${sx(0)}" y2="${height-bottom+2}"/>`;
      items.forEach((item, index) => {
        const y = top + index * rowH, x0 = sx(0), x1 = sx(item.value), barX = Math.min(x0,x1), barW = Math.max(2,Math.abs(x1-x0));
        const color = item.value >= 0 ? '#5ed29a' : '#e06c9f';
        svg += `<text class="axis-label" x="${left-9}" y="${y+13}" text-anchor="end">${escapeHtml(item.label)}</text><rect x="${barX}" y="${y+2}" width="${barW}" height="15" rx="4" fill="${color}" fill-opacity=".76"><title>${escapeHtml(item.title || '')}</title></rect><text class="axis-label" x="${item.value >= 0 ? x1+6 : x1-6}" y="${y+14}" text-anchor="${item.value >= 0 ? 'start':'end'}">${options.format(item.value)}</text>`;
      });
      svg += '</svg>'; return svg;
    }

    function allModeReferenceWinners() {
      const runs = DATA.runs.filter(run => selectedDatasetKeys().includes(run.datasetKey));
      return new Map(bestBy(runs, run => run.datasetKey).map(run => [run.datasetKey, metricValue(run)]));
    }

    function modeProfileRows() {
      const references = allModeReferenceWinners();
      const total = selectedDatasetKeys().length;
      const runs = DATA.runs.filter(run => state.datasets.includes(run.datasetKey) && state.engines.includes(run.engine) && state.modes.includes(run.mode));
      const best = bestBy(runs, run => `${run.engine}|${run.mode}|${run.datasetKey}`);
      const grouped = new Map();
      for (const run of best) {
        const key = `${run.engine}|${run.mode}`;
        if (!grouped.has(key)) grouped.set(key, {engine:run.engine, mode:run.mode, runs:[]});
        grouped.get(key).runs.push(run);
      }
      const byEngine = new Map();
      for (const item of grouped.values()) {
        const rel = item.runs.map(run => metricValue(run) / Math.max(references.get(run.datasetKey) || 0, 1e-12));
        const point = {engine:item.engine, mode:item.mode, value:100*mean(rel), covered:new Set(item.runs.map(r=>r.datasetKey)).size, total};
        if (!byEngine.has(item.engine)) byEngine.set(item.engine, []);
        byEngine.get(item.engine).push(point);
      }
      return [...byEngine].map(([engine,points]) => ({engine,points})).sort((a,b)=>systemName(a.engine).localeCompare(systemName(b.engine)));
    }

    function renderModeProfile() {
      const root = $('#mode-profile');
      if (!root) return;
      if (!state.datasets.length || !state.engines.length || !state.modes.length) { root.innerHTML='<div class="empty">Select datasets, engines, and at least one search mode.</div>'; return; }
      const rows = modeProfileRows();
      if (!rows.length) { root.innerHTML='<div class="empty">No comparable mode data for this selection.</div>'; return; }
      const order=['fulltext','vector','hybrid'];
      const width=1160,rowH=38,top=34,bottom=58,left=184,right=36,height=top+bottom+rows.length*rowH,plotW=width-left-right;
      const sx=value=>left+Math.max(0,Math.min(100,value))/100*plotW;
      let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Search mode quality comparison">`;
      for(let v=0;v<=100;v+=20){const x=sx(v);svg+=`<line class="gridline" x1="${x}" y1="${top-10}" x2="${x}" y2="${height-bottom}"/><text class="axis-label" x="${x}" y="${height-30}" text-anchor="middle">${v}%</text>`;}
      const marker=(mode,x,y,color,complete,title)=>{
        const fill=complete?color:'var(--panel)', stroke=color, sw=complete?1.2:2;
        if(mode==='fulltext') return `<circle cx="${x}" cy="${y}" r="5.5" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"><title>${escapeHtml(title)}</title></circle>`;
        if(mode==='vector') return `<rect x="${x-4.7}" y="${y-4.7}" width="9.4" height="9.4" transform="rotate(45 ${x} ${y})" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"><title>${escapeHtml(title)}</title></rect>`;
        return `<rect x="${x-5}" y="${y-5}" width="10" height="10" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"><title>${escapeHtml(title)}</title></rect>`;
      };
      rows.forEach((row,index)=>{
        const y=top+index*rowH+8,color=COLORS[row.engine];
        svg+=`<text class="axis-label" x="${left-10}" y="${y+4}" text-anchor="end" style="fill:${color};font-weight:700">${escapeHtml(systemName(row.engine))}</text>`;
        const points=order.map(mode=>row.points.find(p=>p.mode===mode)).filter(Boolean);
        for(let i=1;i<points.length;i++){
          const a=points[i-1],b=points[i];
          const dash=(a.covered<a.total||b.covered<b.total)?' stroke-dasharray="4 4"':'';
          svg+=`<line x1="${sx(a.value)}" y1="${y}" x2="${sx(b.value)}" y2="${y}" stroke="${color}" stroke-opacity=".55" stroke-width="1.5"${dash}/>`;
        }
        for(const point of points){
          const complete=point.covered===point.total;
          const title=`${systemName(row.engine)} · ${MODE_NAMES[point.mode]}\nMean relative quality ${dec(point.value,2)}%\nCoverage ${point.covered}/${point.total} datasets`;
          svg+=marker(point.mode,sx(point.value),y,color,complete,title);
        }
      });
      svg+=`<text class="axis-label" x="${left+plotW/2}" y="${height-6}" text-anchor="middle">Mean relative quality · higher is better</text></svg>`;
      root.innerHTML=svg;
    }

    function renderLatency() {
      if (!state.datasets.includes(state.focus)) { $('#latency').innerHTML = '<div class="empty">Select a dataset to inspect latency.</div>'; return; }
      const scoped = DATA.runs.filter(run => run.datasetKey === state.focus && state.engines.includes(run.engine) && state.modes.includes(run.mode));
      const items = bestBy(scoped, run => `${run.engine}|${run.mode}`).filter(v => v.p99 > 0).sort((a,b) => a.p99-b.p99).slice(0,20);
      if (!items.length) { $('#latency').innerHTML = '<div class="empty">No latency data for this selection.</div>'; return; }
      const width=1160, rowH=27, top=24, bottom=32, left=190, right=64, height=top+bottom+items.length*rowH, plotW=width-left-right;
      const min=.1, max=Math.max(...items.map(v=>v.p99))*1.2, lmin=Math.log10(min), lmax=Math.log10(max);
      const sx=value=>left+(Math.log10(Math.max(min,value))-lmin)/(lmax-lmin)*plotW;
      let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="P50 P95 and P99 latency ranges">`;
      for(let i=0;i<=4;i++){const lv=lmin+(lmax-lmin)*i/4,v=10**lv,x=sx(v);svg+=`<line class="gridline" x1="${x}" y1="${top-7}" x2="${x}" y2="${height-bottom}"/><text class="axis-label" x="${x}" y="${height-8}" text-anchor="middle">${v>=100?v.toFixed(0):v.toFixed(1)} ms</text>`;}
      items.forEach((item,index)=>{const y=top+index*rowH+6,x50=sx(item.p50),x95=sx(item.p95),x99=sx(item.p99),color=COLORS[item.engine];svg+=`<text class="axis-label" x="${left-9}" y="${y+4}" text-anchor="end">${escapeHtml(systemName(item.engine))} · ${escapeHtml(item.mode === 'fulltext' ? 'FT' : item.mode[0].toUpperCase())}</text><line x1="${x50}" y1="${y}" x2="${x99}" y2="${y}" stroke="${color}" stroke-width="1.5" stroke-opacity=".7"><title>P50 ${ms(item.p50)} · P95 ${ms(item.p95)} · P99 ${ms(item.p99)}</title></line><circle cx="${x50}" cy="${y}" r="2.5" fill="${color}"/><circle cx="${x95}" cy="${y}" r="2.5" fill="${color}" fill-opacity=".72"/><circle cx="${x99}" cy="${y}" r="3" fill="${color}" stroke="#e8eef4" stroke-width=".8"/>`;});
      svg+='</svg>'; $('#latency').innerHTML=svg;
    }


    function stableHash(text) {
      let h = 2166136261 >>> 0;
      for (let i=0;i<text.length;i++) { h ^= text.charCodeAt(i); h = Math.imul(h,16777619); }
      return h >>> 0;
    }

    function renderSlowQueries() {
      const rows = DATA.slowQueries.filter(item =>
        item.datasetKey === state.focus &&
        state.engines.includes(item.engine) &&
        state.modes.includes(item.mode)
      ).sort((a,b) => b.latency-a.latency || a.engine.localeCompare(b.engine) || a.mode.localeCompare(b.mode) || a.query.localeCompare(b.query));
      $('#query-count').textContent = `${fmt.format(Math.min(rows.length,60))} of ${fmt.format(rows.length)} samples · 8/run`;
      if (!rows.length) { $('#slow-queries').innerHTML='<div class="empty">No slow-query samples match these filters.</div>'; return; }
      const showDataset = false;
      const body=rows.slice(0,60).map((item,index)=>{
        const queryId=`slow-query-${String(item.runId).replace(/[^a-z0-9_-]/gi,'-')}-${index}`;
        const expandable=item.query.length > 60 || item.query.includes('\n');
        const toggle=expandable ? `<button type="button" class="query-toggle" data-query-toggle aria-expanded="false" aria-controls="${queryId}">… Show full query</button>` : '';
        const datasetCell = showDataset ? `<td>${escapeHtml(datasetByKey[item.datasetKey]?.label || item.datasetKey)}</td>` : '';
        return `<tr><td class="query"><div id="${queryId}" class="query-text">${escapeHtml(item.query)}</div>${toggle}</td>${datasetCell}<td class="engine-cell"><b style="color:${COLORS[item.engine]}">${escapeHtml(systemName(item.engine))}</b><small>${escapeHtml(runVersionLabel(item))}</small></td><td>${modeBadge(item.mode)} ${escapeHtml(MODE_NAMES[item.mode])}</td><td class="num">${ms(item.latency)}</td></tr>`;
      }).join('');
      const datasetHeader = showDataset ? '<th>Dataset</th>' : '';
      $('#slow-queries').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Query</th>${datasetHeader}<th>Engine</th><th>Mode</th><th class="num">Latency</th></tr></thead><tbody>${body}</tbody></table></div>`;
      document.querySelectorAll('#slow-queries [data-query-toggle]').forEach(button=>button.addEventListener('click',()=>{
        const cell=button.closest('td.query'), expanded=cell.classList.toggle('expanded');
        button.setAttribute('aria-expanded',String(expanded));
        button.textContent=expanded?'Show less':'… Show full query';
      }));
    }

    const SORT_LABELS = {date:'Date',engine:'Engine',mode:'Mode',dataset:'Dataset',docs:'Docs',quality:'Quality',p50:'P50',p95:'P95',p99:'P99',qps:'QPS'};
    function renderRunTable() {
      const needle = state.search.trim().toLowerCase();
      const rows=visibleRuns().filter(run => !needle || [run.engine, systemName(run.engine), run.mode, run.dataset, datasetByKey[run.datasetKey]?.label, run.profile, run.model].join(' ').toLowerCase().includes(needle));
      const key=state.sort, direction=state.direction==='asc'?1:-1;
      rows.sort((a,b)=>{let av=key==='quality'?metricValue(a):a[key],bv=key==='quality'?metricValue(b):b[key];if(key==='date'){av=a.runAt;bv=b.runAt;}if(key==='engine'){av=systemName(a.engine);bv=systemName(b.engine);}if(key==='dataset'){av=datasetByKey[a.datasetKey]?.label||a.datasetKey;}return (typeof av==='string'?av.localeCompare(bv):(av||0)-(bv||0))*direction;});
      $('#run-count').textContent=`${fmt.format(rows.length)} runs`;
      const header=Object.entries(SORT_LABELS).map(([key,label])=>{const metricKey=label==='Quality'?(state.metric==='canonical'?'NDCG@10 / MRR@10':METRIC_LABELS[state.metric]):label.toUpperCase();const help=label==='Quality'?'The selected quality metric; canonical uses the recommended metric for each dataset.':metricDescription(metricKey);return `<th class="${['docs','quality','qps','p50','p95','p99'].includes(key)?'num':''}"><button class="sort-button" data-sort="${key}" title="${escapeHtml(help)}">${label}${state.sort===key?(state.direction==='asc'?' ↑':' ↓'):''}</button></th>`;}).join('');
      const pageSize=50, pages=Math.max(1,Math.ceil(rows.length/pageSize)); state.page=Math.min(state.page,pages);
      const visible=rows.slice((state.page-1)*pageSize,state.page*pageSize);
      const body=visible.map(run=>`<tr><td><button type="button" class="sort-button" data-run-id="${escapeHtml(run.runId)}" title="Open run details">${escapeHtml(run.date.replace(' UTC',''))}</button></td><td class="engine-cell"><b style="color:${COLORS[run.engine]}">${escapeHtml(systemName(run.engine))}</b><small>${escapeHtml(runVersionLabel(run))}</small></td><td>${modeBadge(run.mode)} ${escapeHtml(MODE_NAMES[run.mode])}</td><td title="${escapeHtml(run.dataset)}">${escapeHtml((datasetByKey[run.datasetKey]?.label||run.dataset).replace(/ · .*/,''))}</td><td class="num">${run.docs==null?'—':fmt.format(run.docs)}</td><td class="num" title="${escapeHtml(metricLabel(run))}">${dec(metricValue(run),4)}</td><td class="num">${ms(run.p50)}</td><td class="num">${ms(run.p95)}</td><td class="num">${ms(run.p99)}</td><td class="num">${dec(run.qps,1)}</td></tr>`).join('');
      $('#run-table').innerHTML=`<div class="table-wrap" style="max-height:680px"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div><div class="pagination"><span>Page ${state.page} of ${pages} · ${fmt.format(rows.length)} results</span><div class="button-row"><button type="button" data-page="prev" ${state.page===1?'disabled':''}>Previous</button><button type="button" data-page="next" ${state.page===pages?'disabled':''}>Next</button></div></div>`;
      document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{const next=button.dataset.sort;if(state.sort===next)state.direction=state.direction==='asc'?'desc':'asc';else{state.sort=next;state.direction=['engine','mode','dataset','date'].includes(next)?'asc':'desc';}renderRunTable();}));
      document.querySelectorAll('#run-table [data-run-id]').forEach(button=>button.addEventListener('click',()=>showRun(button.dataset.runId)));
      document.querySelectorAll('#run-table [data-page]').forEach(button=>button.addEventListener('click',()=>{state.page+=button.dataset.page==='next'?1:-1;renderRunTable();}));
    }

    function showRun(runId) {
      const run = DATA.runs.find(item => item.runId === runId); if (!run) return;
      const dataset = datasetByKey[run.datasetKey];
      const metrics = [
        ['NDCG@10',run.ndcg10],['NDCG@20',run.ndcg20],['MRR@10',run.mrr10],['MAP@10',run.map10],['P@20',run.p20],['Recall@100',run.recall100],
        ['P50',ms(run.p50)],['P95',ms(run.p95)],['P99',ms(run.p99)],['QPS',run.qps]
      ];
      $('#run-dialog-content').innerHTML=`<div class="dialog-head"><div><h2>${escapeHtml(systemName(run.engine))} / ${escapeHtml(MODE_NAMES[run.mode])}</h2><div class="subtle">${escapeHtml(runVersionLabel(run))}</div></div><button class="dialog-close" type="button" aria-label="Close">×</button></div><div class="dialog-body"><div class="detail-hero"><div class="detail-score"><span>${metricNameHtml(metricLabel(run))}</span><strong>${dec(metricValue(run),4)}</strong><small>${escapeHtml(METRIC_DETAILS[metricLabel(run)]?.[0] || '')} · ${escapeHtml(dataset.label)}</small></div><div class="detail-score"><span>${metricNameHtml('P50')}</span><strong>${ms(run.p50)}</strong><small>Tail ${metricNameHtml('P99')} ${ms(run.p99)} · ${metricNameHtml('QPS')} ${dec(run.qps,1)}</small></div></div><div class="performance-disclaimer"><strong>Performance note.</strong> This benchmark is designed primarily to compare search quality. Latency and QPS were measured with a lightweight methodology and may be noisy; use them only as directional context, not as an authoritative performance comparison.</div><div class="detail-grid">${metrics.map(([name,value])=>`<div class="detail-tile"><span>${METRIC_DETAILS[name] ? metricNameHtml(name) : escapeHtml(name)}</span><strong>${typeof value==='number'?dec(value,4):value}</strong></div>`).join('')}</div><dl class="detail-kv"><dt>Run date</dt><dd>${escapeHtml(run.date)}</dd><dt>Dataset</dt><dd>${escapeHtml(dataset.label)}</dd><dt>Profile</dt><dd>${escapeHtml(run.profile)}</dd><dt>Model/options</dt><dd>${escapeHtml(run.model)}</dd><dt>Settings</dt><dd>concurrency ${escapeHtml(run.concurrency)} · limit ${escapeHtml(run.limit)} · warmup ${escapeHtml(run.warmup)} · repeats ${escapeHtml(run.repeats)}</dd></dl></div>`;
      $('#run-dialog-content .dialog-close').addEventListener('click',()=>$('#run-dialog').close());
      $('#run-dialog').showModal();
    }

    function exportCsv() {
      const columns=['date','engine','version','mode','dataset','docs','metric','quality','qps','p50','p95','p99','profile','model'];
      const quote=value=>`"${String(value??'').replaceAll('"','""')}"`;
      const lines=[columns.join(',')];
      for(const run of visibleRuns()) lines.push(columns.map(column=>quote(column==='version'?runVersionLabel(run):column==='engine'?systemName(run.engine):run[column])).join(','));
      const blob=new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
      anchor.href=url;anchor.download='search-benchmark-filtered.csv';anchor.click();URL.revokeObjectURL(url);
    }

    function exportJson() {
      const runs = visibleRuns().map(({runId,engine,...run})=>({engine:systemName(engine),...run}));
      const systems = state.engines.map(engine=>({name:systemName(engine),version:systemVersionLabel(engine)}));
      const {source,excludedManticoreRuns,invalidArtifacts,...meta}=DATA.meta;
      const payload = { meta, filters: {...state}, systems, datasets: DATA.datasets.filter(dataset=>state.datasets.includes(dataset.key)), runs };
      const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
      anchor.href=url;anchor.download='search-benchmark-filtered.json';anchor.click();URL.revokeObjectURL(url);
    }

    function renderTabs() {
      document.querySelectorAll('[data-view]').forEach(panel => panel.hidden = panel.dataset.view !== state.tab);
      document.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active', button.dataset.tab === state.tab));
    }

    function renderAll({ syncHistory = true } = {}) {
      updateFocusOptions();
      if (syncHistory) syncHash();
      renderTabs(); renderLeaderboard(); renderConsistency(); renderScatter(); renderHeatmap(); renderModeProfile(); renderLatency(); renderSlowQueries(); renderRunTable(); updateDatasetSwitcherOffset();
    }

    populateControls();
    const updatedDate = new Date(DATA.meta.generatedAt.replace(' ', 'T').replace(' UTC', 'Z'));
    $('#generated-at').textContent = `Last updated ${new Intl.DateTimeFormat('en-US', {month:'short', day:'numeric', year:'numeric', timeZone:'UTC'}).format(updatedDate)}`;
    $('#header-meta').textContent = `${fmt.format(DATA.meta.runs)} measured runs · ${Object.keys(DATA.systems).length} engines · ${DATA.datasets.length} public datasets`;
    updateDatasetSwitcherOffset();
    window.addEventListener('resize', updateDatasetSwitcherOffset);
    if ('ResizeObserver' in window) new ResizeObserver(updateDatasetSwitcherOffset).observe($('.filter-bar'));

    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    window.addEventListener('popstate', event => {
      const saved = event.state?.[HISTORY_MARKER] ? event.state : null;
      if (!saved) return;
      restoringHistory = true;
      restoreStateSnapshot(saved.ui);
      renderAll({ syncHistory: false });
      const targetScroll = Math.max(0, Number(saved.scrollY) || 0);
      requestAnimationFrame(() => {
        window.scrollTo({ top: targetScroll, left: 0, behavior: 'auto' });
        requestAnimationFrame(() => { restoringHistory = false; });
      });
    });
    window.addEventListener('scroll', scheduleScrollHistoryUpdate, { passive: true });

    const THEME_KEY = 'search-benchmark-theme';
    function applyTheme(theme) {
      const next = theme === 'light' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      const button = $('#theme-toggle');
      if (button) {
        const target = next === 'dark' ? 'light' : 'dark';
        const label = `Switch to ${target} theme`;
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);
      }
      try { localStorage.setItem(THEME_KEY, next); } catch (_) {}
    }
    let initialTheme = 'dark';
    try { initialTheme = localStorage.getItem(THEME_KEY) || 'dark'; } catch (_) {}
    applyTheme(initialTheme);
    $('#theme-toggle').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
    for(const key of ['value','performance']) $(`#${key}-filter`).addEventListener('change',event=>{state[key]=event.target.value;state.page=1;renderAll();});
    document.querySelectorAll('[data-metric-filter]').forEach(node => node.addEventListener('change', event => {
      state.metric = event.target.value;
      $('#metric-filter').value = state.metric;
      document.querySelectorAll('[data-metric-filter]').forEach(other => other.value = state.metric);
      state.page = 1;
      renderAll();
    }));
    $('#scatter-mode-filter').addEventListener('change',event=>{state.scatterMode=event.target.value;renderAll();});
    document.querySelectorAll('[data-multi-option]').forEach(input=>input.addEventListener('change',()=>{const key=input.dataset.multiOption;state[key]=[...document.querySelectorAll(`[data-multi-option="${key}"]:checked`)].map(node=>node.value);state.page=1;updateMultiFilterUi();updateFocusOptions();renderAll();}));
    document.querySelectorAll('[data-multi][data-choice]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.multi;state[key]=button.dataset.choice==='all'?[...ALL_FILTER_VALUES[key]]:[];state.page=1;updateMultiFilterUi();updateFocusOptions();renderAll();}));
    document.querySelectorAll('.multi-select').forEach(menu=>menu.addEventListener('toggle',()=>{if(menu.open)document.querySelectorAll('.multi-select').forEach(other=>{if(other!==menu)other.open=false;});}));
    document.addEventListener('click',event=>document.querySelectorAll('.multi-select[open]').forEach(menu=>{if(!menu.contains(event.target))menu.open=false;}));
    $('#focus-dataset').addEventListener('change',event=>{state.focus=event.target.value;updateFocusOptions();renderAll();});
    $('#focus-prev').addEventListener('click',()=>stepFocusedDataset(-1));
    $('#focus-next').addEventListener('click',()=>stepFocusedDataset(1));
    document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{
      const nextTab = button.dataset.tab;
      if (nextTab === state.tab) return;
      navigateWithHistory(() => { state.tab = nextTab; });
    }));
    $('#run-search').addEventListener('input',event=>{state.search=event.target.value;state.page=1;renderRunTable();});
    $('#reset-button').addEventListener('click',()=>{Object.assign(state,{datasets:[...DEFAULT_DATASET_KEYS],engines:[...DEFAULT_ENGINE_KEYS],modes:[...DEFAULT_MODE_KEYS],metric:'canonical',value:'relative',performance:'p50',scatterMode:'best',focus:DEFAULT_FOCUS,tab:'overview',search:'',page:1});matrixSort={axis:null,key:null,direction:'desc'};matrixOrder={datasets:[...ALL_DATASET_KEYS],engines:[...ALL_ENGINE_KEYS]};for(const key of ['metric','value','performance'])$(`#${key}-filter`).value=state[key];document.querySelectorAll('[data-metric-filter]').forEach(node=>node.value=state.metric);$('#scatter-mode-filter').value=state.scatterMode;updateMultiFilterUi();updateFocusOptions();$('#run-search').value='';renderAll();});
    $('#export-button').addEventListener('click',exportCsv);
    $('#export-json-button').addEventListener('click',exportJson);
    $('#run-dialog').addEventListener('click',event=>{if(event.target===$('#run-dialog'))$('#run-dialog').close();});

    // Metric info: keep the browser's delayed title tooltip on hover, and show
    // the exact same text immediately and persistently when the ⓘ marker is clicked.
    const infoTooltip = document.createElement('div');
    infoTooltip.className = 'info-tooltip-popover';
    infoTooltip.setAttribute('role', 'tooltip');
    infoTooltip.hidden = true;
    document.body.appendChild(infoTooltip);
    let pinnedInfoTarget = null;
    let pinnedInfoTitle = '';

    function closeInfoTooltip() {
      if (pinnedInfoTarget && pinnedInfoTitle && pinnedInfoTarget.isConnected) pinnedInfoTarget.setAttribute('title', pinnedInfoTitle);
      pinnedInfoTarget = null;
      pinnedInfoTitle = '';
      infoTooltip.hidden = true;
      infoTooltip.textContent = '';
    }

    function positionInfoTooltip(target) {
      if (!target || infoTooltip.hidden) return;
      const gap = 8, edge = 12;
      const rect = target.getBoundingClientRect();
      const box = infoTooltip.getBoundingClientRect();
      let left = rect.left + rect.width / 2 - box.width / 2;
      left = Math.max(edge, Math.min(left, window.innerWidth - box.width - edge));
      let top = rect.bottom + gap;
      let above = false;
      if (top + box.height > window.innerHeight - edge && rect.top - box.height - gap >= edge) {
        top = rect.top - box.height - gap;
        above = true;
      }
      infoTooltip.style.left = `${Math.round(left)}px`;
      infoTooltip.style.top = `${Math.round(top)}px`;
      const arrowLeft = Math.max(10, Math.min(box.width - 18, rect.left + rect.width / 2 - left - 4));
      infoTooltip.style.setProperty('--tooltip-arrow-left', `${Math.round(arrowLeft)}px`);
      infoTooltip.classList.toggle('above', above);
    }

    function openInfoTooltip(target) {
      if (pinnedInfoTarget === target && !infoTooltip.hidden) { closeInfoTooltip(); return; }
      closeInfoTooltip();
      const title = target.getAttribute('title') || target.dataset.infoTitle || '';
      if (!title) return;
      pinnedInfoTarget = target;
      pinnedInfoTitle = title;
      target.dataset.infoTitle = title;
      // Suppress the native hover tooltip while the persistent one is open.
      target.removeAttribute('title');
      infoTooltip.textContent = title;
      infoTooltip.hidden = false;
      positionInfoTooltip(target);
    }

    document.addEventListener('click', event => {
      const target = event.target.closest?.('abbr.metric-name');
      if (target) {
        event.preventDefault();
        event.stopPropagation();
        openInfoTooltip(target);
        return;
      }
      if (!infoTooltip.contains(event.target)) closeInfoTooltip();
    });
    document.addEventListener('keydown', event => {
      const target = event.target.closest?.('abbr.metric-name');
      if (target && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        openInfoTooltip(target);
      } else if (event.key === 'Escape') closeInfoTooltip();
    });
    window.addEventListener('resize', () => positionInfoTooltip(pinnedInfoTarget));
    window.addEventListener('scroll', () => positionInfoTooltip(pinnedInfoTarget), true);

    renderAll();
  })();
  </script>
</body>
</html>
'''


def write_html(data: dict[str, Any], output_path: str | Path = OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    return output_path


def generate_html(
    output_path: str | Path = OUTPUT_PATH,
    *,
    default_run_ids: set[str] | None = None,
) -> Path:
    return write_html(collect_data(default_run_ids=default_run_ids), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Standalone HTML output path")
    parser.add_argument(
        "--default-run-ids",
        default="",
        help="Comma-separated run IDs whose engines, datasets, and modes are selected initially",
    )
    args = parser.parse_args()
    default_run_ids = {value.strip() for value in args.default_run_ids.split(",") if value.strip()} or None
    data = collect_data(default_run_ids=default_run_ids)
    output_path = write_html(data, args.output)
    size_mib = output_path.stat().st_size / (1024 * 1024)
    try:
        display_path = output_path.relative_to(ROOT)
    except ValueError:
        display_path = output_path
    print(
        f"Wrote {display_path}: {data['meta']['runs']} runs, "
        f"{len(data['slowQueries'])} slow-query samples, {size_mib:.2f} MiB"
    )


if __name__ == "__main__":
    main()
