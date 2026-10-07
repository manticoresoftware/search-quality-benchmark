import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from bench.summarize import _generate_interactive_report, combined_html_report_path


def _load_report_generator():
    path = Path(__file__).parents[2] / "src" / "bench" / "interactive_report.py"
    spec = importlib.util.spec_from_file_location("search_bench_report_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metrics(
    root: Path,
    run_id: str,
    run_date: datetime,
    version: str | None,
    *,
    engine: str = "qdrant",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "run_at_unix": int(run_date.timestamp()),
        "engine": engine,
        "mode": "vector",
        "dataset_source": "beir/scifact/test",
        "relevance": {"vector": {"ndcg@10": 0.5}},
        "performance": {"vector": {"queries_total": 1}},
    }
    if version is not None:
        payload["engine_version"] = version
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_combined_html_report_path_uses_a_sibling_report_directory() -> None:
    assert combined_html_report_path("artifacts/report-20261001.md") == Path(
        "artifacts/report-20261001/index.html"
    )
    assert combined_html_report_path("artifacts/custom") == Path(
        "artifacts/custom-html/index.html"
    )


def test_generate_interactive_report_passes_selected_runs_as_initial_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    called: dict[str, object] = {}

    def fake_run(command, **kwargs):
        called["command"] = command
        called["kwargs"] = kwargs

    monkeypatch.setattr("bench.summarize.subprocess.run", fake_run)
    markdown_path = tmp_path / "comparison.md"
    selected = [SimpleNamespace(run_id="run-a"), SimpleNamespace(run_id="run-b")]

    html_path = _generate_interactive_report(selected, markdown_path)

    assert html_path == tmp_path / "comparison" / "index.html"
    command = called["command"]
    assert Path(command[1]).name == "interactive_report.py"
    assert Path(command[1]).parent.name == "bench"
    assert command[command.index("--output") + 1] == str(html_path)
    assert command[command.index("--default-run-ids") + 1] == "run-a,run-b"


def test_report_cohort_uses_known_versions_instead_of_a_date_window(
    tmp_path: Path, monkeypatch
) -> None:
    generator = _load_report_generator()
    assert generator.OUTPUT_PATH == (
        Path(__file__).parents[2] / "reports" / "search-benchmark" / "index.html"
    )
    _write_metrics(tmp_path, "before", datetime(2026, 8, 1, tzinfo=UTC), "1.2.3")
    _write_metrics(tmp_path, "after", datetime(2026, 10, 1, tzinfo=UTC), "1.2.4")
    _write_metrics(tmp_path, "missing", datetime(2026, 9, 1, tzinfo=UTC), None)
    _write_metrics(tmp_path, "unknown", datetime(2026, 9, 2, tzinfo=UTC), "unknown")
    _write_metrics(tmp_path, "floating", datetime(2026, 9, 3, tzinfo=UTC), "latest")
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)

    data = generator.collect_data()

    assert [run["runId"] for run in data["runs"]] == ["before", "after"]
    assert [run["version"] for run in data["runs"]] == ["1.2.3", "1.2.4"]


def test_report_includes_serenedb_only_when_it_has_a_versioned_run(
    tmp_path: Path, monkeypatch
) -> None:
    generator = _load_report_generator()
    _write_metrics(
        tmp_path,
        "serene",
        datetime(2026, 10, 1, tzinfo=UTC),
        "26.10.1",
        engine="serenedb",
    )
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)

    data = generator.collect_data()

    assert data["systems"] == {
        "serenedb": {
            "name": "SereneDB",
            "version": "26.10.1",
            "image": "unknown",
            "imageCreatedAt": "",
        }
    }
    assert data["runs"][0]["engine"] == "serenedb"
