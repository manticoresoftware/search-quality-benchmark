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
    mode: str = "vector",
    dataset: str = "beir/scifact/test",
    query_variant: str | None = None,
    model: str | None = None,
    hybrid_implementation: str | None = None,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "run_at_unix": int(run_date.timestamp()),
        "engine": engine,
        "mode": mode,
        "dataset_source": dataset,
        "relevance": {mode: {"ndcg@10": 0.5}},
        "performance": {mode: {"queries_total": 1}},
    }
    if version is not None:
        payload["engine_version"] = version
    if model is not None:
        payload["model"] = model
    if hybrid_implementation is not None:
        payload["hybrid_implementation"] = hybrid_implementation
    if query_variant is not None:
        payload.update(
            {
                "query_variant_requested": query_variant,
                "query_variant": query_variant,
                "query_variants_distinct": True,
            }
        )
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
            "hybridImplementations": ["non-native"],
        }
    }
    assert data["runs"][0]["engine"] == "serenedb"


def test_report_discovers_engines_not_named_in_the_generator(
    tmp_path: Path, monkeypatch
) -> None:
    generator = _load_report_generator()
    _write_metrics(
        tmp_path,
        "future",
        datetime(2026, 10, 2, tzinfo=UTC),
        "1.0.0",
        engine="future_search",
    )
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)

    data = generator.collect_data()

    assert data["systems"] == {
        "future_search": {
            "name": "Future Search",
            "version": "1.0.0",
            "image": "unknown",
            "imageCreatedAt": "",
            "hybridImplementations": [],
        }
    }
    assert data["runs"][0]["engine"] == "future_search"


def test_report_template_assigns_engine_colors_from_embedded_data() -> None:
    generator = _load_report_generator()

    assert "const ALL_ENGINE_KEYS = Object.keys(DATA.systems);" in generator.HTML_TEMPLATE
    assert "Object.fromEntries(ALL_ENGINE_KEYS.map((engine, index)" in generator.HTML_TEMPLATE
    assert "columnar: '#895af8'" not in generator.HTML_TEMPLATE


def test_hybrid_implementation_classifies_legacy_and_recorded_runs() -> None:
    generator = _load_report_generator()

    assert generator.hybrid_implementation(
        {"model": "embedding [fusion=rrf sparse=Qdrant/bm25]"},
        "qdrant",
        "hybrid",
    ) == "non-native"
    assert generator.hybrid_implementation(
        {"model": "embedding [fusion=native-rrf k=61]"},
        "qdrant",
        "hybrid",
    ) == "native"
    assert generator.hybrid_implementation({}, "mariadb", "hybrid") == "non-native"
    assert generator.hybrid_implementation(
        {"model": "embedding [fusion=server-rrf k=60]"},
        "mariadb",
        "hybrid",
    ) == "native"
    assert generator.hybrid_implementation({}, "serenedb", "hybrid") == "non-native"
    assert generator.hybrid_implementation({}, "milvus", "hybrid") == "native"
    assert generator.hybrid_implementation(
        {"hybrid_implementation": "native"}, "mariadb", "hybrid"
    ) == "native"
    assert generator.hybrid_implementation({}, "qdrant", "vector") == ""
    assert generator.engine_hybrid_implementations("qdrant") == [
        "native",
        "non-native",
    ]
    assert generator.engine_hybrid_implementations("mariadb") == [
        "native",
        "non-native",
    ]
    assert generator.engine_hybrid_implementations("serenedb") == ["non-native"]
    assert generator.engine_hybrid_implementations("elasticsearch") == ["native"]
    assert generator.engine_hybrid_implementations("future_search") == []


def test_report_splits_hybrid_mode_variants_and_defaults(tmp_path: Path, monkeypatch) -> None:
    generator = _load_report_generator()
    run_date = datetime(2026, 10, 1, tzinfo=UTC)
    _write_metrics(
        tmp_path,
        "qdrant-client",
        run_date,
        "1.19.1",
        engine="qdrant",
        mode="hybrid",
        model="embedding [fusion=rrf sparse=Qdrant/bm25]",
    )
    _write_metrics(
        tmp_path,
        "qdrant-native",
        run_date,
        "1.19.1",
        engine="qdrant",
        mode="hybrid",
        model="embedding [fusion=native-rrf k=61]",
    )
    _write_metrics(
        tmp_path,
        "mariadb-client",
        run_date,
        "11.8.9",
        engine="mariadb",
        mode="hybrid",
    )
    _write_metrics(
        tmp_path,
        "mariadb-server",
        run_date,
        "11.8.9",
        engine="mariadb",
        mode="hybrid",
        model="embedding [fusion=server-rrf k=60]",
        hybrid_implementation="native",
    )
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(generator, "PREPARED_DIR", tmp_path / "missing-prepared")

    data = generator.collect_data(
        default_run_ids={"qdrant-client", "qdrant-native"}
    )
    by_id = {run["runId"]: run for run in data["runs"]}

    assert by_id["qdrant-client"]["modeVariant"] == "hybrid-non-native"
    assert by_id["qdrant-native"]["modeVariant"] == "hybrid-native"
    assert by_id["mariadb-client"]["modeVariant"] == "hybrid-non-native"
    assert by_id["mariadb-server"]["modeVariant"] == "hybrid-native"
    assert data["meta"]["defaultModes"] == ["hybrid-native"]

    non_native_only = generator.collect_data(default_run_ids={"qdrant-client"})
    assert non_native_only["meta"]["defaultModes"] == []


def test_report_template_filters_hybrid_variants_but_groups_best_by_base_mode() -> None:
    generator = _load_report_generator()

    assert "'hybrid-native', 'hybrid-non-native'" in generator.HTML_TEMPLATE
    assert "state.modes.includes(run.modeVariant)" in generator.HTML_TEMPLATE
    assert "function engineMatchesHybridFilter(engine)" in generator.HTML_TEMPLATE
    assert "DATA.systems[engine]?.hybridImplementations" in generator.HTML_TEMPLATE
    assert "filteredRuns({ respectEngineFilter: false })" in generator.HTML_TEMPLATE
    assert "!restrictToOneHybridCohort || eligibleEngines.has(engine)" in generator.HTML_TEMPLATE
    assert "`${run.engine}|${run.mode}|${run.datasetKey}`" in generator.HTML_TEMPLATE
    assert "Hybrid · non-native (opt-in)" in generator.HTML_TEMPLATE
    assert "non-native-hybrid-warning" in generator.HTML_TEMPLATE
    assert "not directly comparable with native engine results" in generator.HTML_TEMPLATE
    assert ".filter(mode => mode !== 'hybrid-non-native')" in generator.HTML_TEMPLATE
    assert "Comparable modes · best available" in generator.HTML_TEMPLATE
    assert "modeParams.includes('best') ? [...SAFE_DEFAULT_MODE_KEYS]" in generator.HTML_TEMPLATE


def test_legacy_query_variants_split_only_when_topic_texts_differ() -> None:
    generator = _load_report_generator()

    assert generator.artifact_query_variant({}, "fulltext", True) == ("short", "short", True)
    assert generator.artifact_query_variant({}, "vector", True) == ("long", "long", True)
    assert generator.artifact_query_variant({}, "fulltext", False) == ("short", "same", False)
    assert generator.artifact_query_variant({}, "vector", False) == ("long", "same", False)


def test_recorded_long_fulltext_query_variant_is_preserved() -> None:
    generator = _load_report_generator()
    payload = {
        "query_variant_requested": "long",
        "query_variant": "long",
        "query_variants_distinct": True,
    }

    assert generator.artifact_query_variant(payload, "fulltext", None) == ("long", "long", True)
    assert generator.trec_covid_artifact_identity(payload, "fulltext") == (
        "trec-covid-query",
        "query",
    )


def test_unmarked_non_trec_fulltext_is_same_when_prepared_data_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    generator = _load_report_generator()
    _write_metrics(
        tmp_path,
        "scifact-ft",
        datetime(2026, 10, 1, tzinfo=UTC),
        "1.2.3",
        mode="fulltext",
        dataset="beir/scifact/test",
    )
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(generator, "PREPARED_DIR", tmp_path / "missing-prepared")

    data = generator.collect_data()

    assert data["runs"][0]["queryVariant"] == "same"
    assert data["runs"][0]["queryVariantExplicit"] is False


def test_legacy_trec_runs_are_split_into_logical_datasets_without_gating(
    tmp_path: Path, monkeypatch
) -> None:
    generator = _load_report_generator()
    run_date = datetime(2026, 10, 1, tzinfo=UTC)
    _write_metrics(
        tmp_path,
        "legacy-short",
        run_date,
        "1.2.3",
        mode="fulltext",
        dataset="beir/trec-covid",
    )
    _write_metrics(
        tmp_path,
        "recorded-short",
        run_date,
        "1.2.3",
        mode="fulltext",
        dataset="beir/trec-covid",
        query_variant="short",
    )
    _write_metrics(
        tmp_path,
        "long",
        run_date,
        "1.2.3",
        mode="vector",
        dataset="beir/trec-covid",
    )
    monkeypatch.setattr(generator, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(generator, "PREPARED_DIR", tmp_path / "missing-prepared")

    default_data = generator.collect_data()
    mistaken_data = generator.collect_data(default_run_ids={"legacy-short", "long"})
    enabled_data = generator.collect_data(
        default_run_ids={"legacy-short", "recorded-short"}
    )

    assert [run["runId"] for run in default_data["runs"]] == [
        "legacy-short",
        "long",
        "recorded-short",
    ]
    assert default_data["meta"]["excludedQueryCohortRuns"] == 0
    assert [run["runId"] for run in mistaken_data["runs"]] == [
        "legacy-short",
        "long",
        "recorded-short",
    ]
    assert [run["runId"] for run in enabled_data["runs"]] == [
        "legacy-short",
        "long",
        "recorded-short",
    ]
    assert enabled_data["runs"][0]["dataset"] == "trec-covid-query"
    assert enabled_data["runs"][1]["dataset"] == "trec-covid-narrative"
    assert enabled_data["runs"][2]["dataset"] == "trec-covid-query"


def test_only_a_selected_recorded_short_enables_inferred_short_runs() -> None:
    generator = _load_report_generator()
    runs = [
        {"runId": "legacy-short", "dataset": "beir/trec-covid", "docs": 10, "queryVariant": "short", "queryVariantExplicit": False},
        {"runId": "recorded-short", "dataset": "beir/trec-covid", "docs": 10, "queryVariant": "short", "queryVariantExplicit": True},
        {"runId": "long", "dataset": "beir/trec-covid", "docs": 10, "queryVariant": "long", "queryVariantExplicit": True},
    ]

    mistaken = generator.query_cohort_runs(runs, {"legacy-short", "long"})
    enabled = generator.query_cohort_runs(runs, {"recorded-short"})

    assert mistaken == [runs[2]]
    assert enabled == runs
