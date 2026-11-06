import json
from pathlib import Path

from bench.summarize import (
    default_query_cohort_rows,
    discover_artifacts,
    selected_query_cohort_rows,
)


def _write_prepared_dataset(
    root: Path,
    slug: str,
    source: str,
    *,
    short_query: str,
    long_query: str,
) -> None:
    prepared_dir = root / "data" / "prepared" / slug
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "manifest.json").write_text(
        json.dumps({"dataset_source": source, "doc_count": 10, "qrels_count": 1, "topic_count": 1}),
        encoding="utf-8",
    )
    (prepared_dir / "topics.jsonl").write_text(
        json.dumps(
            {
                "topic_id": "1",
                "query_short": short_query,
                "query_long": long_query,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_legacy_artifact(root: Path, run_id: str, source: str, mode: str) -> None:
    run_dir = root / "artifacts" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "engine": "qdrant",
                "mode": mode,
                "dataset_source": source,
                "relevance": {mode: {"ndcg@10": 0.5}},
                "performance": {mode: {"qps": 1.0}},
                "live_state_lines": ["rows (index): 10"],
            }
        ),
        encoding="utf-8",
    )


def test_discover_artifacts_uses_effective_query_set_for_legacy_runs(
    tmp_path: Path, monkeypatch
) -> None:
    _write_prepared_dataset(
        tmp_path,
        "trec",
        "beir/trec-covid",
        short_query="short",
        long_query="long query",
    )
    _write_prepared_dataset(
        tmp_path,
        "scifact",
        "beir/scifact/test",
        short_query="same query",
        long_query="same query",
    )
    _write_legacy_artifact(tmp_path, "trec-ft", "beir/trec-covid", "fulltext")
    _write_legacy_artifact(tmp_path, "trec-vector", "beir/trec-covid", "vector")
    _write_legacy_artifact(tmp_path, "scifact-ft", "beir/scifact/test", "fulltext")
    _write_legacy_artifact(tmp_path, "scifact-vector", "beir/scifact/test", "vector")
    monkeypatch.chdir(tmp_path)

    discovered = discover_artifacts()
    rows = {row.run_id: row for row in discovered}

    assert rows["trec-ft"].dataset_source == "trec-covid-query"
    assert rows["trec-ft"].query_field == "query"
    assert rows["trec-ft"].query_variant == "same"
    assert rows["trec-ft"].query_variant_explicit is False
    assert rows["trec-vector"].dataset_source == "trec-covid-narrative"
    assert rows["trec-vector"].query_field == "narrative"
    assert rows["trec-vector"].query_variant == "same"
    assert rows["trec-ft"].comparison_dataset_source != rows["trec-vector"].comparison_dataset_source
    assert rows["scifact-ft"].query_variant == "same"
    assert rows["scifact-vector"].query_variant == "same"
    assert rows["scifact-ft"].comparison_dataset_source == rows["scifact-vector"].comparison_dataset_source
    assert {row.run_id for row in default_query_cohort_rows(discovered)} == {
        "trec-ft",
        "trec-vector",
        "scifact-ft",
        "scifact-vector",
    }


def test_default_query_cohort_includes_legacy_trec_logical_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    _write_prepared_dataset(
        tmp_path,
        "trec",
        "beir/trec-covid",
        short_query="short",
        long_query="long query",
    )
    _write_legacy_artifact(tmp_path, "trec-ft", "beir/trec-covid", "fulltext")
    monkeypatch.chdir(tmp_path)

    rows = default_query_cohort_rows(discover_artifacts())
    assert [row.dataset_source for row in rows] == ["trec-covid-query"]


def test_unmarked_non_trec_legacy_fulltext_defaults_to_same_without_prepared_data(
    tmp_path: Path, monkeypatch
) -> None:
    _write_legacy_artifact(tmp_path, "scifact-ft", "beir/scifact/test", "fulltext")
    monkeypatch.chdir(tmp_path)

    row = discover_artifacts()[0]

    assert row.query_variant == "same"
    assert row.query_variant_explicit is False
    assert default_query_cohort_rows([row]) == [row]


def test_legacy_trec_mapping_uses_mode_not_old_query_variant_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    _write_prepared_dataset(
        tmp_path,
        "trec",
        "beir/trec-covid",
        short_query="short",
        long_query="long query",
    )
    _write_legacy_artifact(tmp_path, "new-long", "beir/trec-covid", "fulltext")
    metrics_path = tmp_path / "artifacts" / "new-long" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "query_variant_requested": "long",
            "query_variant": "long",
            "query_variants_distinct": True,
        }
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    row = discover_artifacts()[0]

    assert row.dataset_source == "trec-covid-query"
    assert row.query_field == "query"
    assert row.query_variant_requested == "query"
    assert row.query_variant == "same"
    assert row.query_variants_distinct is True
    assert row.query_variant_explicit is False


def test_selected_legacy_trec_views_are_all_retained(
    tmp_path: Path, monkeypatch
) -> None:
    _write_prepared_dataset(
        tmp_path,
        "trec",
        "beir/trec-covid",
        short_query="short",
        long_query="long query",
    )
    _write_legacy_artifact(tmp_path, "legacy-short", "beir/trec-covid", "fulltext")
    _write_legacy_artifact(tmp_path, "recorded-short", "beir/trec-covid", "fulltext")
    _write_legacy_artifact(tmp_path, "long", "beir/trec-covid", "vector")
    metrics_path = tmp_path / "artifacts" / "recorded-short" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "query_variant_requested": "short",
            "query_variant": "short",
            "query_variants_distinct": True,
        }
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rows = {row.run_id: row for row in discover_artifacts()}

    accidental = selected_query_cohort_rows([rows["legacy-short"], rows["long"]])
    enabled = selected_query_cohort_rows(
        [rows["legacy-short"], rows["recorded-short"], rows["long"]]
    )

    assert {row.run_id for row in accidental} == {"legacy-short", "long"}
    assert {row.run_id for row in enabled} == {"legacy-short", "recorded-short", "long"}


def test_recorded_short_gate_does_not_filter_a_non_trec_selection(
    tmp_path: Path, monkeypatch
) -> None:
    _write_legacy_artifact(tmp_path, "other-short", "future/distinct", "fulltext")
    metrics_path = tmp_path / "artifacts" / "other-short" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "query_variant_requested": "short",
            "query_variant": "short",
            "query_variants_distinct": True,
        }
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    row = discover_artifacts()[0]

    assert selected_query_cohort_rows([row]) == [row]


def test_new_trec_question_artifact_uses_logical_dataset_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _write_legacy_artifact(tmp_path, "question", "beir/trec-covid", "vector")
    metrics_path = tmp_path / "artifacts" / "question" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "benchmark_dataset_id": "trec-covid-question",
            "dataset_family": "trec-covid",
            "query_field": "question",
            "query_variant": "question",
        }
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    row = discover_artifacts()[0]

    assert row.dataset_source == "trec-covid-question"
    assert row.physical_dataset_source == "beir/trec-covid"
    assert row.dataset_family == "trec-covid"
    assert row.query_field == "question"
    assert row.query_variant == "same"
    assert row.query_variant_explicit is True
