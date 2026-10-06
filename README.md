# Search Bench

Search Bench compares retrieval quality across open-source search engines using
public information-retrieval datasets. It runs the same queries and relevance
judgments against each engine, keeps reproducible artifacts, and builds a
standalone interactive HTML report.

Quality is the primary benchmark target. Latency and throughput are included as
supporting context, not as authoritative load-test results.

## Supported engines and search modes

The benchmark currently supports Manticore, Elasticsearch, OpenSearch,
Meilisearch, Typesense, Milvus, Qdrant, MariaDB, Weaviate, and Vespa.

Depending on the engine, it can compare:

- full-text search;
- dense vector search;
- hybrid lexical and vector search.

Ready dataset profiles include BEIR datasets, MS MARCO Passage, TREC-DL,
DL-Typo, WANDS, and ACORD. The available profiles live in `config.yaml`.

## Quick start

Requirements:

- Python 3.11 or newer;
- Docker with Docker Compose;
- an embedding provider API key when a profile needs uncached embeddings.

Install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Start the interactive CLI:

```bash
bench
```

The normal workflow is:

1. `engines` — start or stop the required search engines.
2. `prepare` — download a dataset and prepare documents, topics, judgments, and embeddings.
3. `index` — create and populate indexes for selected engine profiles.
4. `run` — execute benchmark queries and write run artifacts.
5. `report` — compare selected runs in Markdown and standalone HTML.

The interactive menus use arrow keys and `Enter`; multi-select screens also use
`Space`. Press `Esc` or `q` to leave a screen.

## Configuration

`config.yaml` contains the profiles used by the CLI. Start with an existing
profile and adjust it for your dataset, engine, or search mode.

`config.yaml.template` documents all available options. For embeddings, prefer
an API-key file rather than placing a secret directly in YAML:

```yaml
embedding:
  provider: "openrouter"
  model: "qwen/qwen3-embedding-8b"
  api_key_file: "/absolute/path/to/keyfile"
```

Prepared datasets are stored below `data/prepared/`. Cached embeddings are kept
in `vector_cache/`; generating them can take time and may incur API costs.

## Interactive report

Run `report` from the `bench` menu, or rebuild the repository’s complete report
directly:

```bash
python3 src/bench/interactive_report.py
```

Open [`reports/search-benchmark/index.html`](reports/search-benchmark/index.html)
in a browser. It is a standalone file with embedded CSS, JavaScript, benchmark
results, and slow-query samples; it does not require a web server or external
assets.

The report provides:

- an overall quality comparison and coverage overview;
- a dataset × engine quality matrix;
- cross-dataset consistency and search-mode comparisons;
- a per-dataset quality/latency explorer;
- latency distribution and slow-query inspection;
- a searchable table of measured runs with CSV and JSON export.

Dataset, engine, and search-mode filters can select any subset. Filter state is
stored in the URL, so a view can be bookmarked or shared.

Only artifacts with a concrete `engine_version` are admitted to the complete
report. Runs with a missing, unknown, or floating `latest` version are omitted;
there is no date cutoff. Artifacts named `manticore` are excluded from this
report, while `columnar` runs are displayed as Manticore.

## Benchmark report methodology

Each dataset uses its recommended ranking-quality metric:

- NDCG@10 for most datasets;
- MRR@10 for MS MARCO and TREC-DL profiles.

For each selected dataset, an engine’s score is normalized against the best
observed score for that dataset:

```text
relative[dataset, engine] = engine_score / best_observed_score
```

The report then shows:

- **Mean relative** — average normalized quality across selected datasets;
- **Quality floor** — the engine’s weakest normalized dataset result;
- **Dataset wins** — datasets on which the engine matches the best result;
- **Coverage** — how many selected datasets have a completed result.

An engine is ranked only when it covers every selected dataset. Incomplete
engines remain visible so failures on larger corpora are not hidden. Reference
winners come from the complete report cohort and do not change when an engine
is hidden with a filter.

When several search modes are selected, each matrix cell uses the best result
among those modes and labels the winning mode. Select exactly one mode for a
fixed-mode comparison.

The benchmark currently uses concurrency 1 and one repeat. QPS is therefore the
observed single-client query rate, and one run cannot estimate run-to-run
variance. Treat small quality and latency differences cautiously. A missing
cell means that no completed matching run exists; it does not distinguish an
unsupported operation from an aborted or unattempted run.

## Metric guide

- **NDCG@10 / NDCG@20:** ranking quality near the top, including graded relevance.
- **MRR@10:** how early the first relevant result appears.
- **MAP@10:** whether multiple relevant results appear early and consistently.
- **P@20:** the relevant share of the first 20 results.
- **Recall@100:** the share of all known relevant documents found in the first 100 results.
- **P50:** typical query latency; lower is better.
- **P95 / P99:** slow-request and tail latency; lower is better.
- **QPS:** observed successful queries per second at concurrency 1.

The report includes contextual tooltips for these metrics.

## Outputs

Every completed run is stored in `artifacts/<run_id>/`:

- `metrics.json` — aggregate quality, performance, configuration, and engine-version metadata;
- `per_query.csv` — per-query quality and timing;
- `summary.md` — human-readable run summary;
- `diagnostics.json` — sampled plans, profiles, or backend diagnostics.

`bench report` creates a Markdown comparison and a sibling directory containing
its interactive `index.html`. Use `--markdown-only` when HTML is not needed.

## Development

Run the unit tests with:

```bash
./.venv/bin/pytest -q tests/unit
```

Detailed configuration fields, backend behavior, dataset preparation rules,
artifact internals, and maintenance notes are kept in [`AGENTS.md`](AGENTS.md).
