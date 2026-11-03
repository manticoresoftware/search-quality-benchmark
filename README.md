# Search Bench

Search Bench compares retrieval quality across open-source search engines using
public information-retrieval datasets. It runs the same queries and relevance
judgments against each engine, keeps reproducible artifacts, and builds a
standalone interactive HTML report.

Quality is the primary benchmark target. Latency and throughput are included as
supporting context, not as authoritative load-test results.

## Supported engines and search modes

The benchmark currently supports Manticore, Elasticsearch, OpenSearch,
Meilisearch, Typesense, Milvus, Qdrant, MariaDB, SereneDB, Weaviate, and Vespa.

Depending on the engine, it can compare:

- full-text search;
- dense vector search;
- hybrid lexical and vector search.

OpenSearch hybrid runs use native weighted reciprocal-rank fusion (RRF). Their
run labels record the rank constant, branch weights, and candidate depth.
This query-time implementation does not require rebuilding an existing combined
index.

MariaDB hybrid runs execute lexical retrieval, vector retrieval, ranking, and
weighted RRF in one server-side SQL statement built from CTEs and window
functions. Switching from the former client-side fusion does not require
reindexing the combined table.

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
3. `index` — create and populate indexes for selected engine profiles. Shared
   targets expose separate full-text and hybrid builds, so an engine can
   still participate in lexical comparisons when its vector build is impractical.
   Those two destructive alternatives are mutually exclusive in the index TUI.
   In Manticore and Qdrant, matching lexical and dense target names refer to one
   combined physical target. The hybrid build creates both representations and
   supports full-text, vector, and hybrid runs. Rebuilding that target as
   full-text-only removes its dense vectors, so the CLI warns before doing so;
   for Qdrant it also offers to skip the full-text-only action. Non-interactive
   destructive rebuilds require `--yes`. Use the hybrid build to retain all
   modes, or configure distinct target names for isolated full-text/vector
   indexes (Qdrant native hybrid requires a shared collection).
   Each completed task prints its total elapsed indexing time, including
   engine-side synchronization or finalization after document ingestion.
4. `run` — execute benchmark queries and write run artifacts.
5. `report` — compare selected runs in Markdown and standalone HTML.

The interactive menus use arrow keys and `Enter`; multi-select screens also use
`Space`. Press `Esc` or `q` in a submenu to return to the main menu. Press either
key in the main menu to exit `bench`. After a prepare, index, or benchmark run
has started, the TUI rings the terminal bell before returning to the main menu.

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

For TREC-COVID, refresh the prepared topics and incrementally add all three
official query formulations to the cache with:

```bash
bench --config config.yaml prepare --source beir/trec-covid
```

Do not use `--reset-embeddings` for this migration: existing document and
narrative vectors are reusable. The `run` matrix then exposes
`trec-covid-query`, `trec-covid-question`, and `trec-covid-narrative` for every
available full-text, vector, and hybrid profile. A direct run selects one with
`--benchmark-dataset trec-covid-question` (or either of the other two IDs).

## Interactive report

Run `report` from the `bench` menu, or rebuild the repository’s complete report
directly:

```bash
python3 src/bench/interactive_report.py
```

Open [`docs/index.html`](docs/index.html) in a browser. It is a standalone file
with embedded CSS, fonts, JavaScript, benchmark results, and slow-query
samples; it does not require a web server or external assets. The same file is
the published GitHub Pages site (deployed from the `/docs` folder).

The report provides:

- an overall quality comparison and coverage overview;
- a dataset × engine quality matrix;
- cross-dataset consistency and search-mode comparisons;
- a per-dataset quality/latency explorer;
- latency distribution and slow-query inspection;
- a searchable table of measured runs with CSV and JSON export.

Dataset, engine, and search-mode filters can select any subset. Hybrid mode is
split into `Hybrid · native` and `Hybrid · non-native`. Selecting both makes
each engine/dataset cell use the better of its eligible hybrid runs, while
selecting exactly one restricts comparison to engines supporting that
implementation; their selected full-text/vector runs remain eligible on
datasets where no hybrid run was completed. Selecting neither hybrid variant
applies no hybrid constraint.
Selecting both applies no hybrid eligibility constraint either: the better
native/non-native hybrid is used where available, while selected
full-text/vector-only results remain eligible. Filter state is stored in the
URL, so a view can be bookmarked or shared.

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
fixed-mode comparison. The two hybrid choices follow the same rule: when both
native and non-native hybrid are selected, the better hybrid result is used for
each engine and dataset. Non-native hybrid is excluded from the default report
selection because its lexical and vector result sets are fused by the benchmark
client rather than by the engine. The results remain available as an explicit
opt-in and the report shows a comparability warning whenever they are enabled.

TREC-COVID is exposed as three logical datasets over the same physical corpus
and qrels: `trec-covid-query` (short keyword query), `trec-covid-question`
(natural-language question), and `trec-covid-narrative` (long description).
Each can be run independently in full-text, vector, and hybrid mode; hybrid uses
the same selected wording in both branches. They share document embeddings and
engine indexes, while `prepare` incrementally fills any missing query vectors.

Reports keep the three formulations in separate cells. Existing unmarked
TREC-COVID artifacts remain usable: full-text runs are interpreted as
`trec-covid-query`, and vector/hybrid runs as `trec-covid-narrative`. Global
scores first average the three views within the TREC-COVID family, preventing
that one corpus from receiving triple weight. The older `short`/`long`/`same`
identity remains available for other prepared datasets and historical schemas.

Bundled SereneDB vector and hybrid profiles use exact vector scans. SereneDB
26.10.1's ANN path was excluded after reproducible server crashes on repeated
4096-dimensional queries; run artifacts identify the exact-search setting.
SereneDB lexical branches use a documented 10% minimum-term match instead of a
pure OR, which avoids the same release's crash on verbose queries while keeping
all query terms available to BM25. This setting is also shown in artifacts.

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
