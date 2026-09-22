# Search Benchmark (Manticore + Qdrant + Meilisearch + OpenSearch + Elasticsearch + Typesense + MariaDB Vector + Weaviate + Vespa)

This CLI benchmarks retrieval quality and performance on IR datasets across:
- Manticore full-text
- Manticore vector
- Manticore hybrid
- Qdrant full-text-like sparse retrieval (BM25 in Qdrant sparse vectors)
- Qdrant dense vector retrieval (vectors sourced from the shared cache)
- Meilisearch lexical search
- Meilisearch vector search (user-provided vectors from the shared cache)
- Meilisearch hybrid search (query text + cached query vector)
- OpenSearch lexical search
- OpenSearch vector search (user-provided vectors from the shared cache)
- OpenSearch hybrid search (query text + cached query vector)
- Elasticsearch lexical search
- Elasticsearch vector search (user-provided vectors from the shared cache)
- Elasticsearch hybrid search (query text + cached query vector)
- Typesense lexical search
- Typesense vector search (user-provided vectors from the shared cache)
- Typesense hybrid search (query text + cached query vector)
- MariaDB full-text search (`MATCH ... AGAINST`)
- MariaDB vector search (`VECTOR INDEX` + `VEC_DISTANCE`)
- MariaDB hybrid search (weighted reciprocal-rank fusion of full-text and vector rankings)
- Weaviate lexical search
- Weaviate vector search (user-provided vectors from the shared cache)
- Weaviate hybrid search (query text + cached query vector)
- Vespa lexical search
- Vespa vector search (user-provided vectors from the shared cache)
- Vespa hybrid search (query text + cached query vector)

It produces reproducible artifacts (`metrics.json`, `per_query.csv`, `summary.md`, diagnostics) per run.

## Dataset

- Default source: `beir/trec-covid` via `ir_datasets`.
- Additional ready examples: `beir/nfcorpus/test`, `beir/scifact/test`, `beir/arguana`, `beir/scidocs`, `beir/fiqa/test`, `beir/cqadupstack/webmasters`, `msmarco-passage/dev/small`, `msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`, `dl-typo`, `wands`, `hf-beir:theatticusproject/acord:test`.
- Dataset catalogs/docs:
  - [ir-datasets.com](https://ir-datasets.com/)
  - [WANDS on GitHub](https://github.com/wayfair/WANDS)
  - [ACORD on Hugging Face](https://huggingface.co/datasets/theatticusproject/acord)
- Corpus: COVID-19 research documents with `title` and `abstract` fields.
- Queries: official TREC-COVID topics.
  - TREC-COVID topics include multiple text variants (a shorter title-style query and longer descriptive text).
  - Benchmark query policy:
    - full-text-like modes (`manticore fulltext`, `qdrant fulltext`, `meilisearch fulltext`, `opensearch fulltext`, `elasticsearch fulltext`) use the shorter query variant.
    - vector/hybrid modes (`manticore vector`, `qdrant vector`, `meilisearch vector`, `meilisearch hybrid`, `opensearch vector`, `opensearch hybrid`, `elasticsearch vector`, `elasticsearch hybrid`) use the longer query variant.
  - Prepared topic fields are explicit: `query_short` and `query_long`.
  - Example topics from TREC-COVID:
    - Topic `1`
      - short: `coronavirus origin`
      - long: `seeking range of information about the SARS-CoV-2 virus's origin, including its evolution, animal source, and first transmission into humans`
    - Topic `2`
      - short: `coronavirus response to weather changes`
      - long: `seeking range of information about the SARS-CoV-2 virus viability in different weather/climate conditions as well as information related to transmission of the virus in different climate conditions`
- Judgments: official TREC-COVID qrels (query-document relevance labels) used for quality metrics.
  - Labels are graded relevance values from the dataset qrels.
  - Metric interpretation in this benchmark:
    - `ndcg@k`: uses graded relevance values directly.
    - `map@k` and `recall@k`: treat documents with relevance `> 0` as relevant.
  - Qrels are filtered to the prepared document set.

Prepared outputs (source-scoped):
- `data/prepared/<source-slug>/docs.jsonl`
- `data/prepared/<source-slug>/topics.jsonl`
- `data/prepared/<source-slug>/qrels.tsv`

Examples:
- `beir/trec-covid` -> `data/prepared/beir-trec-covid/`
- `beir/nfcorpus/test` -> `data/prepared/beir-nfcorpus-test/`
- `beir/scifact/test` -> `data/prepared/beir-scifact-test/`
- `beir/arguana` -> `data/prepared/beir-arguana/`
- `beir/scidocs` -> `data/prepared/beir-scidocs/`
- `beir/fiqa/test` -> `data/prepared/beir-fiqa-test/`
- `beir/cqadupstack/webmasters` -> `data/prepared/beir-cqadupstack-webmasters/`
- `msmarco-passage/dev/small` -> `data/prepared/msmarco-passage-dev-small/`
- `dl-typo` -> `data/prepared/dl-typo/`
- `wands` -> `data/prepared/wands/`
- `hf-beir:theatticusproject/acord:test` -> `data/prepared/hf-beir-theatticusproject-acord-test/`

Recommended smaller BEIR datasets for faster iteration:
- `beir/scifact/test`
- `beir/arguana`
- `beir/scidocs`
- `beir/fiqa/test`
- `beir/cqadupstack/webmasters`

WANDS:
- source id: `wands`
- upstream: [Wayfair/WANDS](https://github.com/wayfair/WANDS)
- corpus: `42,994` products
- queries: `480`
- qrels: `233,448`
- labels are mapped as graded relevance: `Exact=2`, `Partial=1`, `Irrelevant=0`

DL-Typo:
- source id: `dl-typo`
- uses the public typo queries and qrels from `ielab/CharacterBERT-DR`
- reuses the standard MS MARCO Passage corpus on the document side
- prepared topics store the typo query for benchmarking and also keep the corrected query as `query_clean` for reference

Hugging Face BEIR-style sources:
- format: `hf-beir:<repo_id>:<split>`
- current example in this repo: `hf-beir:theatticusproject/acord:test`
- the loader expects a BEIR-style zip containing `corpus.jsonl`, `queries.jsonl`, and `qrels/<split>.tsv`

Note:
- Legacy prepared topic formats are no longer supported. If you have old prepared files, rerun `bench prepare`.

### MSMARCO Profiles

Recommended MSMARCO passage profiles in this repo:
- default benchmark: `msmarco-passage/dev/small`
- practical cheaper variant: `msmarco_lite_*` entries (`max_docs: 250000`, deterministic subset)
- secondary validation: `msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`

Why this split:
- `dev/small` gives a practical judged query set for day-to-day tuning.
- TREC-DL judged sets provide higher-quality secondary validation on a small query set.

Runtime note:
- all of these still use the full MSMARCO passage corpus (~8.84M passages), so indexing/embedding remains heavy.
- `msmarco_lite_*` entries reduce this by limiting prepared docs (`dataset.max_docs`) while still keeping judged docs (`dataset.keep_qrels_docs: true`).

Example commands:
```bash
bench --config config.yaml prepare
bench --config config.yaml run --engine manticore --mode both --entry msmarco_passage_manticore
bench --config config.yaml run --engine manticore --mode both --entry msmarco_lite_manticore
bench --config config.yaml run --engine manticore --mode both --entry msmarco_trecdl_2019_manticore
bench --config config.yaml run --engine manticore --mode both --entry msmarco_trecdl_2020_manticore
```

### ACORD Profiles

Ready ACORD entries in this repo:
- `acord_manticore_ft`: Manticore full-text with lemmatization and `OR` query mode
- `acord_manticore_1bit`: Manticore vector with 1-bit quantization
- `acord_qdrant_ft`: Qdrant BM25
- `acord_qdrant_bq`: Qdrant dense with binary quantization

Prepared ACORD test split observed in this repo:
- docs: `3,931`
- queries: `57`
- qrels: `61,988`

## How It Works and Usage

Run `bench`, then choose the action in the interactive menu.

```bash
bench
```

Main actions:
1. `engines`: start/stop Docker-backed engines and watch their status/load live.
2. `prepare`: choose one or more dataset sources and download/prepare corpus/topics/qrels.
3. `index`: build search indexes/tables for selected engine+mode (opens interactive index matrix when selected from `bench`; prints live indexing progress with `processed/total`, percent, inserted rows, retries/failures, elapsed seconds).
4. `run`: execute topic queries with fixed benchmark settings and compute metrics (opens interactive run matrix; variants come from named config profiles and include `PROFILE`, `ENGINE`, `MODE`, `MODEL/OPTS`).
5. `report`: render a combined report from selected artifacts.

Interactive menu controls: `↑/↓`, `ENTER`, `ESC/q` (and `SPACE` where multi-select is available).

Note:

The index selector combines options discovered from all local `config*.yaml` files (excluding templates).

`prepare` also supports direct filtering:
- `bench --config config.yaml prepare --source msmarco-passage/dev/small`
- `bench --config config.yaml prepare --entry msmarco_lite_manticore`

## Installation

Requirement: Python `3.11+`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Steps:
1. `python3 -m venv .venv` creates an isolated Python 3 environment in `.venv` for this project.
2. `source .venv/bin/activate` activates that environment so `python`/`pip` point to project-local tools.
3. `pip install -e '.[dev]'` installs this project in editable mode plus dev dependencies (for example `pytest`), so code changes are picked up immediately without reinstall.

Notes:
- On some systems `python` points to Python 2; use `python3`.
- In `zsh`, quote `'.[dev]'` to avoid shell glob expansion.

Unit tests:
```bash
./.venv/bin/pytest -q tests/unit
```
They validate core benchmark logic (for example metric calculations and run/report matrix behavior) without requiring live engine services.

## Configuration

Config included in the repo:
- `config.yaml`: single source of truth.

`config.yaml.template` is kept as a documented reference with inline comments for all options.

### Embedding API Credentials
- in `config.yaml` under `embedding`:
  - `provider: "openai"` or `provider: "openrouter"`
  - `api_key: "..."` (inline), or
  - `api_key_file: "/absolute/path/to/keyfile"` (recommended)
  - `model: "..."` for the embedding model used by `bench prepare` and query embedding fallback

Example:
```yaml
embedding:
  provider: "openrouter"
  model: "qwen/qwen3-embedding-8b"
  api_key_file: "/Users/you/openrouter_key.txt"
```

### Manticore Full-Text Ranking Config
- `manticore.table_profile`: Manticore `CREATE TABLE` profile appended as `profile='<value>'`.
  - default: `relevance`
  - with Manticore `dev`, `profile='relevance'` expands to the recommended relevance settings, including stemming, field lengths, exact words, OR boolean mode, and the BM25A ranker with `avgdl=256`.
- `manticore.fulltext_options`: extra raw Manticore RT table options appended after `table_profile`.
  - default: empty
  - example: `stopwords='en'`
- `manticore.search_options`: raw Manticore search options list (including ranker settings).
  - default: empty; leave it empty to use the table profile ranker
  - examples: `ranker=bm25`, `ranker=proximity_bm25`, `ranker=bm25, fuzzy=1`
  - applies to both full-text and vector query paths
- `manticore.fulltext_query_mode`: how full-text query terms are combined before sending them to `MATCH()`.
  - `and`: explicit term conjunction
  - `or` (default): natural disjunction / broader recall
  - `quorum`: uses Manticore quorum syntax like `"term1 term2 term3"/0.5`
- `manticore.fulltext_query_quorum`: quorum threshold used when `fulltext_query_mode: "quorum"` (default: `0.5`)

### Manticore Vector Quantization/Search Config
- `manticore.vector_quantization`: `""` (off), `8bit`, `1bit`, or `1bitsimple`
- `manticore.vector_search_oversampling`: KNN query oversampling factor
- `manticore.vector_search_rescore`: `1`/`0` to enable/disable rescoring

### Qdrant Dense Quantization/Search Config
- `qdrant.dense_binary_quantization`: enable binary quantization for dense collection
- `qdrant.dense_binary_always_ram`: keep quantized vectors in RAM
- `qdrant.dense_binary_query_encoding`: `default|binary|scalar4bits|scalar8bits`
- `qdrant.dense_search_quantization_rescore`: query-time rescore for quantized search
- `qdrant.dense_search_quantization_oversampling`: query-time oversampling
- `qdrant.dense_search_quantization_ignore`: bypass quantized vectors and use full-precision vectors

### Meilisearch Config
- `meilisearch.host`: server URL (default: `http://127.0.0.1:7700`)
- `meilisearch.api_key` / `meilisearch.api_key_file`: optional Meilisearch API key
- `meilisearch.index_name`: combined index name used for full-text, vector, and hybrid search
- `meilisearch.searchable_attributes`: fields searched by lexical and hybrid search
- `meilisearch.ranking_rules`: lexical ranking rules used by Meilisearch
- `meilisearch.stop_words`: optional stop word list
- `meilisearch.synonyms`: optional synonym map
- `meilisearch.typo_tolerance`: enable/disable typo tolerance
- `meilisearch.embedder_name`: name of the user-provided embedder configured on the vector index
- `meilisearch.vector_index`: suffix for the vector index name
- `meilisearch.hybrid_semantic_ratio`: hybrid search blend between lexical and semantic search
- `meilisearch.binary_quantized`: enable Meilisearch 1-bit vector storage for the configured embedder

### OpenSearch Config
- `opensearch.url`: server URL (default: `http://127.0.0.1:9200`)
- `opensearch.username` / `opensearch.password`: optional basic auth
- `opensearch.api_key` / `opensearch.api_key_file`: optional API key
- `opensearch.index_name`: combined index name used for full-text, vector, and hybrid search
- `opensearch.text_fields`: text fields used by lexical and hybrid search
- `opensearch.vector_field`: vector field name used for kNN search
- `opensearch.space_type`: OpenSearch kNN similarity (`cosinesimil`, etc.)
- `opensearch.hybrid_text_boost`: lexical clause boost in hybrid search
- `opensearch.hybrid_vector_boost`: vector clause boost in hybrid search

### Elasticsearch Config
- `elasticsearch.url`: server URL (default: `http://127.0.0.1:9201`)
- `elasticsearch.username` / `elasticsearch.password`: optional basic auth
- `elasticsearch.api_key` / `elasticsearch.api_key_file`: optional API key
- `elasticsearch.index_name`: combined index name used for full-text, vector, and hybrid search
- `elasticsearch.text_fields`: text fields used by lexical and hybrid search
- `elasticsearch.vector_field`: vector field name used for kNN search
- `elasticsearch.similarity`: `dense_vector` similarity (`cosine`, etc.)
- `elasticsearch.hybrid_text_boost`: lexical clause boost in hybrid search
- `elasticsearch.hybrid_vector_boost`: vector clause boost in hybrid search

### Typesense Config
- `typesense.url`: server URL (default: `http://127.0.0.1:8108`)
- `typesense.api_key` / `typesense.api_key_file`: Typesense API key
- `typesense.collection_name`: combined collection name used for full-text, vector, and hybrid search
- `typesense.text_fields`: text fields used by lexical and hybrid search
- `typesense.vector_field`: vector field name in the collection schema
- `typesense.vector_distance`: vector distance metric for the collection schema
- `typesense.hybrid_alpha`: hybrid blend parameter passed to `vector_query`

### Weaviate Config
- `weaviate.url`: server URL (default: `http://127.0.0.1:8080`)
- `weaviate.api_key` / `weaviate.api_key_file`: optional API key
- `weaviate.collection_name`: combined collection name used for full-text, vector, and hybrid search
- `weaviate.text_fields`: text fields used by lexical and hybrid search
- `weaviate.vector_field`: vector field name used when importing cached embeddings
- `weaviate.vector_distance`: vector distance metric for the collection schema
- `weaviate.hybrid_alpha`: hybrid alpha passed to Weaviate hybrid search

### Vespa Config
- `vespa.url`: query/document endpoint (default: `http://127.0.0.1:8081`)
- `vespa.deploy_url`: configserver deploy endpoint (default: `http://127.0.0.1:19071`)
- `vespa.collection_name`: logical collection name used for full-text, vector, and hybrid search
- `vespa.text_fields`: text fields indexed for BM25-style lexical and hybrid search
- `vespa.vector_field`: tensor field name used for dense retrieval
- `vespa.vector_distance`: Vespa distance metric (`angular`, etc.)
- `vespa.hybrid_alpha`: hybrid alpha baked into the generated Vespa rank profile
- `vespa.app_name`: generated application name used during deploy
- `vespa.namespace`: Vespa document namespace
- `vespa.package_dir`: local generated application package/registry directory

### Inheritance Mode
- use top-level named entries (`manticore`, `qdrant`, `manticore_bm25`, ...).
- derived entries define `base: "<entry_name>"` and only overrides.
- any unspecified field is inherited from base.
- keys are flat; loader routes them to the right section automatically.
- optional mode filter: set `type: "vector"`, `type: "hybrid"`, or `type: "full-text"` on an entry (or its base) to expose only that mode in interactive `bench index`/`bench run`.

Example:
```yaml
manticore:
  table_profile: "relevance"

manticore_bm25:
  base: "manticore"
  search_options: "ranker=bm25"

qdrant:
  dense_collection: "trec_covid_qdrant_dense"

qdrant_bq:
  base: "qdrant"
  dense_collection: "trec_covid_qdrant_dense_bq"
  dense_binary_quantization: true

manticore_vector_only:
  base: "manticore"
  type: "vector"
```

### Configured Index/Collection Defaults
- Manticore full-text table (`manticore.fulltext_table`):
  - columns `id, doc_id, title, abstract, doi, date`
  - `profile='relevance'` by default, plus any options from `manticore.fulltext_options`
- Manticore vector table (`manticore.vector_table`):
  - same fields + `FLOAT_VECTOR`
  - `KNN_TYPE='hnsw'`
  - `HNSW_SIMILARITY` from config (`manticore.vector_similarity`)
  - optional `quantization` from config (`vector_quantization`)
  - indexing is cache-first; cache misses are an error until you run `bench prepare`
- Qdrant collections:
  - sparse BM25 collection name: `qdrant.bm25_collection`
  - dense collection name: `qdrant.dense_collection`
- Qdrant BM25 collection defaults (`qdrant.bm25_collection`):
  - points keyed by document `id`
  - native Qdrant sparse BM25 model (`Qdrant/bm25`) is used for indexing/querying
  - sparse vector field is handled by Qdrant server-side inference
  - payload fields: `doc_id`, `title`, `abstract`, `doi`, `date`
- Qdrant dense collection defaults (`qdrant.dense_collection`):
  - points keyed by document `id`
  - dense vector field from `qdrant.dense_vector_name` (default `dense`)
  - distance metric: cosine
  - vector dimensionality inferred from cached vectors
  - payload fields: `doc_id`, `title`, `abstract`, `doi`, `date`
  - vectors are loaded from local vector cache
- Meilisearch indexes:
  - combined index name: `meilisearch.index_name`
  - lexical, vector, and hybrid all use the same physical index
  - lexical and hybrid use `searchable_attributes`, `ranking_rules`, `stop_words`, `synonyms`, and `typo_tolerance`
  - vector and hybrid use a `userProvided` embedder and load document vectors from the local SQLite vector cache
- OpenSearch indexes:
  - combined index name: `opensearch.index_name`
  - lexical, vector, and hybrid all use the same physical index
  - vector and hybrid use a `knn_vector` field and load document vectors from the local SQLite vector cache
- Elasticsearch indexes:
  - combined index name: `elasticsearch.index_name`
  - lexical, vector, and hybrid all use the same physical index
  - vector and hybrid use a `dense_vector` field and load document vectors from the local SQLite vector cache
- Typesense collections:
  - combined collection name: `typesense.collection_name`
  - lexical, vector, and hybrid all use the same physical collection
  - vector and hybrid load document vectors from the local SQLite vector cache
- Weaviate collections:
  - combined collection name: `weaviate.collection_name`
  - lexical, vector, and hybrid all use the same physical collection/class
  - vector and hybrid load document vectors from the local SQLite vector cache
- Vespa collections:
  - combined logical collection name: `vespa.collection_name`
  - lexical, vector, and hybrid all use the same generated Vespa schema/document type
  - indexing auto-generates and redeploys a local Vespa application package
  - vector and hybrid load document vectors from the local SQLite vector cache
- Note: native BM25 mode may download model artifacts on first use.

## Engine Setup

Engine containers are defined in:
- `compose.yaml`

Use the interactive engine window:
```bash
bench engines
```

or open `bench` and choose:
- `Engines`

What the engine window does:
- shows all configured engines as a list
- shows whether each engine is `running`, `starting`, `stopping`, or `stopped`
- shows readiness, uptime, and current Docker load
- lets you start/stop the selected engine with `ENTER`
- refreshes every second

Container lifecycle policy:
- engines are started with Docker Compose services from `compose.yaml`
- when you stop an engine from the UI, its container is removed
- persistent state is kept in the bind-mounted `data/...` directories

The services currently managed there are:
- `manticore`
- `qdrant`
- `meilisearch`
- `opensearch`
- `elasticsearch`
- `typesense`
- `mariadb`
- `weaviate`
- `vespa`

Notes:
- OpenSearch still needs the `opensearch-knn` plugin, and the provided service image includes it.
- Before starting OpenSearch for the first time, prepare the host data directory:
  ```bash
  mkdir -p data/opensearch_data
  chmod -R 777 data/opensearch_data
  ```
  OpenSearch itself runs as a non-root user, so the bind-mounted host directory must already be writable.
- Before starting Elasticsearch for the first time, prepare the host data directory:
  ```bash
  mkdir -p data/elasticsearch_data
  chmod -R 777 data/elasticsearch_data
  ```
  The bundled service maps Elasticsearch to `http://127.0.0.1:9201`.
- Typesense uses the local development API key `xyz` by default in both `compose.yaml` and `config.yaml`.
- MariaDB uses `127.0.0.1:3307`, database `search_bench`, and local development credentials `bench`/`bench` by default.
- Weaviate is exposed at `http://127.0.0.1:8080` by the bundled Compose service.
- Vespa is exposed at `http://127.0.0.1:8081`, with deploy API at `http://127.0.0.1:19071`.
- Before starting Vespa for the first time, prepare the host bind-mount directories:
  ```bash
  mkdir -p data/vespa_data data/vespa_logs
  sudo chown -R 1000:1000 data/vespa_data data/vespa_logs
  chmod -R 755 data/vespa_data data/vespa_logs
  ```
  The Vespa container runs as user `vespa` (`1000:1000`), so both mounted directories must be writable by that uid/gid.
- Meilisearch lexical search is not BM25 in the same sense as Manticore or Qdrant sparse BM25, so treat it as a separate retrieval model in reports.


## Vector Cache

Vector cache is stored as SQLite files referenced by profile `vector_cache_path`.

Recommended defaults in this repo:
- `vector_cache/trec-covid-openai.sqlite`
- `vector_cache/msmarco-passage-embeddings.sqlite`
- `vector_cache/msmarco-lite-openai.sqlite`

This directory is important:
- regenerating cache can take a long time and may cost money (for API-based models),
- keep backups if you rely on expensive embeddings.

Populate cache with:
```bash
bench --config config.yaml prepare --source beir/trec-covid
```

Behavior:
- cache key = `md5(normalized_text + model + hnsw_similarity)`,
- normalized text = selected `vector_from` fields joined with single spaces,
- query embeddings use the same cache (same keying by normalized text + model + similarity)
- Manticore vector indexing:
  - uses cache only,
  - fails on cache misses,
  - reuses the same cached document/query vectors as the other engines for fair comparison.
- Qdrant vector indexing:
  - uses cache only,
  - fails on cache misses.
- Meilisearch vector indexing:
  - uses cache only,
  - fails on cache misses,
  - reuses the same cached document/query vectors as the other engines for fair comparison.
- OpenSearch vector indexing:
  - uses cache only,
  - fails on cache misses.
- Typesense vector indexing:
  - uses cache only,
  - fails on cache misses.
- MariaDB vector indexing:
  - requires MariaDB 11.7 or newer (the bundled service uses MariaDB 11.8 LTS),
  - uses `index_timeout_seconds` for ingestion; index DDL uses an unbounded socket and is actively monitored every `index_progress_interval_seconds`, so a healthy long-running build is not disconnected,
  - reports the server's `PROCESSLIST` stage, stage progress, state, and connection ID while FULLTEXT and VECTOR indexes are built,
  - sends cached vectors as packed `float32` values instead of server-side JSON parsing,
  - the bundled container defaults to a 32 GiB InnoDB buffer pool and an 8 GiB per-index HNSW cache; override these with `MARIADB_INNODB_BUFFER_POOL_SIZE` and `MARIADB_MHNSW_MAX_CACHE_SIZE`,
  - creates a combined InnoDB table with FULLTEXT and VECTOR indexes,
  - uses cache only and fails on cache misses,
  - uses weighted reciprocal-rank fusion for hybrid queries (`hybrid_rrf_k`, `hybrid_alpha`).

## Metrics

This benchmark reports two metric groups:
- quality metrics (how relevant the returned ranking is against dataset qrels),
- performance metrics (how fast and stable query/index execution is).

Configured in `metrics` section:

```yaml
metrics:
  enabled:
    - "ndcg@10"
    - "ndcg@20"
    - "map@10"
    - "mrr@10"
    - "p@20"
    - "recall@100"
    - "latency"
    - "qps"
    - "error_rate"
  ndcg_ks: [10, 20]
  map_k: 10
  mrr_k: 10
  precision_k: 20
  recall_k: 100
```

Meaning:
- `ndcg@k`: ranking quality at top-k with graded relevance.
- `map@k`: average precision at top-k across topics.
- `mrr@k`: mean reciprocal rank at top-k (first relevant hit rank quality).
- `p@k`: precision at top-k (fraction of top-k docs that are relevant).
- `recall@k`: fraction of all relevant docs found in top-k.
- `latency_p50/p95/p99`: median and tail query latency.
- `qps`: successful queries per second.
- `error_rate`: failed queries / total queries.

`ndcg@20` is supported through `metrics.ndcg_ks`.

### NDCG@k
Asks: "Did you rank the *most* useful items highest?"
  - In this toy setup, think of graded usefulness like: `A=high`, `B=medium`, `C=low`.
  - It cares about relevance levels, not just yes/no relevance.
  - **Higher `ndcg@k` means better ordering quality near the top.**
```mermaid
flowchart TB
  subgraph L["Low NDCG (bad order)"]
    direction TB
    L1["1 · X (irrelevant)"]
    L2["2 · C (low)"]
    L3["3 · A (high)"]
    L4["4 · Z (irrelevant)"]
    L5["5 · B (medium)"]
    L1 --> L2 --> L3 --> L4 --> L5
  end
  subgraph H["High NDCG (good order)"]
    direction TB
    H1["1 · A (high)"]
    H2["2 · B (medium)"]
    H3["3 · C (low)"]
    H4["4 · X (irrelevant)"]
    H5["5 · Z (irrelevant)"]
    H1 --> H2 --> H3 --> H4 --> H5
  end
  classDef g0 fill:#e6edf5,stroke:#e6edf5,color:#111;
  classDef g1 fill:#cfdceb,stroke:#cfdceb,color:#111;
  classDef g2 fill:#9fc0df,stroke:#9fc0df,color:#111;
  classDef g3 fill:#5f9fd7,stroke:#5f9fd7,color:#111;
  class L1,L4,H4,H5 g0;
  class L2,H3 g1;
  class L5,H2 g2;
  class L3,H1 g3;
```

Color legend (for all diagrams in this section):
```mermaid
flowchart LR
  G0["irrelevant"]
  G1["low relevance"]
  G2["relevant"]
  G3["high relevance"]
  G0 --- G1 --- G2 --- G3
  classDef g0 fill:#e6edf5,stroke:#e6edf5,color:#111;
  classDef g1 fill:#cfdceb,stroke:#cfdceb,color:#111;
  classDef g2 fill:#9fc0df,stroke:#9fc0df,color:#111;
  classDef g3 fill:#5f9fd7,stroke:#5f9fd7,color:#111;
  class G0 g0;
  class G1 g1;
  class G2 g2;
  class G3 g3;
```

### MAP@k
Asks: "Did useful things appear early, not buried?"
  - In the **low-MAP diagram** (`X, Y, Z, A, B`), useful videos `A` and `B` appear late (ranks 4 and 5).
  - In the **high-MAP diagram** (`A, B, X, Y, Z`), they appear early (ranks 1 and 2).
  - Early useful hits give better precision scores.
  - **Higher `map@k` means relevant docs appear earlier and more consistently.**
```mermaid
flowchart TB
  subgraph ML["Low MAP (late hits)"]
    direction TB
    ML1["1 · X"]
    ML2["2 · Y"]
    ML3["3 · Z"]
    ML4["4 · A"]
    ML5["5 · B"]
    ML1 --> ML2 --> ML3 --> ML4 --> ML5
  end
  subgraph MH["High MAP (early hits)"]
    direction TB
    MH1["1 · A"]
    MH2["2 · B"]
    MH3["3 · X"]
    MH4["4 · Y"]
    MH5["5 · Z"]
    MH1 --> MH2 --> MH3 --> MH4 --> MH5
  end
  classDef noRel fill:#e6edf5,stroke:#e6edf5,color:#111;
  classDef yesRel fill:#6ea8dc,stroke:#6ea8dc,color:#111;
  class ML1,ML2,ML3,MH3,MH4,MH5 noRel;
  class ML4,ML5,MH1,MH2 yesRel;
```

### MRR@k
Asks: "How early is the first relevant result?"
  - It only looks at the **first** relevant hit.
  - Formula intuition: `RR = 1 / (rank of first relevant)`, then average over queries.
  - If first relevant is at rank 1, score is `1.0`; at rank 5, score is `0.2`.
  - **Higher `mrr@k` means users see a good answer sooner.**
```mermaid
flowchart TB
  subgraph MRL["Low MRR (first hit is late)"]
    direction TB
    MRL1["Top-5: X, Y, Z, W, A"]
    MRL2["First relevant: A at rank 5"]
    MRL3["RR = 1/5 = 0.20"]
    MRL1 --> MRL2 --> MRL3
  end
  subgraph MRH["High MRR (first hit is early)"]
    direction TB
    MRH1["Top-5: A, X, Y, Z, W"]
    MRH2["First relevant: A at rank 1"]
    MRH3["RR = 1/1 = 1.00"]
    MRH1 --> MRH2 --> MRH3
  end
  classDef low fill:#dce8f4,stroke:#dce8f4,color:#111;
  classDef high fill:#79addd,stroke:#79addd,color:#111;
  class MRL1,MRL2,MRL3 low;
  class MRH1,MRH2,MRH3 high;
```

### P@k
Asks: "Out of the top-`k` results, how many are relevant?"
  - Example with `k=5` and ranked list `X, A, Y, B, Z`: only `A` and `B` are relevant.
  - So `P@5 = 2/5 = 0.40`.
  - **Higher `p@k` means cleaner top-`k` results (less noise near the top).**
```mermaid
flowchart TB
  subgraph PL["Low P@5"]
    direction TB
    PL1["Top-5: X, A, Y, B, Z"]
    PL2["Relevant in top-5: {A,B}"]
    PL3["P@5 = 2/5 = 0.40"]
    PL1 --> PL2 --> PL3
  end
  subgraph PH["High P@5"]
    direction TB
    PH1["Top-5: A, B, C, X, Y"]
    PH2["Relevant in top-5: {A,B,C}"]
    PH3["P@5 = 3/5 = 0.60"]
    PH1 --> PH2 --> PH3
  end
  classDef low fill:#dce8f4,stroke:#dce8f4,color:#111;
  classDef high fill:#79addd,stroke:#79addd,color:#111;
  class PL1,PL2,PL3 low;
  class PH1,PH2,PH3 high;
```

### Recall@k
Asks: "Did you find most useful things?"
  - In the toy example: found `2` out of `3` useful videos in top-5 (`Recall@5 = 2/3`).
  - **Higher `recall@k` means you miss fewer relevant docs in top-`k`.**
  - TREC-COVID note: `Recall@100` can look numerically low even for decent systems, because each topic often has hundreds of relevant documents in qrels. With only top-100 retrieved docs, the score is naturally compressed (for example, values around `0.08-0.14` are normal in this benchmark setup).
```mermaid
flowchart TB
  subgraph RL["Low Recall"]
    direction TB
    RL1["Relevant: {A,B,C}"]
    RL2["Found in top-k: {A,B}"]
    RL3["Missed: {C}"]
    RL4["Recall@k = 2/3"]
    RL1 --> RL2 --> RL3 --> RL4
  end
  subgraph RI["Ideal Recall"]
    direction TB
    RI1["Relevant: {A,B,C}"]
    RI2["Found in top-k: {A,B,C}"]
    RI3["Missed: {}"]
    RI4["Recall@k = 3/3 = 1.0"]
    RI1 --> RI2 --> RI3 --> RI4
  end
  classDef low fill:#dce8f4,stroke:#dce8f4,color:#111;
  classDef ideal fill:#79addd,stroke:#79addd,color:#111;
  class RL1,RL2,RL3,RL4 low;
  class RI1,RI2,RI3,RI4 ideal;
```

## Known Limitations

### Non-Embeddable Documents

Some documents may have `NULL` in vector cache and are skipped in vector indexing.
Typical causes:
- input exceeds embedding model context limit,
- provider-side response cannot be parsed as a vector for specific inputs.

In those cases, full-text and vector row counts can differ slightly by design.

### Shared Query Embedding Source (Parity Constraint)

Vector query embeddings are intentionally shared across engines in this benchmark.

Reason:
- we want Manticore and Qdrant vector runs to use identical query vectors for fair comparison.

Implementation:
- both Manticore and Qdrant vector queries use explicit query vectors,
- query vectors are looked up in SQLite vector cache first,
- cache misses are generated via `ManticoreClient.embed_texts(...)` and saved back to the same SQLite cache.

Note:
- this is a benchmark design choice for parity, not a statement about Qdrant capabilities.

## Output Paths

Per benchmark run:
- `artifacts/<run_id>/metrics.json`: aggregated run-level metrics (quality + latency/QPS/error).
- `artifacts/<run_id>/per_query.csv`: one row per evaluated query with per-query quality and timing fields.
- `artifacts/<run_id>/summary.md`: human-readable single-run report.
- `artifacts/<run_id>/diagnostics.json`: sampled query diagnostics (query plans/profiles or backend debug details).

Combined report (`bench report`):
- default output path: `artifacts/report-YYYYMMDD-HHMMSS.md`
- custom output path: use `bench report --output <path>`
