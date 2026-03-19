# TREC-COVID Benchmark (Manticore + Qdrant)

This CLI benchmarks retrieval quality and performance on TREC-COVID across:
- Manticore full-text (`MATCH(...)`)
- Manticore vector (`FLOAT_VECTOR` + auto-embeddings)
- Qdrant full-text-like sparse retrieval (BM25 in Qdrant sparse vectors)
- Qdrant dense vector retrieval (vectors sourced from Manticore)

It produces reproducible artifacts (`metrics.json`, `per_query.csv`, `summary.md`, diagnostics) per run.

## Dataset

- Source: `beir/trec-covid` via `ir_datasets`.
- Corpus: COVID-19 research documents with `title` and `abstract` fields.
- Queries: official TREC-COVID topics.
  - TREC-COVID topics include multiple text variants (a shorter title-style query and longer descriptive text).
  - Benchmark query policy:
    - full-text-like modes (`manticore fulltext`, `qdrant fulltext`) use the shorter query variant.
    - vector modes (`manticore vector`, `qdrant vector`) use the longer query variant.
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

Prepared outputs:
- `data/prepared/docs.jsonl`
- `data/prepared/topics.jsonl`
- `data/prepared/qrels.tsv`

Note:
- Legacy prepared topic formats are no longer supported. If you have old prepared files, rerun `bench prepare`.

## How It Works and Usage

Run `bench`, then choose the action in the interactive menu.

```bash
bench
```

Main actions:
1. `prepare`: download and prepare corpus/topics/qrels.
2. `index`: build search indexes/tables for selected engine+mode (opens interactive index matrix when selected from `bench`; prints live indexing progress with `processed/total`, percent, inserted rows, retries/failures, elapsed seconds).
3. `run`: execute topic queries with fixed benchmark settings and compute metrics (opens interactive run matrix; variants come from named config profiles and include `PROFILE`, `ENGINE`, `MODE`, `MODEL/OPTS`).
4. `report`: render a combined report from selected artifacts.

Interactive menu controls: `↑/↓`, `ENTER`, `ESC/q` (and `SPACE` where multi-select is available).

Note:
- `cache-populate` is available as a direct CLI command (not a top-level menu item):
  - `bench cache-populate --table trec_covid_vec --model openai/text-embedding-3-large --similarity cosine --vector-from title,abstract`

The index selector combines options discovered from all local `config*.yaml` files (excluding templates).

## Installation

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
They validate core benchmark logic (for example metric calculations and run/report matrix behavior) without requiring live Manticore/Qdrant services.

## Configuration

Config included in the repo:
- `config.yaml`: single source of truth.

`config.yaml.template` is kept as a documented reference with inline comments for all options.

### Embedding API Credentials
- in `config.yaml` under `manticore`:
  - `vector_api_key: "..."` (inline), or
  - `vector_api_key_file: "/absolute/path/to/keyfile"` (recommended).

Example:
```yaml
manticore:
  vector_model: "openai/text-embedding-3-large"
  vector_api_key_file: "/Users/you/openai_key.txt"
```

### Manticore Full-Text Ranking Config
- `manticore.fulltext_options`: raw Manticore RT table options appended to `CREATE TABLE ... )`.
  - default: `morphology='stem_en' index_field_lengths='1'`
  - example: `index_field_lengths='1' stopwords='en' morphology='lemmatize_en_all'`
- `manticore.search_options`: raw SQL `OPTION` list appended to Manticore search queries (including ranker settings).
  - default: `ranker=expr('1000*(bm25a(1.2,0.75)+sum(lcs*user_weight))')`
  - examples: `ranker=bm25`, `ranker=proximity_bm25`, `ranker=bm25, fuzzy=1`
  - applies to both full-text and vector query paths
- full-text queries also set: `idf='plain,tfidf_unnormalized'`

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

### Inheritance Mode
- use top-level named entries (`manticore`, `qdrant`, `manticore_bm25`, ...).
- derived entries define `base: "<entry_name>"` and only overrides.
- any unspecified field is inherited from base.
- keys are flat; loader routes them to the right section automatically.
- optional mode filter: set `type: "vector"` or `type: "full-text"` on an entry (or its base) to expose only that mode in interactive `bench index`/`bench run`.

Example:
```yaml
manticore:
  search_options: "ranker=expr('1000*(bm25a(1.2,0.75)+sum(lcs*user_weight))')"

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
  - options from `manticore.fulltext_options` (default: `morphology='stem_en' index_field_lengths='1'`)
- Manticore vector table (`manticore.vector_table`):
  - same fields + `FLOAT_VECTOR`
  - `KNN_TYPE='hnsw'`
  - `HNSW_SIMILARITY` from config (`manticore.vector_similarity`)
  - optional `quantization` from config (`vector_quantization`)
  - indexing is cache-first; cache misses are auto-embedded by Manticore
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
- Note: native BM25 mode may download model artifacts on first use.

## Engine Setup

### Manticore

Quick start (Docker):
```bash
docker run -d --name trec-manticore \
  -p 9306:9306 -p 9308:9308 \
  -v "$(pwd)/data/manticore_storage:/var/lib/manticore" \
  manticoresearch/manticore:latest

mysql -h 127.0.0.1 -P 9306 -e "SHOW TABLES"
```

Expected reachable endpoints from config:
- SQL: `manticore.host:manticore.sql_port` (default `127.0.0.1:9306`)
- HTTP: `manticore.host:manticore.http_port` (default `127.0.0.1:9308`)

### Qdrant

Run Qdrant as a server (Docker).

Quick start (Docker):
```bash
docker run -d --name trec-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest

curl http://127.0.0.1:6333/
```


## Vector Cache

Vector cache is stored as SQLite file (default):
- `vector_cache/embeddings.sqlite`

This directory is important:
- regenerating cache can take a long time and may cost money (for API-based models),
- keep backups if you rely on expensive embeddings.

Populate cache from existing Manticore vector table:
```bash
bench cache-populate --table trec_covid_vec --model openai/text-embedding-3-large --similarity cosine --vector-from title,abstract
```

Behavior:
- cache key = `md5(normalized_text + model + hnsw_similarity)`,
- normalized text = selected `vector_from` fields joined with single spaces,
- cache can store `NULL` status (embedding unavailable); such docs are skipped during vector indexing and not retried.
- Manticore vector indexing:
  - uses cached vectors when available,
  - auto-embeds only cache misses,
  - syncs produced vectors back into cache automatically after indexing.
- Qdrant vector indexing:
  - uses cache only,
  - fails on cache misses (no auto-embedding path in Qdrant indexing flow).

## Metrics

This benchmark reports two metric groups:
- quality metrics (how relevant the returned ranking is against TREC-COVID qrels),
- performance metrics (how fast and stable query/index execution is).

Configured in `metrics` section:

```yaml
metrics:
  enabled:
    - "ndcg@10"
    - "ndcg@20"
    - "map@10"
    - "p@20"
    - "recall@100"
    - "latency"
    - "qps"
    - "error_rate"
  ndcg_ks: [10, 20]
  map_k: 10
  precision_k: 20
  recall_k: 100
```

Meaning:
- `ndcg@k`: ranking quality at top-k with graded relevance.
- `map@k`: average precision at top-k across topics.
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
