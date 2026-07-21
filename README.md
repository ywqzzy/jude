<p align="center">
  <img src="assets/jude-logo.svg" alt="jude" width="440">
</p>

<h3 align="center">Take a sad song and make it better.</h3>

<p align="center">
  <b>jude</b> is the distributed data engine for <b>LLM training &amp; RAG</b> — curate, index,
  and retrieve web-scale multimodal data, all on stock <b>DuckDB</b> with a <b>Rust</b> core.
</p>

<p align="center">
  <a href="https://github.com/ywqzzy/jude/actions/workflows/ci.yml"><img src="https://github.com/ywqzzy/jude/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/built%20on-DuckDB-black.svg" alt="Built on DuckDB">
  <img src="https://img.shields.io/badge/scales%20on-Ray-028cf0.svg" alt="Scales on Ray">
</p>

<p align="center">
  <a href="README.zh.md">中文</a> ·
  <a href="#capabilities">Capabilities</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/vector_retrieval_report.zh.md">Benchmarks</a>
</p>

---

> Named after the Beatles' *Hey Jude* — *"take a sad song and make it better."*
> That's the job: take raw, messy, web-scale data and make it **better** —
> cleaner, deduplicated, higher-quality, ready to train on.

### Why jude

- **One engine, the whole pipeline.** SQL analytics, LLM data curation, and
  vector/full-text retrieval in a *single* distributed runtime — no glue code
  stitching three systems together.
- **Fast where it counts.** 1M × 768-d vector search at **182 QPS / 99.9% recall**
  on a single node; curation and retrieval both scale near-linearly across workers.
- **Rust core, unforked DuckDB.** The engine and distributed scheduling are Rust;
  DuckDB is stock; Python is a thin shim. No GIL bottleneck, no vendor lock-in.
- **Multimodal-native.** Image / audio / video / document as first-class columns
  with a fluent decode-and-transform API.
- **Bring your own model.** jude stops at the inference boundary — it's the data
  engine *around* your model, not another inference server.

---

## Capabilities

### 🗄️ SQL & the relational core
A DuckDB-compatible `Connection` / `Relation` API backed by a real `LogicalPlan`
operator DAG (not SQL-string munging — SQL is one lowering).

```python
import jude
con = jude.connect()

# Full DuckDB SQL: joins, window functions, CTEs, aggregates, set ops
con.sql("""
    SELECT category, count(*) AS n, avg(score) AS mean
    FROM read_parquet('docs/*.parquet')
    WHERE year >= 2020
    GROUP BY category HAVING n > 100
    ORDER BY mean DESC
""").show()

# Or the DataFrame-style relational API (lazy LogicalPlan; SQL is a lowering)
r = (con.from_parquet("docs/*.parquet")
        .filter("year >= 2020")
        .aggregate("category", "count(*) AS n, avg(score) AS mean"))
r.to_arrow()
```

- CSV / Parquet / JSON / Arrow / Lance / Hive / Iceberg sources.
- Streaming execution: results stream as Arrow `RecordBatch`es (bounded memory).
- Out-of-process **UDFs** — a Rust subprocess pool that releases the GIL
  (`Python::detach`) for real CPU parallelism (scalar, table/generator,
  flat-map, and aggregate UDFs).

### 🌐 Distributed query engine (Rust-orchestrated, over Ray)
The same SQL / relational programs run distributed over **stock** DuckDB — no
forked binding. The scheduling **brain** is in Rust (`WorkerManager`, split
assigners, resource admission control, cross-query bin-packing); Python only
forwards Ray RPCs.

```python
from jude.runners.ray import RayRunner
runner = RayRunner(num_workers=8)
result = runner.collect(relation)          # partitioned, parallel, merged
```

- **Streaming stage-DAG executor** — cuts a plan at shuffle boundaries
  (Aggregate / Join / Distinct / Order / SetOp) into pipelined stages; map and
  reduce overlap.
- **Distributed joins & aggregations** — hash-shuffle exchange flows Arrow
  through the Ray object store (`distributed_join`, streaming variant, two-phase
  aggregate); deterministic FNV-1a shuffle routing.
- **Robustness** — query-level retry: on a worker fault the whole distributed
  read re-executes (`RAY_MAX_QUERY_RETRIES`), rather than actor-level recovery.
- **Resource-aware scheduling** — GPU / memory / object-store admission control
  and worst-fit bin-packing across concurrent queries, in Rust.
- **Distributed data sources** — generator-backed streaming scans, distributed
  Hive read, distributed Lance write + distributed vector-index build.

### 🔎 Vector & full-text retrieval — RAG at scale
**Single-node** (1M × 768-d, top-100, measured — see the report):

| method | recall | QPS |
|---|---|---|
| exact brute force (1 core) | 100% | 1.3 |
| `knn_rerank` (IVF + exact re-rank) | 99.9% | 137 |
| `knn_ann_resident` (IVF ids + in-RAM re-rank) | 99.9% | **182** |

- **All Lance index types**: IVF_FLAT / IVF_SQ / IVF_PQ / IVF_HNSW_{FLAT,SQ,PQ},
  with in-memory index caching (`set_index_cache_size`).
- **Use-case operators**: filtered ANN (metadata pre-filter pushed into the index
  scan), range/threshold search (dedup / entity resolution), MMR diversification.
- **Full-text**: BM25 over Lance inverted indexes; **hybrid** (vector + BM25,
  RRF-fused).

**Distributed retrieval** (fan out across shards, merge on the driver):

| method | what |
|---|---|
| `distributed_knn_resident` | exact, data resident on workers, 100% recall |
| `distributed_knn_resident_batch` | batched — aggregate QPS scales ~linearly with workers |
| `distributed_ann_knn` | sharded ANN (each shard its own index) — billion-scale |
| `distributed_ann_knn_routed` | **cluster-routed** — query touches only the nearest-centroid shards |
| `distributed_fts` | sharded BM25 full-text |
| `distributed_hybrid` | distributed vector + FTS, RRF-fused |

Batched throughput scales near-linearly (1→8 workers ≈ 5.3×); filtered ANN
(`where=`) works in the distributed path too. Full benchmark matrix (methods ×
index types × workers): [`docs/vector_retrieval_report.zh.md`](docs/vector_retrieval_report.zh.md).

### 🖼️ Multimodal
Native `TensorType` (Arrow `fixed_shape_tensor`) plus Image / Audio / Video /
Document columns, with a Daft-style fluent expression API and batch decoders.

```python
import jude
rel = con.from_arrow(table)

# fluent multimodal expressions — decode/resize/crop/encode in the plan
rel = rel.with_column("img", jude.mm("bytes").image.decode().image.resize(224, 224))
rel = rel.with_column("wav", jude.mm("audio").audio.decode(sample_rate=16000, mono=True))
```

- **Image**: `.image.decode().resize(w,h).crop(...).encode("PNG")` → tensors.
- **Audio**: `.audio.decode(sample_rate=, mono=)` with resampling.
- **Video**: frame extraction / streaming decode; **Document**: PDF → per-page.
- **Ingestion sources** for image/audio/video/document, and batch decoders
  (`decode_image_batch`, `decode_audio_batch`, `decode_video_batch`,
  `decode_document_batch`) — all runnable distributed on cosmos-xenna.

### 🧹 LLM data curation — the differentiator
The operators that shape a training set, single-node **and** distributed
(`jude.curate` / `jude.curate_dist`):

- **Deduplication** — exact (SHA-256), fuzzy (MinHash-LSH, bands auto-calibrated
  to the Jaccard threshold; distributed form recall-matches single-node via
  global connected-components), semantic (greedy non-transitive SemDeDup +
  distributed k-means clustering), and **exact-substring** (Lee et al. rolling-hash:
  strips shared passages/boilerplate across differing docs).
- **Web-corpus cleaning** — C4-style line filtering, cross-document line dedup,
  mojibake repair + Unicode NFC normalization.
- **Quality filtering** — Gopher/C4 heuristics (stopword gate, repeated-n-gram,
  digit/symbol ratios), **model-based** classifiers as a batched stage
  (`jude.model_stage`: bring a CPU fastText/ONNX model, a remote vLLM/API
  endpoint, or any callable — GPU optional), token-aware length gates, language ID.
- **PII** — detect / redact (email, URL, IPv4, SSN, Luhn-validated credit cards).
- **Decontamination** — dilution-resistant benchmark-coverage overlap against eval sets.
- **Chunking** — fixed / recursive, for tokenization and RAG.
- **Training-ready output** — `jude.tokenize`: tokenize (pluggable byte /
  tiktoken / HF), pack into fixed-length sequences (doc boundaries + EOS), and
  write token shards (Lance or memory-mappable `.bin`/`.idx.json`).
- **Profiling** — `jude.profile`: one-pass corpus stats with HyperLogLog
  cardinality (dup-rate), length percentiles, language mix — O(1) memory at scale.

Rust hot loops with cosmos-xenna stage pipelines; every operator has a
distributed form.

### 💾 Storage & versioning
Lance (read + single-machine & distributed write, git-like **branches/tags** for
data versioning, vector + FTS indexes), plus Hive and Iceberg read.
**Object-store IO** (`jude.storage`): read/write parquet/csv/json (incl. `.gz`)
over any fsspec URL — `s3://` / `gs://` / local / `memory://` — so sources and
sinks point at S3/GCS/MinIO (creds via `storage_options`). **WARC/WET ingest**
(`jude.warc`): stream CommonCrawl archives into Arrow.

### 📊 Observability
A GIL-free Rust metrics registry (per query / stage / UDF / cluster), a durable
audit log (redb), and a React dashboard. `python -m jude.observe` serves it and
attaches to the Ray cluster. Endpoints: `/api/metrics` (full JSON snapshot),
`/api/summary` (rollups: latency p50/p95/p99, rows/sec, task progress,
**curation keep-rate**), and `/api/prometheus` (Prometheus text → Grafana).
**Data-quality observability**: wrap curation ops with `observe.curate(op, rows_in)`
to track rows in→out / removed / keep-rate per operator.

---

## Quickstart

```python
import jude

con = jude.connect()
con.sql("SELECT 42 AS answer").show()

# curation: fuzzy-dedup a corpus
from jude import curate
clean = curate.fuzzy_dedup(corpus, threshold=0.7)   # bands auto-calibrated to threshold

# training-ready output: clean text -> tokens -> packed sequences -> shards
from jude import tokenize as tk
toks = tk.tokenize(clean, tokenizer="bytes")         # or a tiktoken / HF tokenizer
packed = tk.pack_sequences(toks, seq_len=2048)       # doc boundaries + EOS
tk.write_token_shards(packed, "train_shard", fmt="bin")   # memmap-able .bin/.idx.json

# vector search: build an index, query with in-RAM rerank
from jude import vector
jude.connect().create_lance_vector_index("emb.lance", "v",
                                          index_type="IVF_SQ", metric="cosine")
hits = vector.knn_ann_resident("emb.lance", "v", query, k=100, nprobes=16)
```

### Running distributed (Ray)

Point jude at a Ray cluster (or let it start a local one) and the *same* relational
programs execute in parallel — jude partitions, schedules (in Rust), and merges.
**The simplest form is one SQL string:** `runner.collect()` reads the query's stage
DAG from the Rust planner and auto-routes it (ORDER BY → distributed sort, DISTINCT
→ distributed distinct, scan/filter/project → parallel partition scan) — no need to
hand-pick a distributed op.

```python
import ray; ray.init(address="auto")          # attach to your cluster (or omit for local)
from jude.runners.ray import RayRunner
runner = RayRunner(num_workers=8)              # or num_gpus_per_worker=1 for GPU stages

# one SQL string, executed distributed (auto-routed by the Rust stage planner)
table = runner.collect(con.sql("""
    SELECT category, count(*) AS n
    FROM read_parquet('s3://bucket/docs/*.parquet')
    WHERE year >= 2020
    GROUP BY category ORDER BY n DESC
"""))

# the relational API works the same way
rel = con.from_parquet("s3://bucket/docs/*.parquet").filter("year >= 2020")
table = runner.collect(rel)

# or drive a specific distributed op directly
runner.distributed_sort(rel, ["score DESC"])
runner.distributed_top_k(rel, ["score DESC"], 100)
runner.distributed_write_lance(rel, "out.lance", mode="overwrite",
                               vector_index={"column": "v", "index_type": "IVF_SQ"})

# distributed curation + vector search take the same runner
from jude import curate_dist, vector
curate_dist.dist_fuzzy_dedup(table, runner=runner, threshold=0.7, bands=16)
vector.distributed_ann_knn(shard_paths, "v", query, k=100, runner=runner)

# streaming data sources read distributed too
rel = jude.datasource.read(my_source, distributed=True)
```

Or set a process-global runner once and every op picks it up:
`jude.runners.set_runner_local(num_workers=8)` (or assign a `RayRunner`).

### Multimodal pipelines (cosmos-xenna stages)

Chain decode + curation + your model as **stages**; the pipeline you develop
locally (`engine="local"`, order-preserving) runs unchanged distributed on
cosmos-xenna (`engine="cosmos"`), with per-stage CPU/GPU/batch resourcing:

```python
from jude.pipeline import RelationPipeline
from jude.sources import ImageFileSource

out = (RelationPipeline.from_source(ImageFileSource("imgs/*.png"), engine="cosmos")
         .load_files()                              # read bytes as a scalable stage
         .decode("image")                           # bytes → tensors
         .quality_filter(min_words=8)               # a curation stage
         .map_batches(embed_fn, gpus=1, batch_size=64)  # your GPU model stage
         .to_relation(con))                         # → a queryable jude Relation

out.filter("score > 0.9").to_arrow()               # keep going in SQL/relational land
```

Stages available out of the box: `load_files`, `decode("image"|"audio"|"video"|"document")`,
`chunk`, `quality_filter`, `content_hash`, and `map_batches(fn)` for anything custom
(e.g. embedding or LLM calls — bring your own model).

### Watch it live

```bash
python -m jude.observe
```

---

## Build & test

Requirements: Rust (stable) + Cargo, Python 3.14 with a venv at `.venv`,
[`maturin`](https://www.maturin.rs/).

```bash
source .venv/bin/activate
maturin develop --release        # build the native extension into the venv
cargo check --lib                # fast inner-loop type check (no Python link)
python -m pytest tests/ -q -m "not slow and not benchmark"
```

We also run **Vane's own test suite, unmodified**, against jude
(`tests/vane_ported/`, the harness aliases `import duckdb` to `jude`);
architecturally-blocked cases are tracked in `known_gaps.txt` and auto-`xfail`ed,
so the suite stays green and a new failure means a real regression.

Benchmarks live in `benchmarking/` (attach to the resident `python -m
jude.observe` Ray so runs show on the dashboard).

## Docs

- [`docs/vector_retrieval_report.zh.md`](docs/vector_retrieval_report.zh.md) — vector/FTS retrieval: methods × index types × workers, recall/latency/QPS.
- [`docs/llm_data_operators.zh.md`](docs/llm_data_operators.zh.md) — how each curation operator works and what it's for.
- [`docs/distributed_and_udf_internals.zh.md`](docs/distributed_and_udf_internals.zh.md) — the Rust/Python line, `WorkerManager`, shuffle, UDF pool.
- [`docs/observability_and_ray_dashboard.zh.md`](docs/observability_and_ray_dashboard.zh.md) — metrics, audit, Ray + jude dashboards.
- [`docs/billion_scale_vector_search.zh.md`](docs/billion_scale_vector_search.zh.md) · [`docs/vector_high_recall.zh.md`](docs/vector_high_recall.zh.md) — scale + recall deep dives.
- [`docs/cosmos_pipeline.md`](docs/cosmos_pipeline.md) — multi-stage multimodal pipelines over cosmos-xenna.

## Positioning

jude is a data engine, **not** an inference server: it deliberately stops at the
LLM/embedding-inference boundary (bring your own model). Everything up to and
around inference — ingestion, SQL analytics, curation, indexing, retrieval,
distribution — is in scope. The bet: most of the work in "LLM data" is data
engineering, and that work wants an engine that fuses SQL, curation operators,
and retrieval in one distributed runtime.
