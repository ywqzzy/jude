# Multi-stage pipelines with cosmos (jude.pipeline)

For embarrassingly-parallel *multi-stage* work — load files → decode → transform
→ write, where each stage wants its own resources (CPU for I/O, GPU for a model)
— jude wraps [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) as
`jude.pipeline`. The distinguishing feature: **the source and the sink are jude
relations**, so a pipeline composes with the SQL/relation engine instead of
living beside it.

This is a usage guide. The design rationale is in
[`multimodal_design.md`](multimodal_design.md).

## When to use a pipeline vs `map_batches`

- **`map_batches`** (see [`ray_getting_started.md`](ray_getting_started.md)) — one
  transform applied over partitions. Simple, one kind of worker.
- **`jude.pipeline.RelationPipeline`** — *several* stages, each independently
  scaled and resourced (stage 1 on 8 CPUs for downloading, stage 2 on 1 GPU for a
  model). cosmos streams data between stages and scales each stage's worker pool
  to keep the slowest stage fed. This is the batch-inference shape.

## Install

```bash
pip install "jude[pipeline]"     # pulls cosmos-xenna
python -c "import jude.pipeline as p; print('cosmos:', p.is_cosmos_backed())"
```

If cosmos isn't installed, `jude.pipeline` falls back to a local, order-preserving
sequential engine with the **same API** — your pipeline code is identical, it
just doesn't scale out. `is_cosmos_backed()` tells you which you're on.

## The fluent pipeline

```python
import jude
from jude.sources import ImageFileSource

# source -> load bytes -> decode+resize (GPU-ish stage) -> per-image UDF -> relation
pipe = (
    jude.pipeline.RelationPipeline
        .from_source(ImageFileSource("/data/imgs/*.jpg"), read_bytes=False, engine="cosmos")
        .load_files(path_column="path", out_column="data", cpus=1.0)      # I/O stage
        .decode("image", size=(224, 224), cpus=1.0)                        # decode stage
        .map_batches(mean_pixel, cpus=0.5)                                 # transform stage
)

rel = pipe.to_relation(jude.connect())     # sink is a jude Relation
rel.aggregate("avg(mean) AS m").fetchall() # ...queryable with SQL
```

Each `.load_files()/.decode()/.map_batches()` adds a cosmos stage; `cpus=`/`gpus=`
set that stage's per-worker resources, and cosmos sizes each stage's pool
independently. Between stages the unit of work is an Arrow-table **shard** (the
morsel), so decoders and UDFs drop in unchanged and a 1→many stage (video frames)
just returns more rows.

Entry points: `.from_source(DataSource, read_bytes=...)`, `.from_relation(rel)`,
`.from_table(pa.Table)`. Exits: `.to_relation(con)` (queryable) or `.run()` (raw
`pa.Table`).

## Choosing the engine

```python
RelationPipeline.from_relation(rel, engine="cosmos")  # real Ray, per-stage actor pools
RelationPipeline.from_relation(rel, engine="local")   # in-process, order-preserving, same API
RelationPipeline.from_relation(rel, engine="auto")    # cosmos if installed, else local
```

`engine="local"` is the right default for tests and small data — identical Stage
API, no Ray spin-up. Use `engine="cosmos"` when the data is large or a stage needs
a GPU pool.

## Custom stages

`.decode(kind)` and `.map_batches(fn)` cover most cases. For a bespoke stage,
subclass `ArrowStage` (a cosmos `Stage` over Arrow shards) and implement
`transform(table) -> table`:

```python
from jude.pipeline import ArrowStage

class Watermark(ArrowStage):
    def __init__(self, text, **kw):
        super().__init__(**kw)          # cpus=/gpus=/batch_size= flow to cosmos
        self.text = text
    def transform(self, table):         # Arrow shard in, Arrow shard out
        ...
        return table

pipe.add_stage(Watermark("©jude", cpus=1.0))
```

For a **stateful GPU stage** (load a model once per worker), keep the model on
the instance — cosmos constructs the stage once per worker and reuses it across
shards, so `__init__` is where you load weights and `transform` runs inference.

## Execution modes

cosmos supports `BATCH` (finite input, the default here), `STREAMING`, and
`SERVING`. jude's `RelationPipeline` runs in BATCH — a relation is a finite
dataset. One honest caveat: **BATCH does not preserve row order across stages**
(the local engine does). For set-oriented relational work that's fine; if you
need order, carry an explicit sort key column and `ORDER BY` it at the end.

## Gotchas

- **Order** across cosmos stages is not preserved (above).
- **Pickling**: stage classes and UDFs are shipped to workers by value
  (cloudpickle), so a class defined in your script/notebook works — but it must be
  importable-by-value (no unpicklable captures like open file handles).
- **`is_cosmos_backed()` is False** → you're on the local fallback; `pip install
  cosmos-xenna` to scale out.
- Observing a running pipeline (per-stage throughput, worker pools) — see
  [`observability.md`](observability.md).
