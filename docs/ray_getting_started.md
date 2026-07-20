# Getting started with Ray in jude

jude runs single-node by default. To scale a query or a `map_batches` UDF across
cores or machines, it uses **Ray** — but you rarely touch Ray directly. This is a
beginner's guide: what to install, how to turn it on, what actually happens, and
how to tell it's working.

## What Ray does for jude (in one paragraph)

jude does not fork DuckDB and it does not schedule in Python. It splits a
relation into **partitions**, runs the *same* local DuckDB query on each partition
on a Ray worker, and collects the results. All the scheduling decisions
(how many partitions, which worker, how many in flight) are made in Rust
(`jude.dist.WorkerManager`); Ray is only the transport that ships Arrow tables to
workers and back. So "using Ray" mostly means: install it, and pick the `ray`
execution backend.

## 1. Install

```bash
pip install "jude[ray]"      # or: pip install ray
```

Ray is optional. If it isn't installed, jude silently runs everything locally —
your code doesn't change, it just doesn't distribute.

## 2. Turn it on

There are two ways; pick one.

**Per call (explicit):** pass the backend to `map_batches`.

```python
import jude

con = jude.connect()
rel = con.sql("SELECT * FROM range(1_000_000) t(n)")

def add_sq(batch):           # batch in, batch out (pyarrow.Table)
    import pyarrow as pa
    return batch.append_column("sq", pa.array([n * n for n in batch["n"].to_pylist()]))

out = rel.map_batches(add_sq, execution_backend="ray")
print(out.fetchall()[:3])
```

**For the whole session (default runner):** set the env var and let jude pick Ray.

```bash
export JUDE_RUNNER=ray       # VANE_RUNNER also honored
```

```python
import jude
# distributed aggregate / join / map now go through Ray automatically
```

The first Ray call starts a local Ray instance automatically
(`ray.init(...)`) if one isn't already running — you don't have to call
`ray.init` yourself.

## 3. What runs where

```
   your driver process                         Ray workers (N actors)
   ─────────────────────                       ──────────────────────
   rel.map_batches(fn, "ray")
     │  WorkerManager decides: N partitions,          each holds its own
     │  which worker runs each, in-flight window      stock-DuckDB connection
     ▼                                                + your pickled fn
   partition 0 ─Arrow──▶ worker 0  ─▶ fn(batch) ─▶ Arrow ─┐
   partition 1 ─Arrow──▶ worker 1  ─▶ fn(batch) ─▶ Arrow ─┼─▶ collected in order
   partition 2 ─Arrow──▶ worker 2  ─▶ fn(batch) ─▶ Arrow ─┘
```

Your function is shipped to the workers with `cloudpickle` (by value, so a
function defined in your script or a notebook works), unpickled once per worker,
and run against Arrow batches. Arrow tables travel through the Ray object store,
so there's no serialization tax beyond the Arrow buffers themselves.

## 4. Distributed SQL, not just UDFs

Two relational operations distribute out of the box when the runner is Ray:

```python
runner = jude.runners.get_or_create_runner()   # the Ray runner

# two-phase aggregate: partial per partition, merge on the driver — bit-exact
from jude.runners._agg import build_two_phase
partial_sql, final_sql = build_two_phase(group_by=["g"], aggs=["SUM(v)", "COUNT(*)"])
result = runner.distributed_aggregate(rel, partial_sql, final_sql)

# hash-shuffle join: both sides bucketed by key, each bucket joined on a worker
joined = runner.distributed_join(left, right, keys=["id"], how="inner")
```

## 5. Tuning (all optional, all read once into the Rust WorkerManager)

| Env var (`JUDE_` or `VANE_` prefix) | Meaning | Default |
|---|---|---|
| `JUDE_RAY_SCAN_TASK_SIZE_GROUPING` | size-based partition sizing on/off | `true` |
| `JUDE_RAY_SCAN_TASK_OPEN_COST_BYTES` | target bytes per partition | 4 MiB |
| `JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM` | floor on partition count | worker count |
| `JUDE_RAY_MAX_TASK_BACKLOG` | max tasks in flight (`0` = unbounded) — backpressure | `0` |

Use `repartition(n)` on a relation to pin the partition count explicitly instead
of letting size-grouping decide.

```python
rel.repartition(8).map_batches(fn, execution_backend="ray")
```

## 6. GPUs (for model UDFs)

For a stateful, GPU-pinned worker pool (load a model once per worker), use the
Ray actor backend:

```python
rel.map_batches(fn, execution_backend="ray_actor")   # actor pool, GPU-capable
```

Each actor pins its `CUDA_VISIBLE_DEVICES`, loads the model once (put weights in a
class and use `jude.cls`), and processes batches — the right shape for batched
inference.

## 7. Is it actually working?

```python
import jude
r = jude.runners.get_or_create_runner()
print(type(r).__name__)          # "RayRunner" (not "LocalRunner")
print(r.num_workers)             # how many worker actors
print(r.mgr)                     # the Rust jude.dist.WorkerManager making the decisions
```

If `type(r).__name__` is `LocalRunner`, Ray isn't installed or failed to import,
and jude fell back to local — check `pip show ray`.

## Where to go deeper

The architecture — why partition-level, where the Rust/Python line falls, how the
shuffle works, and how this compares to Vane — is in
[`distributed_design.md`](distributed_design.md) (中文：
[`distributed_design.zh.md`](distributed_design.zh.md)).
