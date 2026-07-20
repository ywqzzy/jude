# Observing jude jobs on Ray

When a jude job runs distributed — `map_batches(execution_backend="ray")`,
`distributed_aggregate`/`distributed_join`, or a `jude.pipeline` cosmos pipeline
— you'll want to see what's happening: which workers exist, how far along it is,
where the time and memory go. This guide covers the three layers you can observe,
from "always available" to "cosmos-specific".

## Layer 1 — the Ray Dashboard (works for everything)

Any jude job on Ray is a Ray application, so the **Ray Dashboard** sees it. By
default jude starts Ray with `log_to_driver=False` (so worker logs don't flood
your process), but the dashboard still comes up.

```python
import ray
ray.init()                       # or let jude auto-init on first use
print(ray.get_dashboard_url())   # e.g. http://127.0.0.1:8265
```

Open that URL for: live actors and tasks, per-actor CPU/GPU/memory, the object
store (how much Arrow data is in flight — jude passes partitions as Arrow through
the object store), logs, and a timeline/flamegraph. This is the first place to
look for "why is it slow / stuck / OOMing". `ray memory` on the CLI dumps
object-store references if you suspect a partition isn't being freed.

To also stream worker `print`/logs back to your driver, opt in:

```python
ray.init(log_to_driver=True)     # before jude touches Ray
```

## Layer 2 — jude's own runner state (what jude decided)

jude exposes the scheduling decisions the Rust `WorkerManager` made, which the
dashboard can't show you because they're jude-level, not Ray-level:

```python
import jude
r = jude.runners.get_or_create_runner()
print(type(r).__name__, r.num_workers)   # RayRunner, worker count
print(r.mgr)                             # the Rust jude.dist.WorkerManager
print(r.mgr.target_partitions(nbytes, num_rows))  # how many partitions it would cut
print(r.mgr.dispatch_window(n_tasks))             # in-flight backpressure window
```

For a `map_batches`/aggregate, the number of partitions and the in-flight window
are the two knobs that determine parallelism and memory; if the job under-uses the
cluster, check `target_partitions` and `JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM`; if
it OOMs the object store, cap `JUDE_RAY_MAX_TASK_BACKLOG` (fewer partitions in
flight). See [`ray_getting_started.md`](ray_getting_started.md) §5.

## Layer 3 — cosmos pipeline monitoring (per-stage)

A `jude.pipeline` cosmos pipeline is a *multi-stage* job, and cosmos-xenna prints
per-stage progress and worker-pool allocation on its own schedule. It's off-ish by
default in jude; turn it up through the cosmos `PipelineConfig` knobs:

| `PipelineConfig` field | What it shows |
|---|---|
| `monitoring_verbosity_level` | per-stage progress / throughput. `VerbosityLevel.INFO` or `DEBUG` |
| `actor_pool_verbosity_level` | how many workers each stage has, autoscaling decisions |
| `logging_interval_s` | how often the above is logged |
| `log_worker_allocation_layout` | one-shot dump of which worker sits on which node/GPU |

```python
from cosmos_xenna.pipelines import v1

cfg = v1.PipelineConfig(
    execution_mode=v1.ExecutionMode.BATCH,
    return_last_stage_outputs=True,
    monitoring_verbosity_level=v1.VerbosityLevel.INFO,
    actor_pool_verbosity_level=v1.VerbosityLevel.INFO,
    logging_interval_s=5.0,
)
# pass it to the pipeline: RelationPipeline.from_source(src, engine="cosmos",
#   pipeline_config=cfg)  — a supplied config overrides jude's default BATCH one.
```

What you get: a periodic line per stage — rows in/out, throughput, queue depth,
and current worker count — which is exactly how you find the **bottleneck stage**
(the one whose queue backs up and whose pool cosmos keeps growing). The fix is
usually to give that stage more `cpus`/`gpus` or a bigger `batch_size`.

Under the hood cosmos scales each stage's pool toward keeping the slowest stage
saturated; `actor_pool_verbosity_level=DEBUG` shows those scale-up/scale-down
decisions if throughput isn't what you expect.

## A practical checklist

1. **Is it even distributed?** `type(get_or_create_runner()).__name__` must be
   `RayRunner` (not `LocalRunner`), and `is_cosmos_backed()` must be True for a
   pipeline. If not, you're running local — nothing to observe on Ray.
2. **Is it using the cluster?** Dashboard → Actors: you should see `num_workers`
   busy. If idle, partition count is too low (Layer 2).
3. **Is it stuck / slow?** Dashboard → Tasks/Timeline for stragglers; for a
   pipeline, Layer 3's per-stage throughput to find the bottleneck stage.
4. **Is it OOMing?** Dashboard → object store; lower `JUDE_RAY_MAX_TASK_BACKLOG`
   (fewer in-flight partitions) or shrink `batch_size` on the heavy stage.

## Honest limits

jude does not (yet) emit its own metrics stream — there is no Prometheus endpoint
or built-in per-partition timing from the Rust side; observability today is
"whatever Ray + cosmos surface, plus the WorkerManager state above". A jude-native
metrics/tracing layer (per-partition timings, shuffle bytes) is future work and
overlaps with the fault-tolerance/backpressure track in the distributed design.
