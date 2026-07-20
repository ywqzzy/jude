# UDFs in jude

## The two doors, and why one of them is a trap

jude today has two completely separate ways to run Python over data, and the honest starting point
for this document is that a user cannot tell, from the shape of their code, which one is fast. The
first door is `conn.create_function("f", py_fn, ...)`, which registers `f` as a real DuckDB scalar
function so it can be called from inside SQL: `SELECT f(x) FROM t`. The second door is
`rel.map_batches(py_fn, execution_backend="subprocess")`, which ships the function to a pool of
worker interpreters and streams Arrow batches to them off the GIL. The first door is the one that
*reads* like the natural thing to do — it's SQL, it composes, it's what DuckDB users expect — and it
is the slow one, by a wide margin, because it calls Python once per row with the GIL held the entire
time. The second door is the fast one (~6.5× on the multimodal benchmark) but it is only reachable
by leaving SQL and restructuring your pipeline around `map_batches`.

Vane does not have this trap, and understanding *why* is the spine of this design. Vane also has two
scalar registration paths, but it wired the SQL-callable one into the fast execution layer, so a UDF
called from inside a query runs on the same out-of-process, GIL-free, dynamically-batched machinery
as its batch API. jude's job is to close that gap Rust-first: make the in-SQL scalar path
vectorized instead of row-by-row, and give it a bridge to the batch pool for the heavy cases —
without building a second execution engine, because jude already has a good one.

## The native scalar path, honestly

`src/expression_udf/registration.rs` is the whole in-SQL scalar surface, and it was just improved in
two real ways worth stating precisely. A single generic adapter, `PyUdf`, now implements
duckdb-rs's `VScalar` for *any* arity: DuckDB hands `invoke` the whole input chunk, the arity is
read from `state.param_types.len()`, and one type serves 0-arg through N-arg functions
(registration.rs:171-203) — the old 1-to-4 cap is gone. And NULL handling is correct at the
boundary in both directions: a SQL NULL argument becomes Python `None`
(`extract_row_value` returns `py.None()` when `vec.row_is_null`, registration.rs:91-93), and a UDF
returning `None` becomes SQL NULL via `FlatVector::set_null` (`write_row_output`,
registration.rs:130-132). Those are genuine correctness wins.

But the core of `invoke` is the trap. It is `Python::attach(|py| { for row in 0..len { … build a
tuple … func.call1 … write one output … } })` (registration.rs:179-195). The GIL is acquired once
and *held for the entire chunk*, and inside that region we make one `call1` per row. For a
2048-row DuckDB vector that is 2048 Python calls, 2048 argument-tuple allocations, and 2048
per-value conversions, all serialized under the lock. This is the single worst property of the
system and the thing the plan below is built to fix.

The type surface is the second problem, and it is subtler because it fails *silently*.
`extract_row_value` and `write_row_output` handle exactly five logical types — Varchar, Integer,
Bigint, Double, Boolean — and everything else falls through a `_ =>` arm that treats the value as a
string (registration.rs:119-124 and 163-166). Meanwhile `type_str_to_id` happily maps `"FLOAT"`,
`"BLOB"`, `"DATE"`, and `"TIMESTAMP"` to their DuckDB type ids (registration.rs:18-31). The
consequence is a correctness bug, not merely a missing feature: register a function with
`return_type="FLOAT"` and its result is coerced through the string fallback rather than written as
an `f32`. A DATE argument arrives at the UDF as a stringified blob. Vane's native path, by
contrast, round-trips the full type matrix — TINYINT through HUGEINT, UUID, DATE/TIME/TIMESTAMP/
INTERVAL, BLOB, DECIMAL, and nested LIST/STRUCT — and its test suite asserts every one of them
(`vane/tests/fast/udf/test_scalar.py:56-81`).

Three more things the native path does not have, all of which Vane exposes as first-class knobs on
`create_function`. There is no **NULL-handling mode**: Vane distinguishes `DEFAULT` (rows with any
NULL argument are filtered out before the UDF is called and set NULL in the result, and the UDF is
forbidden from returning NULL) from `SPECIAL` (the UDF sees the NULLs and decides), a two-value enum
resolved from a string or int (`null_handling_enum.hpp:15-34`, semantics in `python_udf.cpp:147-178`
and `202-227`). There is no **exception-handling mode**: Vane offers `FORWARD_ERROR` (re-raise) vs
`RETURN_NULL` (a throwing row becomes NULL and the scan continues), a real choice for a
billion-row job where one corrupt input shouldn't abort everything
(`exception_handling_enum.hpp:13`, applied at `python_udf.cpp:237-247` and `346-363`). And there is
no **volatility / side-effects** control: Vane maps `side_effects=True` to
`FunctionStability::VOLATILE` so DuckDB won't fold repeated calls (`python_udf.cpp:552-555`), which
matters for anything nondeterministic. jude's `create_function` signature is
`(name, func, parameters=None, return_type=None, **_kwargs)` (connection.rs:422-439) — the `**_kwargs`
swallows `type=`, `null_handling=`, `exception_handling=`, and `side_effects=` and drops them on the
floor. Finally, deregistration is a no-op: DuckDB refuses to `DROP` a scalar function registered
through the C API ("internal catalog entry"), so `detach_function` swallows the error
(registration.rs:76-79), where Vane can atomically replace and remove its registrations.

## The out-of-process batch path, honestly

This is the strong side of jude, and it is strong for exactly the reason the distributed design doc
argues: the orchestration is in Rust and the GIL is released across the whole dispatch.
`src/udf/subprocess.rs` is a pool of persistent worker subprocesses, each running
`python -m jude.execution._worker`, speaking length-prefixed Arrow IPC over stdin/stdout. `map_batches`
assigns input batches to workers round-robin, runs each worker's slice on its own OS thread so the
pipes overlap (subprocess.rs:112-154), and the *caller* enters this whole region under `py.detach`
(relation.rs:353-356) so N workers are N real interpreters running in genuine parallel while other
Python threads keep going. Pools are cached by `(python, worker_count, hash(init_payload))` so the
~100ms-per-worker spawn cost is paid once per distinct UDF (subprocess.rs:243-256) — the same idea as
Vane's actor pools. Byte-aware rebatching (`rechunk_batches_bytes`, relation.rs:480-513) lets a
caller cap batches by bytes as well as rows, which is what keeps a GPU/model batch inside a memory
budget regardless of row width.

Above that sit two more backends, routed by `map_batches_py` on the `execution_backend` string
(relation.rs:1396-1423): `ray_task`/`ray_actor` go through `jude.execution.udf_ray`
(`RayTaskExecutor`, `RayActorExecutor`) which carry Arrow tables through the Ray object store and
preserve submission order, with the actor pool loading the UDF (and its model weights) once and
optionally claiming a GPU; and plain `ray` goes through the partition-level `map_relation` runner.
Four call modes exist in the worker (`map_batches`, `map_batches_rows`, `flat_map`, `map`) with a
scalar `map` that applies a function per row over the first column
(`python/jude/execution/_common.py:42-50`). The `jude.func` / `jude.cls` / `jude.cls.batch`
decorators (`python/jude/expression_udf.py`) mark a callable so the subprocess/Ray path knows to
instantiate a class once per worker for stateful actors.

What this path is missing, measured against Vane's ~11.5k-line execution layer, is everything that
turns "ship batches, concat results" into a streaming, admission-controlled, fault-aware runtime.
There is no streaming: `SubprocessPool::map_batches` materializes every input batch, runs them all,
and concatenates (subprocess.rs:112-154) — Vane produces output incrementally through a Ray
block/metadata generator protocol with real cross-process backpressure (`udf_ray_stream_protocol.py`)
and, for the local pool, a `/dev/shm` shared-memory budget manager with input leases and output
grants (`ref_bundle.py`). There is no admission control: jude's per-UDF path has no analogue of
Vane's non-blocking one-lookahead task leases acquired from a query-driver actor
(`udf_task_admission.py:61-250`). And the actor lifecycle is bare — `RayActorExecutor` builds a
fixed pool and `ray.kill`s it on shutdown (`udf_ray.py:104-110`) with none of Vane's two-phase
readiness, post-construction payload injection, deadline-bounded init, side-effect-aware retry
suppression, or actor-loss detection. These are real gaps, but they are *distributed-runtime* gaps,
and they overlap heavily with the fault-tolerance and backpressure work already scoped in the
distributed design; this document treats them as shared future work rather than re-planning them.

Two honest non-gaps, so the comparison stays fair. Async *user callables* are not supported here,
and they are deliberately not supported in Vane either — Vane removed actor async mode and runs
actor methods serially at `max_concurrency=1` because user callables aren't thread-safe. And
neither engine has a Python **aggregate** UDF (UDAF) registration path at all; Vane's only aggregate
surface is an experimental Spark `registerJavaUDAF` stub. So "no async UDFs" and "no aggregate UDFs"
are places jude is level with Vane, not behind it, and the plan does not chase them.

## The gap that actually matters: no bridge from SQL to the pool

Step back and the picture is stark. jude has a fast batch engine and a slow in-SQL scalar path, and
*nothing connects them*. If you write `SELECT classify(text) FROM docs`, `classify` runs row-by-row
under the GIL — even if `classify` is a batched model call that would be 100× faster over an Arrow
batch — because the only way to reach the pool is to abandon SQL and hand-write `map_batches`.

Vane closed exactly this. Its `create_function` native/arrow path (`python_udf.cpp`) is the same
in-engine shape as jude's, but its *other* scalar path — `vane.func` / `attach_function`, and the
SQL `CREATE FUNCTION` behind it — registers a placeholder scalar function whose bind step,
`LowerRegisteredExpressionUDF` (`python_udf_utils.cpp:282-305`), rewrites the call to carry a
pickled-UDF payload and routes execution through `CreatePythonUDFExecutor`
(`udf_executor.cpp:3718`, installed as the executor factory at `udf_executor.cpp:3777`) — i.e. the
same subprocess/Ray dispatcher as `map_batches`, chosen by an `execution_backend` field baked into
the payload (`BuildExpressionScalarUDFPayload`, `python_udf_utils.cpp:517`;
`CreateVaneFunctionInternal`, `pyconnection.cpp:1070-1122`). The upshot is that a scalar UDF *called
from inside a SQL query* in Vane executes GIL-free on the batch pool. That single wire is the most
important thing jude is missing, and it is worth being clear that jude cannot copy the mechanism
literally: Vane can intercept a bound expression tree because it forks the engine, and jude lowers
plans to SQL *strings* over stock DuckDB, so it has no bound-expression hook to hang a rewrite on.
The plan therefore reaches the same destination by a different, honest road — the
materialization-boundary seam jude already built for multimodal.

## The plan

### Phase 1 — Vectorized (arrow-native) scalar UDFs

This is the headline fix, and the pleasant surprise is that the hard part is already done for us.
duckdb-rs, at the version jude pins, exposes a `VArrowScalar` trait (`vscalar/arrow.rs:75-102`) whose
blanket `impl VScalar` (arrow.rs:104-128) does precisely the conversion we want: DuckDB hands over
the whole `DataChunkHandle`, `data_chunk_to_arrow` turns it into an Arrow `RecordBatch` in one shot,
`invoke(state, batch) -> Arc<dyn Array>` is called *once for the whole chunk*, and
`write_arrow_array_to_vector` writes the result back (arrow.rs:115-116). jude's `Cargo.toml` already
enables the `vscalar-arrow` feature. So the vectorized path is not a research problem; it is
plumbing.

The design: add a `type="arrow"` (equivalently `vectorized=True`) mode to `create_function`. Under
it, jude registers a `VArrowScalar` adapter. Its `invoke` takes the `RecordBatch`, exports it to a
pyarrow `RecordBatch` once (jude already has `arrow_ffi::batches_to_pyarrow_table` for exactly this
FFI hop), calls the user function once with the columns as arguments — matching Vane's arrow
semantics where the UDF receives `pa.ChunkedArray`s and returns an array/table
(`python_udf.cpp:180-307`, and the contract documented in `vane/duckdb/udf.py`'s `vectorized`
decorator) — takes back a pyarrow array, imports it to an Arrow `ArrayRef`, and returns it. The GIL
is acquired **once per ~2048-row vector** instead of once per row, and for the overwhelmingly common
case of a UDF whose body is itself numpy/pyarrow-vectorized, the Python-side work is C-speed columnar
math with no per-row interpreter overhead at all. This is the direct cure for the row-by-row-under-
GIL weakness, and it lives entirely in Rust.

A second, free win rides along: the arrow path inherits **full type coverage** from duckdb-rs's
Arrow↔DuckDB conversion, because `data_chunk_to_arrow` / `write_arrow_array_to_vector` already know
the whole type matrix. The moment the arrow path exists, FLOAT, BLOB, DATE, TIMESTAMP, DECIMAL,
LIST, and STRUCT UDFs work without hand-writing an arm per type in `extract_row_value`. The existing
row-by-row native path stays for scalar functions that genuinely aren't vectorizable, but its silent
string fallback must be fixed to *error* on an unsupported type rather than corrupt the data
(registration.rs:119-124, 163-166), and its five-type set should be widened to the common scalars.

### Phase 2 — NULL, error, and volatility semantics

With the arrow adapter in place, wire the knobs `create_function` currently drops. `null_handling`
becomes a two-mode enum matching Vane: `default` filters NULL-argument rows out of the batch before
the call and reinstates them as NULL after (in the arrow path this is an Arrow filter + a
scatter-back, the same shape as `python_udf.cpp:202-227`), while `special` passes NULLs through.
`exception_handling` becomes `forward` (re-raise, today's only behavior) vs `null` (a throwing chunk,
or row in the native path, becomes NULL and execution continues). `side_effects=True` sets the
duckdb-rs `ScalarFunction` stability to volatile so DuckDB won't fold calls. Varargs falls out of
`VArrowScalar`'s variadic signature support (`ArrowScalarParams::Variadic`, arrow.rs:16-20). Optional
but cheap: read the Python signature's type annotations in Rust to infer `parameters`/`return_dtype`
when omitted, mirroring Vane's `AnalyzeSignature` (`python_udf.cpp:497-525`), so users aren't forced
to pass SQL-type strings.

### Phase 3 — The bridge: in-SQL UDFs that run on the pool

This is the phase that closes the gap that matters, and it deliberately reuses the multimodal seam
rather than inventing anything. Recall from the multimodal design that a `LogicalPlan::MultimodalMap`
node is a *materialization boundary*: it can't lower to SQL, so `to_sql` returns "not lowerable,"
`materialize` runs a Rust/Python kernel over the Arrow batch DuckDB produced below it, and the result
re-enters the plan as a materialized leaf that everything above treats as an ordinary table. A
pool-backed scalar UDF is the same kind of node. When a UDF is registered with
`execution_backend="subprocess"|"ray_task"|"ray_actor"`, jude does *not* register it as a DuckDB
scalar function at all; instead the expression API exposes it as an accessor —
`col("text").udf(classify)`, the same fluent shape as `col("img").image.decode()` — that builds a
`MapBatches`/boundary node over the referenced column(s). At materialize time that node routes
through the **existing** `serialize_udf` → `SubprocessPool` / `RayTaskExecutor` / `RayActorExecutor`
machinery (relation.rs:306-442). The column is transformed off-GIL in batches and stitched back as a
new column; everything above the boundary — filters, joins, aggregates, the distributed runner —
remains ordinary SQL on stock DuckDB, exactly as multimodal columns already compose for free.

This is honest about jude's architecture rather than pretending to be Vane. jude can't intercept
`f(x)` inside an arbitrary SQL *string* the way a forked engine intercepts a bound expression, so
the pool-backed form is reached through the relational/expression API, not through hand-written SQL
text. That is a real ergonomic difference from Vane and the doc should not hide it. What it buys, at
zero new-engine cost, is the thing that actually mattered: a UDF that is expensive per call — a model
inference, a remote API, anything that wants a batch — runs GIL-free on the pool while reading like
part of the query. The decision rule for a user (and, later, for an automatic planner heuristic) is
clean: a cheap, vectorizable transform is an arrow-native scalar function (Phase 1, in-engine, no
process hop); an expensive or model-backed call is a pool-backed boundary UDF (Phase 3). The two
share the type system and the expression surface and differ only in where the work runs.

### Phase 4 — Python table functions (UDTFs)

Vane can register a Python function as a DuckDB table function via `create_table_function` /
`RegisterTableUDF` (`pyconnection.cpp:1177-1229`, `1240-1244`), backed by the same out-of-process
executor and distributable. jude's `table_function` only invokes *built-in* DuckDB table functions
by name (connection.rs:487-512) — there is no way to register a Python generator as a table source.
The Rust-first, boundary-consistent design is to make a UDTF a *leaf* rather than an in-SQL
`FROM f(x)`: `jude.table_function(gen, schema=…)` builds a materialized/boundary source node that
runs the Python generator through the pool and re-enters as a relation with the declared schema. As
with Phase 3, jude won't get true in-SQL `FROM f(args)` without engine support, but a relation-level
UDTF covers the ingest-and-expand shape (one input row fanning to many, a document to its pages) that
the target workloads actually use, and it reuses the pool and the boundary with no new executor.

### Phase 5 — Output schema, and batched-inference polish

Two smaller items. First, **output schema** is currently accepted and ignored — `map_batches_py`
has a literal `let _ = schema;` (relation.rs:1408) — so the result schema is whatever the function
happened to return. Vane declares the output schema in the payload and *enforces* it, including
tensor `fixed_shape_tensor` outputs, full nested types, and correctly-typed empty results
(`udf_output_schema.py`). jude should thread the declared schema through the boundary node so a UDF
that returns nothing still yields a correctly-typed empty relation and a mismatched return is a clear
error, not silent drift — and this is where the multimodal `TensorType` work (see the multimodal
design) and UDF outputs converge on one Arrow tensor encoding. Second, **batched inference**: jude
already has byte-based dynamic batching (`rechunk_batches_bytes`) and a GPU-capable Ray actor pool,
which is a real foundation. The polish that ties into the multimodal `embed_image` track (P-E there)
is load-aware routing across the actor pool and, for LLM inference specifically, an optional async
executor over a continuous-batching engine — the one place Vane genuinely runs concurrent in-flight
requests per actor (its vLLM family, `duckdb/execution/vllm.py`). This is explicitly the *last*
phase and gated on the model-runtime work existing, not invented ahead of it.

## What this does and doesn't duplicate

The load-bearing claim of the plan is that it adds exactly **one** new execution mechanism — the
arrow-native in-process scalar adapter in Phase 1 — and everything else is a new *front door* onto
machinery jude already has. Phase 3's pool-backed UDFs, Phase 4's UDTFs, and Phase 5's inference
polish all funnel into the same `serialize_udf` payload and the same `SubprocessPool` / Ray
executors, through the same materialization-boundary seam that multimodal uses; there is no second
scheduler and no parallel worker protocol. The arrow adapter is the only genuinely new code path,
and it earns its place because it serves the case the pool serves *badly*: a cheap per-call transform
in a tight in-engine loop, where crossing a process boundary would cost far more than the GIL
acquisition it saves. Draw the line there — vectorize in Rust when the work is small and columnar,
cross to the pool when the work is heavy — and jude matches Vane's UDF ergonomics while keeping the
Rust-first, no-fork thesis intact.

## Milestones

Done: a single generic `VScalar` adapter for any arity with correct NULL-in/NULL-out at the boundary;
the out-of-process subprocess pool with cached pools, thread-overlapped pipes, and GIL-released
dispatch; Ray task/actor backends and byte-based dynamic batching; the `func`/`cls`/`cls.batch`
decorators. Next, in priority order: Phase 1 (arrow-native scalar via `VArrowScalar`, plus fixing the
silent-stringify fallback), which is mostly plumbing over existing crate support and delivers the
biggest single win; Phase 2 (null/error/volatility knobs and signature inference); Phase 3 (the
boundary bridge from the expression API to the pool — the architecturally important one); then Phase
4 (relation-level UDTFs) and Phase 5 (schema enforcement and inference polish), the latter converging
with the multimodal and distributed tracks rather than duplicating them.
