# Distributed execution in jude

## The bet

jude exists because of one wager: that Vane's distributed layer is slow *for the wrong reason*.
Vane forks DuckDB and then wraps a ~28,700-line Python control plane around it — a driver, a
scheduler, a split assigner, a resource manager, a fault-tracker — and every one of those
components runs in the interpreter, under the GIL, on the hot path of every task dispatch. When you
are scheduling tens of thousands of splits across a cluster, the scheduler *is* the workload, and a
GIL-bound scheduler serializes precisely the thing you wanted to parallelize.

So jude does not fork DuckDB, and it does not schedule in Python. It orchestrates at the **partition
level** over stock DuckDB — the same shape as Daft and Ray Data — and it puts the orchestration in
Rust. Python survives only as the Ray RPC boundary: it holds ObjectRefs, it calls `.remote()`, it
blocks in `ray.get`. It does not decide anything. That division is the whole design, and most of
this document is about defending the line between "deciding" and "forwarding," because that line is
where the performance claim lives and also where it is easiest to accidentally cheat.

## Why partition-level, and what it costs us

Forking DuckDB and teaching its optimizer to emit distributed plans (Vane's approach, and Presto's,
and Spark's) buys you pipelined, operator-fused distributed execution: a hash join can stream build
and probe sides across the network without materializing. It costs you an engine fork you now
maintain forever, and it drags the scheduler *into* the engine, which is C++ calling into Python for
UDFs — the tax jude is trying to avoid.

Partition-level orchestration (jude's choice) treats DuckDB as a black-box single-node executor and
only ever hands it a *partition* of Arrow data plus a SQL string. The distributed logic lives
entirely outside the engine: split the input, run the same local query on each shard, shuffle when
an operator needs a global view, merge. The price, stated against a *hypothetical fully-pipelined
engine* (streaming Presto/Spark), is that we materialize at shuffle boundaries — an operator such a
system would fuse and stream becomes two stages with an exchange in between.

It is worth being precise here, because it is the project's central comparison and easy to get
wrong: **this is not a deficit relative to Vane.** Vane runs a *fault-tolerant execution* (FTE)
engine, and FTE by construction materializes exchanges — `vane/duckdb/runners/fte/fte_exchange.py`
spools each exchange partition to a filesystem path per attempt, so a lost task can be retried by
re-reading the spill. Vane does not pipeline across shuffle boundaries either. So on the
materialize-at-shuffle axis jude and Vane are the same shape, and on the two axes that differ, jude
is ahead on speed: jude moves shuffle partitions through the **Ray object store (memory)** where
Vane spools to **disk**, and jude's scheduling around the shuffle is **Rust and GIL-free** where
Vane's is Python — and shuffle-heavy means many tasks, which is exactly where a GIL-bound scheduler
hurts. The one thing Vane's disk spooling buys that jude does not yet have is shuffle *fault
tolerance*: jude currently keeps exchange data in the object store with no retry, trading
fault tolerance for happy-path speed. That is a feature gap (see "not doing yet"), not a slower
shuffle, and the `FteTaskAttemptId` vocabulary exists to close it with optional spooling later.

Where partition-level execution would genuinely lose is a query with a deep join tree and small
per-partition work — a TPC-DS-shaped analytical query — because there the per-stage materialization
and scheduling overhead dominates the actual compute. jude's target workloads are the opposite
(scan → decode → map → sink over multimodal data, plus decomposable aggregations and hash joins):
few shuffle boundaries, enormous per-partition work, where the materialization cost rounds to zero.
We are explicitly not optimizing for the deep-join-tree case yet.

### The shuffle boundary, drawn

Take a two-stage query: scan + partial-aggregate, then **shuffle by group key**, then
final-aggregate. The whole question is what happens *at the shuffle*.

```
MODEL A — PIPELINED (streaming Presto/Spark; NOT fault-tolerant)
Stage-1 tasks stream rows straight into Stage-2. Stage 2 runs before Stage 1 finishes.

  S1-p0 ─rows┐
  S1-p1 ─rows┼──▶ (network, live) ──▶ S2-p0, S2-p1   ← already running
  S1-p2 ─rows┘
        no barrier · nothing written · fastest · one dead task = restart everything

MODEL B — MATERIALIZED EXCHANGE   ← both Vane AND jude
Every Stage-1 task finishes and WRITES its output; only then does Stage 2 read.

  S1-p0 ─▶[write]┐
  S1-p1 ─▶[write]┤   ══ barrier ══▶   S2-p0 ◀─[read]
  S1-p2 ─▶[write]┘                    S2-p1 ◀─[read]
        materialize, THEN next stage · slower than A · a dead task just re-reads/re-runs
```

jude did not choose the slow side relative to Vane — both are Model B. The only difference is
*where the write lands*:

```
  Vane (FTE):  S1 ─▶ 💾 DISK file        ─▶ S2 reads from disk
                     durable → retry a lost task, but pays disk I/O
  jude:        S1 ─▶ 🧠 Ray object store ─▶ S2 reads from memory
                     faster (no disk); no retry yet → no fault tolerance
```

So on this query: pipelined engine = no barrier (fastest, fragile); Vane = barrier + disk;
jude = barrier + memory (faster than Vane on the happy path, not yet fault-tolerant). The
deep-join-tree caveat is simply about how many barriers stack up — jude's target pipelines have ~0,
so the barrier cost is nil; a TPC-DS query has many, and there Model B (Vane included) loses to a
pipelined engine.

## The line: deciding vs. forwarding

Everything that constitutes a *scheduling decision* is in Rust, in `src/dist/`. Everything that
touches a Ray handle is in Python, in `python/jude/runners/_ray_shim.py`. The test I apply to every
line of code is: *if I deleted Ray and swapped in a different execution substrate (a local thread
pool, a different actor framework), would this line survive?* If yes, it is a decision and belongs
in Rust. If it would be rewritten, it is RPC glue and belongs in the shim.

Concretely, the decisions are: how many partitions to cut the input into, where the row boundaries
of those partitions fall, which worker runs partition *i*, how many tasks may be in flight at once,
and — for a join — how many hash buckets to shuffle into and which worker owns each. The forwarding
is: initializing Ray, constructing actors, calling `.remote()`, and running the `ray.wait` loop that
collects results. The shim contains no arithmetic on data sizes, no configuration constants, no
policy branches — this is enforced by a grep in CI, and it is not a joke: the single most likely way
to erode the performance thesis is for a "small" sizing tweak to land in Python because that is where
the ObjectRef happens to be. When you need such a tweak, it goes in `WorkerManager` and the shim is
told the answer.

There is exactly one place this line bends, and it is worth being honest about: the bounded-dispatch
loop that keeps *N* tasks in flight — priming a window, then `ray.wait(num_returns=1)`, popping the
finished ref, submitting the next — physically must run in Python, because a Ray ObjectRef cannot
cross into Rust. We considered a design where Rust drives the loop and calls back into Python per
completion; we rejected it, because it trades one GIL acquisition per *batch of results* for one per
*completion*, which is strictly worse, for no benefit. So the loop is Python — but the *window size*
it loops to is computed in Rust (`WorkerManager::dispatch_window`), and the loop itself has no idea
why the window is what it is. The policy is in Rust; only the mechanism is in Python.

## Anatomy of the Rust side

The Rust orchestration is built bottom-up from three layers that already exist and compile, plus a
planner that is next.

**The vocabulary** (`src/dist/fte.rs`) is the noun set every distributed engine needs, ported from
Vane's `fte_types.py`. A `FteTaskId` is the logical identity of a unit of work — `(query,
fragment-execution, partition)` — and a `FteTaskAttemptId` wraps it with an attempt number so that
retries and speculative copies are addressable. An `FteSplit` is one indivisible piece of input:
either a scan split (a range of a file, some parquet paths) or an exchange split (one upstream
shuffle partition's output). It carries `size_bytes` so the assigner can pack by bytes rather than
by count, and `addresses` for locality — carried but not yet consulted, which I flag here rather
than pretend it's wired. The `TaskDescriptor` is the mutable, growing record for one partition:
splits arrive incrementally as upstream fragments produce output, so `append_splits` dedups by
sequence id and bumps a `descriptor_version` on every real change. That version is not decoration —
it is what lets a running worker be sent a *delta* ("here are three more splits, version 7") instead
of a full re-send, and what lets `seal_source` declare an input exhausted so the worker knows it can
finalize. This is the machinery of incremental, fault-tolerant scheduling; most of it is latent
today because the reachable operators don't yet stream splits, but it is the right vocabulary and it
is already in Rust rather than waiting to be ported under time pressure later.

**The assigners** (`src/dist/split_assigner.rs`) turn a stream of splits into partitions. Three
strategies, one trait. `SingleSplitAssigner` funnels everything to partition 0 — the degenerate
case, but a real one for global aggregates. `HashSplitAssigner` routes by `source_partition_id %
n`, which is how a hash shuffle co-locates matching keys. The interesting one is
`ArbitrarySplitAssigner`, which does size-based bin-packing with *adaptive growth*: the first
partitions are deliberately small so the first results come back fast (latency), and the target
partition size grows geometrically — ×1.26 every 64 packed splits, up to a cap — so that later
partitions are large and throughput-efficient. This is a direct port of Vane's tuned heuristic
(64 MiB standard split, 2048-split ceiling per task), and getting it *bit-identical* matters,
because it is the difference between "jude schedules like Vane but in Rust" and "jude schedules
differently and now every performance comparison is confounded." It also handles broadcast
(replicated) sources — a split marked replicated is fanned out to every partition, and late-created
partitions retroactively receive the replicated splits seen so far, which is the fiddly correctness
detail that a naive reimplementation gets wrong.

**The brain** (`src/dist/worker_manager.rs`, exposed as `jude.dist.WorkerManager`) is the pyclass
the runner actually calls. It holds the config — worker count, size-grouping toggle, backlog limit,
open-cost byte target, minimum partition floor — read once from the environment at construction, and
it answers the five scheduling questions. `target_partitions(nbytes, num_rows)` is a line-for-line
port of the Spark-style sizing DuckDB-Python inherited: floor at `min_partition_num or num_workers`
so no worker sits idle, and with size-grouping on, also demand at least `ceil(nbytes /
open_cost_bytes)` tasks so no single task is absurdly large. `partition_plan` returns the actual
`(start, len)` row slices — the manager decides *where* to cut, Python only calls `table.slice`.
`dispatch_window` returns the in-flight bound. And `shuffle_bucket_count` / `shuffle_bucket_workers`
answer the join questions, the latter by actually running the ported `HashSplitAssigner` to derive
the canonical bucket set and then round-robining buckets onto workers — so even the join's routing
decision goes *through* the Rust assigner rather than being recomputed by hand. Nine unit tests pin
these against the exact values the old Python produced, which is what lets me claim the rewrite is
behavior-preserving rather than merely plausible.

**The planner** (`src/dist/stage.rs`, next) is the piece that generalizes what today is a
single-level `match` buried in `Relation::plan_json`. A `LogicalPlan` is a tree; a distributed plan
is that tree cut into stages at the operators that need a global view — aggregate, join, distinct,
sort, set-ops, explicit repartition. The planner walks the tree, and at each such boundary it emits
a stage carrying the SQL for its local (non-shuffle) work, the partition keys it shuffles on, and
its upstream dependencies. This is the artifact a general N-stage streaming executor consumes; I am
shipping the *plan* and being explicit that the executor over arbitrary stage DAGs is future work,
because the two operators that are actually reachable today — aggregate and join — are handled by
purpose-built two-phase code (below) and pretending otherwise would be the kind of naive
over-claim this design is trying not to make.

## How a query actually runs

Take the common case first: a partitioned scan or a distributed `map_batches`. The relation
materializes to one Arrow table on the driver; `WorkerManager.partition_plan` slices it into shards;
each shard is submitted to `worker_for(i)` as a `.remote()` call whose result is an ObjectRef; the
shim's `run_bounded` collects those refs to the window the manager chose, in submission order, and
hands back the tables. The UDF, if any, was cloudpickled by value on the driver and unpickled inside
the actor, where it runs against that actor's own stock-DuckDB connection. Nothing about this path
computed a schedule in Python.

The two-phase aggregate is where the partition-level model earns its keep. `_agg.build_two_phase`
(pure SQL string manipulation — it decides nothing about placement, so it stays in Python) rewrites
`GROUP BY … COUNT/SUM/MIN/MAX/AVG` into a *partial* query and a *final* merge query. The partial runs
on every partition in parallel (dispatched exactly as above); the driver concatenates the partials —
and here is a real bug that cost real time: `concat_tables` can leave the Arrow buffers unaligned,
and the arrow-rs C-stream importer *panics* on unaligned buffers, so the concat is followed by
`combine_chunks()` before anything crosses back into Rust — then runs the final merge on a local
DuckDB. `COUNT` becomes `SUM`-of-counts, `AVG` decomposes into `SUM/COUNT` and recombines, and the
result is bit-identical to the single-node answer, which the tests assert directly rather than
approximately.

The hash join is the one place jude does a genuine shuffle. Both sides are bucketed by `hash(keys) %
b` — done *in SQL* on DuckDB, because hashing is execution, not scheduling — where `b =
WorkerManager.shuffle_bucket_count`. Matching keys land in the same bucket on both sides, so bucket
*i* of the left and bucket *i* of the right can be joined locally on one actor, chosen by
`shuffle_bucket_workers[i]`. The join projection is `lhs.*, rhs.* EXCLUDE(keys)` so the shared key
columns don't duplicate. The empty-result case has a subtlety the code handles: if every bucket is
empty you've lost the output schema, so it re-runs bucket 0 purely to recover column types.

## What I'm deliberately not doing yet

No fault tolerance: the attempt-id vocabulary is in place but there is no retry policy, no
speculative execution, no lost-partition recovery. No locality-aware placement: `FteSplit.addresses`
is carried and ignored; `worker_for` is blind round-robin. No general stage executor: only aggregate
and join have distributed implementations, both via the two-phase path, and an arbitrary N-stage DAG
is planned-but-not-run. These are named as gaps rather than smoothed over because the honest state of
the engine is "the orchestration brain is real and in Rust, the reachable operators work and are
bit-exact, and the general streaming runtime is the next milestone" — and a design doc that implied
otherwise would be exactly the naive kind that's useless to the next person who reads it.

## Milestones

Done: the FTE vocabulary and the three split assigners; the `WorkerManager` brain exposed as
`jude.dist` with nine passing unit tests; the thin Ray shim with the `_JudeWorker` actor moved into
it; the `RayRunner` rewired so every decision is delegated to Rust and the existing Ray/scheduling
tests still pass unchanged. Next: `StagePlanner` for recursive shuffle-boundary stages. After that,
in rough priority: a streaming executor over arbitrary stage DAGs, then locality-aware assignment,
then retry/fault-tolerance built on the attempt-id vocabulary.
