# Multimodal in jude

## The problem SQL can't see

A relational engine has a closed world of types: numbers, strings, dates, the occasional list or
struct. An image is none of those. You can stuff the encoded PNG bytes into a `BLOB` column and
DuckDB will happily store and shuffle them, but it cannot *decode* them, cannot resize them, cannot
turn them into the `(H, W, C)` tensor a model wants — there is no `RESIZE(image, 224, 224)` in SQL
and there never will be. So every engine that does multimodal work faces the same fork: either teach
the SQL layer a pile of opaque scalar UDFs (slow, per-row, untyped), or grow a *second* expression
system that lives beside SQL and knows about pixels and samples and frames.

Daft took the second road, and it's the right one: `col("url").url.download().image.decode()
.image.resize(224, 224).image.to_tensor()` reads like the relational algebra it sits next to, but
each `.image.*` step is a typed, columnar, native-code kernel, not a Python function called a million
times. jude is aiming at that same surface. This document is about how we get there over stock
DuckDB without a fork, where the Rust/Python line falls (the user's standing constraint is
"maximize Rust"), and — most importantly — the one architectural seam that makes the whole thing
work: multimodal operators are a *materialization boundary*, not a SQL expression.

## What already exists, and why it isn't enough

The ingestion and decode layers are built and tested. `jude.sources` turns a glob or directory into
a relation with a `path`, a `size_bytes`, and a binary column of encoded bytes tagged with a logical
type — `ImageFileSource`, `AudioFileSource`, `VideoFrameSource`, `DocumentSource`, all four
modalities, not just images. `jude.multimodal` has batch decoders for each: image → tensor via PIL,
audio → float samples via soundfile (with resampling), video → one row per frame via PyAV, document
→ one row per page via pypdf. And `jude.pipeline.RelationPipeline` chains these as cosmos-xenna
stages with a relation as both source and sink.

What's missing is the thing that makes multimodal feel *native* rather than bolted-on: you can't yet
write the decode as part of a query. Today you reach for `map_batches` with a hand-written function,
or you build an explicit pipeline. Both work; neither is `col("img").image.decode().image
.resize(...)` composed into a `.filter()` and an `.aggregate()`. Closing that gap is this design, and
it also lets us collapse the current decoders and the pipeline's `DecodeStage` onto *one*
implementation instead of the two parallel ones we have now.

## The seam: multimodal ops are a materialization boundary

Here is the crux. A jude relation is a `LogicalPlan` tree that lowers to SQL and runs on DuckDB. A
multimodal op *cannot* lower to SQL — there is no SQL for "decode this PNG." So the moment a plan
contains an `image.decode()`, that node becomes a wall: everything *below* it is still relational and
runs on DuckDB as usual; the op itself runs as a Rust kernel over the Arrow batch that DuckDB
produces; and the *result* re-enters the plan as a materialized leaf, so everything *above* it —
a filter on the new tensor column, an aggregate over it — is once again ordinary SQL on DuckDB.

This is not a new mechanism invented for multimodal. It is exactly how `LogicalPlan::MapBatches` and
`Materialized` leaves already behave: `to_subquery_sql` has a `resolve` closure that registers an
in-memory batch as a temp table on demand, so a SQL-less node can sit in the middle of a plan and
the layers around it never know. The multimodal node, `LogicalPlan::MultimodalMap`, reuses that
machinery. `to_sql` on it deliberately returns "not lowerable," which is the signal to `materialize`
to run the kernel and stash the result. The payoff of reusing the existing boundary is that
multimodal columns compose with *everything* — joins, aggregates, window functions, the distributed
runner — for free, because above the boundary it's just a table with a tensor column, and jude
already knows how to distribute and query tables.

There is one place this seam is genuinely uncertain, and I'd rather name it than discover it in
production: the tensor column's Arrow type. `TensorType` wants to be Arrow's `fixed_shape_tensor`
extension type, but when we round-trip a batch through a DuckDB temp table to run the SQL *above* the
boundary, DuckDB may not preserve the extension metadata — it might hand back a plain
`fixed_size_list<u8>` and drop the shape. The design's answer is that `tensor.rs` owns both
encodings and can fall back to `list<u8>` + an explicit `shape` column (the variable-shape path we
need anyway), but which one survives the round-trip is the first thing P-A has to measure, because it
determines whether "query over a tensor column" is seamless or needs a reconstruction step.

## What the boundary costs

A boundary is not free, and pretending otherwise would be the naive kind of claim this doc avoids.
It has three costs, in decreasing order of how much you should care.

First, it is a **full-stop barrier — no pipelining across it**. Everything below the boundary must
finish and produce all its batches before the kernel runs, and the kernel must finish before the SQL
above it starts. This is the same Model B barrier from the distributed design; you cannot stream rows
through a decode the way a fully-pipelined engine could.

Second, a **temp-table round-trip — one copy**. The kernel's output is registered as a DuckDB temp
table and re-scanned by the layer above. That is an Arrow→DuckDB write plus a re-scan: a memory copy
whose cost scales with bytes.

Both of these round to nothing *for the workloads a boundary is for*, and that is the whole point.
You only put an op behind a boundary when the op is expensive — decoding a JPEG, resizing, running a
model — which is **milliseconds per row**. The boundary copy is **microseconds per batch**. The
ratio isn't close:

```
   time to process one batch of images:
   ├─ decode + resize (🦀 the kernel actually working)  ████████████████████  ~95%+
   └─ boundary copy (temp-table round-trip)             ▏                     a few %
```

And jude's target pipelines (scan → decode → map → sink) have **one** boundary, not a stack of them,
so the barrier doesn't compound. The boundary is also exactly where the parallel speedup comes from
(the kernel runs GIL-free / on the pool), so its "cost" is also the mechanism that replaces the slow
row-by-row-under-GIL alternative. The decision rule is therefore clean: cheap, vectorizable transform
→ in-engine, no boundary; expensive or model-backed → boundary. You only pay the boundary when the
work behind it dwarfs it, so it is never the bottleneck.

The third cost is the one that can actually make you do useless work: **DuckDB can't optimize across
the wall.** In particular, a filter above the boundary is not pushed below it. If your filter doesn't
depend on the decoded output, decoding first and filtering second means you decoded rows you then
throw away:

```
   ❌ slow — decode everything, then filter:
        Filter(category = 'cat')          ← above the wall
          └─ MultimodalMap(decode) ★      ← decoded 1,000,000 images
               └─ Scan                       ...to keep 10,000  → 99% wasted

   ✅ fast — filter first, then decode:
        MultimodalMap(decode) ★           ← decodes only 10,000
          └─ Filter(category = 'cat')     ← below the wall (plain SQL; DuckDB filters first)
               └─ Scan
```

As long as the predicate doesn't reference the decoded column (filter on `category`, not on
`img.width`), it belongs *below* the boundary — today by writing `.filter(...)` before
`.with_column(decode)`, and later by an optimizer pass that pushes boundary-independent predicates
under the wall automatically (predicate pushdown through the boundary — a named, not-yet-done
optimization).

## Where Rust ends and Python begins

The constraint is "maximize Rust," and for multimodal that has a natural, defensible boundary rather
than a dogmatic one. Decode/resize/crop/encode for images is `image`-crate territory — mature,
fast, operates on byte buffers, releases the GIL cleanly — so it's Rust, operating directly on
Arrow `BinaryArray` → tensor and back, one `Python::detach` for the whole batch. Audio decode +
resample is `symphonia`, also Rust. URL download is `std::fs` for local and `ureq` for HTTP, Rust,
with an explicit `NotImplemented` arm for `s3://`/`gs://` so the gap is loud, not silent.

Where Rust *doesn't* win, we don't pretend. Video demuxing and PDF text extraction have no
Rust library that matches PyAV and pypdf in format coverage, and shipping a worse decoder to satisfy
a language preference would be the wrong call. So `python_fallback.rs` calls the existing
`decode_video_batch` / `decode_document_batch` over the Arrow batch through PyO3. The *expression
API* is uniform — `col("v").video.decode()` looks identical whether the kernel underneath is Rust or
Python — but the implementation honors reality: Rust where a good codec exists, Python only where it
doesn't. This is the honest reading of "maximize Rust," not "rewrite libav in Rust to make a point."

## Fusion, and why the op chain is a list

A multimodal expression is a *chain*: decode, then resize, then to-tensor. If each link were its own
plan node, we'd decode to a full-resolution tensor column, materialize it, resize it into a second
column, materialize that, and so on — three passes and two throwaway columns. Instead the chain is a
single `MultimodalMap` carrying `ops: Vec<MmOp>`, and the kernel dispatcher folds the whole chain in
one pass: decode each element, apply resize and to-tensor in registers, write one output column. The
intermediate full-res tensors never become Arrow arrays. This is why the Python accessors
(`.image.decode().image.resize(...)`) accumulate an op list rather than building nested plan nodes —
the fluent surface and the fused execution are the same list, read from two ends.

The dispatcher folds `ArrayRef -> ArrayRef` per op, propagates nulls (a null input row is a null
output row, never a decode attempt on garbage), and honors a per-op `on_error` of `raise` (default,
matching Daft) or `null` (turn a corrupt image into a null instead of aborting a billion-row job —
which, at scale, is the option you actually want).

## One implementation, three doors

The same `MmOp` chain and the same `multimodal::apply_ops` dispatcher serve three entry points, and
that unification is a feature, not an accident. The **expression** path (`col("x").image.decode()`)
is the new typed surface. The **pipeline** path (`RelationPipeline.decode(kind)`) builds the same
chain, so the cosmos multi-stage pipeline and the query engine share one decoder — no drift between
"decode in a pipeline" and "decode in a query." And **`map_batches`** stays as the escape hatch for
arbitrary Python UDFs that aren't expressible as typed ops. Three doors, one room.

Distribution comes along for free precisely because of the materialization-boundary design: a
`MultimodalMap` over a partitioned relation runs its kernel on each partition's batch on the worker
that owns it, because the ops are already batch-shaped and the boundary already knows how to be a
per-partition leaf. `col("img").image.decode()` under `execution_backend="ray"` is decode-on-the-
workers with no extra code, and the result must match single-node — which the test suite checks.

## Phasing, honestly ordered

P-A is the foundation and the riskiest part, so it's first: the `MultimodalMap` node, the
materialization-boundary execution, the dispatcher, and `tensor.rs` — shipped with an identity op so
the boundary is proven end-to-end (including that DuckDB temp-table round-trip) before any real
kernel depends on it. P-B is the image kernel, the meat of the demo, and the point where
`DecodeStage` collapses onto the shared implementation. P-C adds URL download and audio. P-D wires
video and document through the Python fallback so the expression API covers all four modalities. P-E
is `embed_image` — an op that runs a batched model on a GPU stage — which ties into the separate GPU
inference track and does not pretend to exist before its model runtime does.

The measure of done at each phase is the same: Rust unit tests for the kernels (decode a synthesized
2×2 PNG and assert the pixels, resize, encode/decode round-trip, null propagation, `on_error`
behavior on garbage bytes), a Python end-to-end test that decodes real fixtures and queries the
result with SQL, a distributed-parity test, and the whole suite staying at zero failures.
