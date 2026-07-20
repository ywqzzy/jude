#!/usr/bin/env python3
"""Multimodal batch-inference benchmark (Vane methodology).

Mirrors the shape of Vane's multimodal_inference_benchmarks/image_classification:
a stream of images is decoded and run through a (CPU-bound stand-in for a) model
that produces an embedding tensor. We measure end-to-end throughput of jude's
execution backends to demonstrate GIL-free scaling — the core of the
"faster than Vane's Python control plane" claim.

The per-image work is a deterministic CPU load (no GPU/network needed), so the
benchmark runs anywhere and isolates the *scheduling / GIL* dimension, which is
where jude's Rust orchestration + out-of-process workers win.

Run:
    python benchmarking/bench_multimodal_inference.py --images 400 --dim 64 --work 8000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa

import jude
from jude.types import tensor_array, tensor_to_numpy


def make_images(n: int, size: int = 64) -> pa.Table:
    """A table of fake encoded image blobs."""
    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, size, dtype="uint8").tobytes() for _ in range(n)]
    return pa.table({"id": list(range(n)), "img": pa.array(blobs, type=pa.binary())})


# Module-level so it is picklable by value for the worker backends.
class Infer:
    """Decode + 'model inference': per image, do CPU work, emit a `dim` tensor.

    A picklable callable (holds dim/work as attributes) so it ships to
    subprocess / Ray workers cleanly.
    """

    def __init__(self, dim: int = 64, work: int = 8000):
        self.dim = dim
        self.work = work

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import numpy as np

        from jude.types import tensor_array

        out = []
        for blob in batch["img"].to_pylist():
            arr = np.frombuffer(blob, dtype="uint8").astype("float32")
            acc = 0.0
            for i in range(self.work):
                acc += (arr[i % arr.size] * i) % 7.0
            vec = np.full(self.dim, acc % 1000.0, dtype="float32")
            out.append(vec)
        embs = np.stack(out)
        return batch.append_column("emb", tensor_array(embs, dtype="float32", shape=[self.dim]))


def _bind(dim: int, work: int):
    return Infer(dim, work)


def run_backend(rel, fn, backend: str | None, batch_size: int, num_workers: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    if backend is None:
        out = rel.map_batches(fn, batch_size=batch_size)
    else:
        out = rel.map_batches(fn, batch_size=batch_size, execution_backend=backend, num_workers=num_workers)
    rows = out.num_rows
    elapsed = time.perf_counter() - t0
    return elapsed, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=400)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--work", type=int, default=8000, help="CPU work iterations per image")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=64)
    args = ap.parse_args()

    con = jude.connect()
    con.register("images", make_images(args.images, args.image_size))
    fn = _bind(args.dim, args.work)

    print(f"\nMultimodal inference benchmark: {args.images} images, dim={args.dim}, "
          f"work={args.work}/img, batch_size={args.batch_size}, workers={args.workers}")
    print("-" * 68)

    results = {}

    # 1) in-process (GIL-bound) baseline
    rel = con.sql("SELECT * FROM images")
    t, rows = run_backend(rel, fn, None, args.batch_size, args.workers)
    results["in-process"] = t
    print(f"{'in-process (GIL-bound)':28} {t*1000:8.1f} ms   {rows/t:8.0f} img/s")

    # 2) subprocess pool (Rust, GIL-free) — warm the pool first
    try:
        _ = con.sql("SELECT * FROM images LIMIT 1").map_batches(
            fn, execution_backend="subprocess", num_workers=args.workers
        ).num_rows
        rel = con.sql("SELECT * FROM images")
        t, rows = run_backend(rel, fn, "subprocess", args.batch_size, args.workers)
        results["subprocess"] = t
        print(f"{'subprocess pool (GIL-free)':28} {t*1000:8.1f} ms   {rows/t:8.0f} img/s")
    except Exception as e:  # pragma: no cover
        print(f"subprocess pool: skipped ({type(e).__name__}: {e})")

    # 3) ray_actor (distributed, GIL-free) if Ray available
    try:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        # Cap actors to available CPUs so the pool can actually be scheduled.
        ray_workers = min(args.workers, max(1, int(ray.cluster_resources().get("CPU", 1))))
        # warm
        _ = con.sql("SELECT * FROM images LIMIT 1").map_batches(
            fn, execution_backend="ray_actor", num_workers=ray_workers
        ).num_rows
        rel = con.sql("SELECT * FROM images")
        t, rows = run_backend(rel, fn, "ray_actor", args.batch_size, ray_workers)
        results["ray_actor"] = t
        print(f"{'ray_actor (' + str(ray_workers) + ' actors)':28} {t*1000:8.1f} ms   {rows/t:8.0f} img/s")
    except Exception as e:  # pragma: no cover
        print(f"ray_actor: skipped ({type(e).__name__}: {e})")

    print("-" * 68)
    base = results.get("in-process")
    if base:
        for name in ("subprocess", "ray_actor"):
            if name in results:
                print(f"  speedup {name:12} vs in-process: {base/results[name]:5.2f}x")

    try:
        jude.shutdown_udf_pools()
    except Exception:
        pass


if __name__ == "__main__":
    main()
