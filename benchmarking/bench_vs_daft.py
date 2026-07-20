#!/usr/bin/env python3
"""Head-to-head: jude vs Daft on a CPU-bound multimodal batch-inference workload.

Vane isn't runnable in this environment (it's a DuckDB C++ fork needing Python
<3.13 + a cmake build of an un-checked-out submodule), so a direct jude-vs-Vane
run isn't possible here. Daft is a real, installed distributed engine that
Vane's own benchmarks compare against — so this is a genuine head-to-head on the
same workload shape: decode an image blob and run a CPU-bound "model" per image,
producing an embedding, over N images.

Both engines get the identical per-image work; we measure end-to-end throughput.

    python benchmarking/bench_vs_daft.py --images 1200 --work 60000 --workers 12
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa


def make_images(n: int, size: int = 64) -> pa.Table:
    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, size, dtype="uint8").tobytes() for _ in range(n)]
    return pa.table({"id": list(range(n)), "img": pa.array(blobs, type=pa.binary())})


def _work_one(blob: bytes, dim: int, work: int) -> float:
    arr = np.frombuffer(blob, dtype="uint8").astype("float32")
    acc = 0.0
    for i in range(work):
        acc += (arr[i % arr.size] * i) % 7.0
    return acc % 1000.0


# ---- jude workload (module-level class = picklable to subprocess workers) ----
class JudeInfer:
    def __init__(self, dim: int, work: int):
        self.dim = dim
        self.work = work

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import numpy as np

        vals = [_work_one(b, self.dim, self.work) for b in batch["img"].to_pylist()]
        embs = np.stack([np.full(self.dim, v, dtype="float32") for v in vals])
        import pyarrow as pa

        return batch.append_column("emb0", pa.array(embs[:, 0].tolist(), type=pa.float32()))


def run_jude(images: pa.Table, dim: int, work: int, batch_size: int, workers: int) -> tuple[float, int]:
    import jude

    con = jude.connect()
    con.register("images", images)
    fn = JudeInfer(dim, work)
    # warm the pool
    con.sql("SELECT * FROM images LIMIT 1").map_batches(fn, execution_backend="subprocess", num_workers=workers).num_rows
    rel = con.sql("SELECT * FROM images")
    t0 = time.perf_counter()
    out = rel.map_batches(fn, batch_size=batch_size, execution_backend="subprocess", num_workers=workers)
    rows = out.num_rows
    t = time.perf_counter() - t0
    try:
        jude.shutdown_udf_pools()
    except Exception:
        pass
    return t, rows


def run_daft(images: pa.Table, dim: int, work: int, batch_size: int, workers: int, concurrency: int | None = None) -> tuple[float, int]:
    import daft

    _work = _work_one

    kw = dict(return_dtype=daft.DataType.float32(), batch_size=batch_size, num_cpus=1)
    if concurrency:
        kw["concurrency"] = concurrency  # N parallel UDF instances (processes)

    @daft.udf(**kw)
    def infer(img):  # batched: img is a daft Series
        blobs = img.to_pylist()
        return [_work(b, dim, work) for b in blobs]

    df = daft.from_arrow(images)
    # warm
    daft.from_arrow(images.slice(0, 1)).with_column("emb0", infer(daft.col("img"))).to_arrow()
    t0 = time.perf_counter()
    out = df.with_column("emb0", infer(daft.col("img"))).to_arrow()
    rows = out.num_rows
    t = time.perf_counter() - t0
    return t, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=1200)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--work", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    images = make_images(args.images)
    print(f"\njude vs Daft — {args.images} images, work={args.work}/img, "
          f"batch_size={args.batch_size}, workers={args.workers}")
    print("-" * 60)
    results = {}
    backends = [
        ("jude (subprocess pool)", lambda im: run_jude(im, args.dim, args.work, args.batch_size, args.workers)),
        ("daft (default runner)", lambda im: run_daft(im, args.dim, args.work, args.batch_size, args.workers)),
        ("daft (concurrency=workers)", lambda im: run_daft(im, args.dim, args.work, args.batch_size, args.workers, concurrency=args.workers)),
    ]
    for name, fn in backends:
        try:
            t, rows = fn(images)
            results[name] = t
            print(f"  {name:30} {t*1000:8.1f} ms   {rows/t:8.0f} img/s")
        except Exception as e:  # pragma: no cover
            print(f"  {name:30} skipped ({type(e).__name__}: {e})")
    print("-" * 60)
    j = results.get("jude (subprocess pool)")
    if j:
        for name in ("daft (default runner)", "daft (concurrency=workers)"):
            if name in results:
                print(f"  jude vs {name:26}: {results[name]/j:5.2f}x faster")


if __name__ == "__main__":
    main()
