#!/usr/bin/env python3
"""UDF execution-backend matrix micro-benchmark.

Mirrors Vane's benchmarking/bench_udf_subprocess_pool.py + bench_inflight.py:
a small, fast, CI-able matrix that isolates the *scheduling / GIL* axis of
`map_batches` across:

    backend  ∈ {in_process, subprocess, ray_task, ray_actor}
    workload ∈ {cpu (GIL-bound compute), sleep (latency-bound), arrow (near-noop)}
    workers  ∈ {1, 2, 4, 8}

and reports rows/s. This is the axis where jude's Rust-orchestrated,
out-of-process pools beat a single-process Python engine: the `cpu` and `sleep`
rows should scale with workers for the out-of-process backends and stay flat
(GIL-serialized) for in_process.

    python benchmarking/bench_udf_backends.py --rows 4096 --batch 256 --cpu-iters 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa

import jude


# ---- workloads (module-level classes = picklable by value to workers) --------
class CpuUDF:
    """GIL-bound pure-Python compute per row (the classic GIL stress)."""

    def __init__(self, iters: int):
        self.iters = iters

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        out = []
        for v in batch.column(0).to_pylist():
            acc = 0.0
            for i in range(self.iters):
                acc += (v * i) % 7.0
            out.append(acc % 1000.0)
        return batch.append_column("r", pa.array(out, type=pa.float64()))


class SleepUDF:
    """Latency-bound: a fixed per-batch sleep (models a slow remote call)."""

    def __init__(self, ms: float):
        self.ms = ms

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import time as _t

        _t.sleep(self.ms / 1000.0)
        return batch.append_column("r", pa.array([0] * batch.num_rows, type=pa.int64()))


class ArrowUDF:
    """Near-noop: a vectorized Arrow op, no Python per-row loop (measures pure
    dispatch/serialization overhead of each backend)."""

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import pyarrow.compute as pc

        return batch.append_column("r", pc.add(batch.column(0), 1))


def make_table(rows: int) -> pa.Table:
    return pa.table({"x": np.arange(rows, dtype="int64")})


def run_backend(table, fn, backend: str, batch: int, workers: int) -> float:
    """Return rows/s for one (backend, workers) cell. Warms the pool once."""
    con = jude.connect()
    con.register("t", table)
    kw = {"batch_size": batch}
    if backend != "in_process":
        kw["execution_backend"] = backend
        kw["num_workers"] = workers

    # warm (pool spin-up / actor start amortized out of the timing)
    con.sql("SELECT * FROM t LIMIT 1").map_batches(fn, **kw).num_rows
    rel = con.sql("SELECT * FROM t")
    t0 = time.perf_counter()
    n = rel.map_batches(fn, **kw).num_rows
    dt = time.perf_counter() - t0
    return n / dt if dt > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cpu-iters", type=int, default=20000)
    ap.add_argument("--sleep-ms", type=float, default=5.0)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["in_process", "subprocess", "ray_actor"],
        help="subset of {in_process, subprocess, ray_task, ray_actor}",
    )
    ap.add_argument("--workloads", nargs="+", default=["cpu", "sleep", "arrow"])
    args = ap.parse_args()

    table = make_table(args.rows)
    workloads = {
        "cpu": CpuUDF(args.cpu_iters),
        "sleep": SleepUDF(args.sleep_ms),
        "arrow": ArrowUDF(),
    }

    has_ray = any(b.startswith("ray") for b in args.backends)
    if has_ray:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
        except Exception as e:  # pragma: no cover
            print(f"(ray unavailable: {e}; dropping ray backends)")
            args.backends = [b for b in args.backends if not b.startswith("ray")]

    print(
        f"\nUDF backend matrix — rows={args.rows} batch={args.batch} "
        f"cpu_iters={args.cpu_iters} sleep_ms={args.sleep_ms}"
    )
    for wl in args.workloads:
        if wl not in workloads:
            continue
        fn = workloads[wl]
        print(f"\n[{wl}] rows/s by backend × workers")
        header = "  " + "backend".ljust(14) + "".join(f"{w:>12}" for w in args.workers)
        print(header)
        print("  " + "-" * (14 + 12 * len(args.workers)))
        base_row = None
        for backend in args.backends:
            cells = []
            wlist = [1] if backend == "in_process" else args.workers
            for w in args.workers:
                use_w = w if backend != "in_process" else 1
                try:
                    thr = run_backend(table, fn, backend, args.batch, use_w)
                    cells.append(thr)
                except Exception as e:  # pragma: no cover
                    cells.append(None)
                    print(f"    ! {backend} w={w}: {type(e).__name__}: {e}")
            row = "  " + backend.ljust(14) + "".join(
                (f"{c:>12,.0f}" if c is not None else f"{'—':>12}") for c in cells
            )
            print(row)
            if backend == "in_process" and cells and cells[0]:
                base_row = cells[0]
        if base_row:
            print(f"  (in_process 1-worker baseline = {base_row:,.0f} rows/s)")

    try:
        jude.shutdown_udf_pools()
    except Exception:
        pass
    try:
        from jude.execution import shutdown_ray_pools

        shutdown_ray_pools()
    except Exception:
        pass


if __name__ == "__main__":
    main()
