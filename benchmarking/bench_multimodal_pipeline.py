#!/usr/bin/env python3
"""Multimodal pipeline benchmark — mirrors Vane's public benchmark page
(vane.astrovela.ai/benchmarks): four workloads (Document / Image / Audio /
Video), each `decode -> model -> output`, comparing engines by elapsed time.

WHY CPU MODE (no GPU needed): Vane's page runs a real GPU model per workload
(embedding / ResNet / Whisper / YOLO). On GPU the benchmark measures whether the
DATA ENGINE can keep the GPU fed (decode/batch/schedule without stalls). With no
GPU we replace the model with a deterministic CPU-bound stand-in of comparable
relative cost per workload. This isolates the **scheduling / GIL** axis — exactly
where jude's Rust control plane + out-of-process pool win — and runs anywhere.
It does NOT measure GPU utilization; it measures the engine's ability to drive a
per-item Python "model" in parallel. Honest proxy, same shape as the page.

Data is synthetic (generated in-process), so no downloads. Each workload has a
decode cost and a per-item model cost tuned to the workload's relative weight
(video heaviest, document lightest), matching the page's ordering.

    python benchmarking/bench_multimodal_pipeline.py --items 400 --workers 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa


# --- workload definitions (module-level = picklable to workers) --------------

# (name, n_items, decode_iters, model_iters, blob_size) — relative costs mirror
# the page's ordering: document light, video heaviest.
WORKLOADS = {
    "document": dict(decode=2000, model=8000, size=256),
    "image": dict(decode=6000, model=20000, size=1024),
    "audio": dict(decode=20000, model=40000, size=2048),
    "video": dict(decode=60000, model=60000, size=4096),
}


def _cpu_work(blob: bytes, iters: int) -> float:
    arr = np.frombuffer(blob, dtype="uint8").astype("float32")
    acc = 0.0
    for i in range(iters):
        acc += (arr[i % arr.size] * i) % 7.0
    return acc % 1000.0


class PipelineUDF:
    """decode + 'model' for one workload, over a batch (Arrow table -> table)."""

    def __init__(self, decode_iters: int, model_iters: int):
        self.decode_iters = decode_iters
        self.model_iters = model_iters

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import numpy as np
        import pyarrow as pa

        out = []
        for blob in batch["blob"].to_pylist():
            # decode stage (cheap-ish) then model stage (dominant)
            _ = _cpu_work(blob, self.decode_iters)
            out.append(_cpu_work(blob, self.model_iters))
        return batch.append_column("result", pa.array(out, type=pa.float32()))


def make_data(n: int, size: int, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    blobs = [rng.integers(0, 256, size, dtype="uint8").tobytes() for _ in range(n)]
    return pa.table({"id": list(range(n)), "blob": pa.array(blobs, type=pa.binary())})


def run_jude(table, decode_iters, model_iters, batch_size, workers) -> float:
    import jude

    con = jude.connect()
    con.register("t", table)
    fn = PipelineUDF(decode_iters, model_iters)
    con.sql("SELECT * FROM t LIMIT 1").map_batches(fn, execution_backend="subprocess", num_workers=workers).num_rows
    rel = con.sql("SELECT * FROM t")
    t0 = time.perf_counter()
    rel.map_batches(fn, batch_size=batch_size, execution_backend="subprocess", num_workers=workers).num_rows
    dt = time.perf_counter() - t0
    try:
        jude.shutdown_udf_pools()
    except Exception:
        pass
    return dt


def run_daft(table, decode_iters, model_iters, batch_size, workers) -> float:
    import daft

    dec, mod = decode_iters, model_iters

    @daft.udf(return_dtype=daft.DataType.float32(), batch_size=batch_size, num_cpus=1, concurrency=workers)
    def infer(blob):
        blobs = blob.to_pylist()
        out = []
        for b in blobs:
            _ = _cpu_work(b, dec)
            out.append(_cpu_work(b, mod))
        return out

    df = daft.from_arrow(table)
    daft.from_arrow(table.slice(0, 1)).with_column("r", infer(daft.col("blob"))).to_arrow()
    t0 = time.perf_counter()
    df.with_column("r", infer(daft.col("blob"))).to_arrow()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=400, help="items per workload")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--engines", nargs="+", default=["jude", "daft"])
    args = ap.parse_args()

    print(f"\nMultimodal pipeline bench (CPU mode) — {args.items} items/workload, {args.workers} workers")
    print("mirrors vane.astrovela.ai/benchmarks structure (decode -> model), GPU model -> CPU stand-in")
    print("-" * 70)
    header = "  " + "workload".ljust(12) + "".join(e.rjust(14) for e in args.engines) + "jude speedup".rjust(16)
    print(header)
    print("  " + "-" * (12 + 14 * len(args.engines) + 16))

    for wl in args.workloads:
        cfg = WORKLOADS[wl]
        data = make_data(args.items, cfg["size"])
        times: dict[str, float] = {}
        for eng in args.engines:
            try:
                if eng == "jude":
                    times[eng] = run_jude(data, cfg["decode"], cfg["model"], args.batch_size, args.workers)
                elif eng == "daft":
                    times[eng] = run_daft(data, cfg["decode"], cfg["model"], args.batch_size, args.workers)
            except Exception as e:  # pragma: no cover
                times[eng] = float("nan")
                print(f"    ! {eng} {wl}: {type(e).__name__}: {e}")
        cells = "".join((f"{times[e]:>11.1f}s" if e in times else f"{'—':>12}") for e in args.engines)
        speed = ""
        if "jude" in times and "daft" in times and times["jude"] > 0:
            speed = f"{times['daft'] / times['jude']:>13.2f}x"
        print("  " + wl.ljust(12) + cells + speed.rjust(16))
    print("-" * 70)


if __name__ == "__main__":
    main()
