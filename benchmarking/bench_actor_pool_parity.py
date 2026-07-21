"""Bench: ray_actor RESIDENT pool vs subprocess/ray_task — proves the actor pool
amortizes startup so repeated map_batches don't re-pay actor+model init (the
'actor startup overhead' item). Evidence: first ray_actor call pays startup;
subsequent calls are warm (~ray_task per-call cost or better), i.e. at parity.

    python -m benchmarking.bench_actor_pool_parity --calls 6 --rows 200000
"""

from __future__ import annotations

import argparse
import time

import pyarrow as pa

import jude


def _work(t: pa.Table) -> pa.Table:
    # a trivial map so we measure dispatch/startup overhead, not the UDF itself
    x = t.column("x")
    return t.set_column(t.column_names.index("x"), "x",
                        pa.array([v.as_py() + 1 for v in x], type=pa.int64()))


def _time_calls(backend: str, calls: int, rows: int, workers: int) -> list[float]:
    ray = __import__("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    con = jude.connect()
    times = []
    for _ in range(calls):
        rel = con.from_arrow(pa.table({"x": list(range(rows))}))
        t0 = time.perf_counter()
        out = rel.map_batches(_work, execution_backend=backend, num_workers=workers)
        _ = out.to_arrow() if hasattr(out, "to_arrow") else out
        times.append(time.perf_counter() - t0)
    return times


def run(calls: int = 6, rows: int = 200000, workers: int = 4) -> dict:
    from jude.execution import shutdown_ray_pools

    shutdown_ray_pools()  # cold start
    actor = _time_calls("ray_actor", calls, rows, workers)
    task = _time_calls("ray_task", calls, rows, workers)
    return {
        "ray_actor": [round(t, 4) for t in actor],
        "ray_task": [round(t, 4) for t in task],
        "actor_first": round(actor[0], 4),
        "actor_warm_avg": round(sum(actor[1:]) / max(1, len(actor) - 1), 4),
        "task_avg": round(sum(task) / len(task), 4),
        # warm actor calls should be within a small factor of ray_task (no
        # per-call actor startup) — the parity claim.
        "warm_vs_task_ratio": round((sum(actor[1:]) / max(1, len(actor) - 1)) /
                                    (sum(task) / len(task)), 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    r = run(a.calls, a.rows, a.workers)
    print(f"\nray_actor per-call: {r['ray_actor']}")
    print(f"ray_task  per-call: {r['ray_task']}")
    print(f"\nactor 1st (cold, pays startup): {r['actor_first']}s")
    print(f"actor warm avg (calls 2+):      {r['actor_warm_avg']}s")
    print(f"ray_task avg:                   {r['task_avg']}s")
    print(f"warm-actor / ray_task ratio:    {r['warm_vs_task_ratio']}x  "
          f"({'PARITY' if r['warm_vs_task_ratio'] <= 1.5 else 'actor slower'})")
