"""Standing goal (a): the ray_actor RESIDENT pool amortizes actor startup — warm
calls (2nd+) are at parity with a stateless ray_task (no per-call actor init),
i.e. the 'actor startup overhead' no longer slows repeated map_batches."""

from __future__ import annotations

import pytest

pytest.importorskip("ray")


def test_ray_actor_pool_is_resident_and_amortizes_startup():
    from benchmarking.bench_actor_pool_parity import run

    r = run(calls=5, rows=40000, workers=2)
    # cold first call pays startup; warm calls are dramatically cheaper
    assert r["actor_first"] > r["actor_warm_avg"] * 3, r
    # warm actor calls are at parity with ray_task (within a small factor) —
    # the resident pool means no per-call actor/model startup.
    assert r["warm_vs_task_ratio"] <= 1.5, r


def test_actor_pool_cached_across_calls():
    # the same (fn, workers, gpus, mode) reuses one pool object across calls
    import pyarrow as pa
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    import jude
    from jude.execution import _ACTOR_POOLS, shutdown_ray_pools

    shutdown_ray_pools()
    con = jude.connect()
    fn = lambda t: t  # noqa: E731
    for _ in range(3):
        con.from_arrow(pa.table({"x": [1, 2, 3]})).map_batches(
            fn, execution_backend="ray_actor", num_workers=2).to_arrow()
    assert len(_ACTOR_POOLS) == 1        # one resident pool reused, not rebuilt per call
