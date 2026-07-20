"""Ray execution backends for UDFs: ray_task and ray_actor.

Mirrors Vane's duckdb/execution/udf_ray.py:

- ``ray_task``  — stateless: each batch runs in a fresh Ray task. Simple,
  elastic, no warm state. Good for pure functions.
- ``ray_actor`` — stateful: a pool of long-lived Ray actors, each loads the UDF
  (and its model weights) once. Batches are dispatched round-robin. This is the
  backend for GPU model inference and ``jude.cls`` actors.

Both carry Arrow tables through the Ray object store (zero-copy) and preserve
input order. Results can be collected eagerly or streamed as they complete.
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa
import ray

from jude.execution._common import apply_callable, coerce_table, load_callable, rechunk


def _record_pool_metrics(ex: "RayActorExecutor", n_batches: int, rows: int, secs: float) -> None:
    """Best-effort: report actor-pool utilization to the observability registry.

    Observability must never break execution, so any failure is swallowed.
    """
    try:
        from jude import observe

        key = "ray_actor:" + str(ex.payload.get("fn_hex", ""))[:16] + f":{ex.num_workers}"
        observe.record_pool(
            key,
            "ray_actor",
            ex.num_workers,
            batches_in=n_batches,
            batches_out=n_batches,
            rows=rows,
            busy_ns=int(secs * 1e9),
        )
    except Exception:  # noqa: BLE001
        pass


@ray.remote
def _ray_task_apply(payload: dict, table: pa.Table, call_mode: str) -> pa.Table:
    fn = load_callable(payload)
    return apply_callable(fn, table, call_mode)


@ray.remote
class _RayUDFActor:
    def __init__(self, payload: dict, num_gpus: float):
        import os

        if num_gpus > 0:
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        # Load once (weights persist across batches — stateful actor contract).
        self.fn = load_callable(payload)
        self.call_mode = payload.get("call_mode", "map_batches")

    def apply(self, table: pa.Table) -> pa.Table:
        return apply_callable(self.fn, table, self.call_mode)

    def ping(self) -> bool:
        """Cheap readiness probe — used for eager warm-up so the first real
        batch doesn't pay actor-start + UDF-load latency."""
        return True

    def node_id(self) -> str:
        """Ray node this actor is placed on (for observability / placement checks)."""
        return str(ray.get_runtime_context().get_node_id())


class RayTaskExecutor:
    """Stateless Ray-task backend."""

    def __init__(self, payload: dict, call_mode: str = "map_batches"):
        self.payload = dict(payload)
        self.payload["call_mode"] = call_mode
        self.call_mode = call_mode

    def map(self, table: pa.Table, batch_size: int | None = None) -> pa.Table:
        chunks = rechunk(table, batch_size)
        refs = [_ray_task_apply.remote(self.payload, c, self.call_mode) for c in chunks]
        outs = ray.get(refs)
        return pa.concat_tables(outs).combine_chunks() if outs else table.slice(0, 0)

    def imap(self, table: pa.Table, batch_size: int | None = None) -> Iterator[pa.Table]:
        """Streaming: yield outputs as tasks complete (order not guaranteed)."""
        chunks = rechunk(table, batch_size)
        refs = [_ray_task_apply.remote(self.payload, c, self.call_mode) for c in chunks]
        while refs:
            done, refs = ray.wait(refs, num_returns=1)
            yield ray.get(done[0])


class RayActorExecutor:
    """Stateful Ray-actor-pool backend."""

    def __init__(self, payload: dict, call_mode: str = "map_batches", num_workers: int = 1, num_gpus: float = 0.0):
        self.payload = dict(payload)
        self.payload["call_mode"] = call_mode
        self.num_workers = max(1, num_workers)
        self.num_gpus = num_gpus
        self._actors: list[Any] = []

    def _pool(self) -> list[Any]:
        if not self._actors:
            import os

            opts: dict[str, Any] = {}
            if self.num_gpus > 0:
                opts["num_gpus"] = self.num_gpus
            # Actor-level fault tolerance is OPT-IN and OFF by default: the
            # preferred recovery for a distributed op is a coarse whole-query
            # retry (RayRunner.with_retry), not fine-grained actor/task retries.
            # Set these envs > 0 only if you specifically want a resident actor
            # to survive/restart a worker fault in place.
            restarts = int(os.environ.get("JUDE_UDF_ACTOR_MAX_RESTARTS", "0"))
            task_retries = int(os.environ.get("JUDE_UDF_ACTOR_MAX_TASK_RETRIES", "0"))
            if restarts:
                opts["max_restarts"] = restarts
                if task_retries:
                    opts["max_task_retries"] = task_retries
            cls = _RayUDFActor.options(**opts)
            self._actors = [cls.remote(self.payload, self.num_gpus) for _ in range(self.num_workers)]
            # Eager warm-up: block until every actor has started and loaded the
            # UDF (ping), so the first real batch isn't slowed by cold starts.
            # Opt out with JUDE_UDF_EAGER_WARMUP=0.
            if os.environ.get("JUDE_UDF_EAGER_WARMUP", "1") != "0":
                try:
                    ray.get([a.ping.remote() for a in self._actors])
                except Exception:  # noqa: BLE001 — warm-up is best-effort
                    pass
        return self._actors

    def map(self, table: pa.Table, batch_size: int | None = None) -> pa.Table:
        import time

        chunks = rechunk(table, batch_size)
        pool = self._pool()
        t0 = time.perf_counter()
        refs = [pool[i % len(pool)].apply.remote(c) for i, c in enumerate(chunks)]
        outs = ray.get(refs)  # order preserved (refs are in submission order)
        result = pa.concat_tables(outs).combine_chunks() if outs else table.slice(0, 0)
        _record_pool_metrics(self, len(chunks), result.num_rows, time.perf_counter() - t0)
        return result

    def imap(self, table: pa.Table, batch_size: int | None = None) -> Iterator[pa.Table]:
        chunks = rechunk(table, batch_size)
        pool = self._pool()
        refs = [pool[i % len(pool)].apply.remote(c) for i, c in enumerate(chunks)]
        for ref in refs:
            yield ray.get(ref)

    def shutdown(self) -> None:
        for a in self._actors:
            try:
                ray.kill(a)
            except Exception:
                pass
        self._actors = []
