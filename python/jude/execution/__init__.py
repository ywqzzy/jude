"""jude.execution — unified out-of-process UDF execution engine.

Mirrors Vane's ``duckdb/execution/`` layer: a single ``build_executor`` router
that dispatches on ``execution_backend`` to one of:

- ``subprocess_task`` / ``subprocess_actor`` — the Rust subprocess pool
  (jude._udf, GIL-free), driven from the Relation.
- ``ray_task``  — stateless Ray tasks (udf_ray.RayTaskExecutor).
- ``ray_actor`` — stateful Ray actor pool (udf_ray.RayActorExecutor).

Each executor exposes ``map(table, batch_size)`` (eager) and ``imap(...)``
(streaming generator, one output per input batch). The subprocess path stays in
Rust for the GIL-free hot loop; the Ray paths live here.
"""

from __future__ import annotations

from typing import Any

__all__ = ["serialize_udf", "build_executor", "BACKENDS", "shutdown_ray_pools"]

BACKENDS = ("subprocess_task", "subprocess_actor", "ray_task", "ray_actor")

# Resident ray_actor pools, keyed by (udf, num_workers, num_gpus, call_mode), so
# actor startup + per-actor setup (model load) is amortized across map_batches.
_ACTOR_POOLS: dict = {}


def shutdown_ray_pools() -> None:
    """Tear down all cached resident ray_actor pools (their actors)."""
    for ex in list(_ACTOR_POOLS.values()):
        try:
            ex.shutdown()
        except Exception:  # noqa: BLE001
            pass
    _ACTOR_POOLS.clear()


def serialize_udf(fn, *, call_mode: str = "map_batches", is_class: bool = False) -> dict:
    """Pickle a UDF callable (with cloudpickle) into a control payload dict.

    The callable is pickled *by value* whenever possible so worker processes do
    not need to import the module that defined it.
    """
    import cloudpickle

    mod = getattr(fn, "__module__", None)
    if mod and mod not in ("builtins", "__main__"):
        try:
            import importlib

            cloudpickle.register_pickle_by_value(importlib.import_module(mod))
        except Exception:
            pass

    return {
        "fn_hex": cloudpickle.dumps(fn).hex(),
        "call_mode": call_mode,
        "is_class": is_class,
    }


def build_executor(
    payload: dict,
    execution_backend: str = "ray_task",
    *,
    num_workers: int = 1,
    num_gpus: float = 0.0,
) -> Any:
    """Construct an executor for the requested backend.

    Returns an object with ``map(table, batch_size)`` and ``imap(...)``.
    Subprocess backends are handled in the Relation (Rust pool) and are not
    built here; this router covers the Ray backends.
    """
    call_mode = payload.get("call_mode", "map_batches")
    if execution_backend in ("ray_task",):
        from jude.execution.udf_ray import RayTaskExecutor

        return RayTaskExecutor(payload, call_mode=call_mode)
    if execution_backend in ("ray_actor",):
        from jude.execution.udf_ray import RayActorExecutor

        return RayActorExecutor(payload, call_mode=call_mode, num_workers=num_workers, num_gpus=num_gpus)
    raise ValueError(
        f"build_executor: unsupported backend {execution_backend!r} "
        f"(subprocess backends are handled by the Relation's Rust pool)"
    )


def run_ray_map(
    payload: dict,
    table: Any,
    execution_backend: str = "ray_task",
    batch_size: int | None = None,
    num_workers: int = 1,
    num_gpus: float = 0.0,
) -> Any:
    """Apply a UDF payload to a pyarrow Table via a Ray backend; return a Table.

    Entry point called from the Rust Relation for
    ``map_batches(execution_backend='ray_task'|'ray_actor')``.
    """
    # ray_actor is a RESIDENT pool: cache it across calls so actor startup (and
    # any heavy per-actor setup, e.g. loading a model) is paid once, not per
    # map_batches. ray_task is stateless -> build and tear down each call.
    if execution_backend == "ray_actor":
        key = (payload.get("fn_hex"), int(num_workers), float(num_gpus), payload.get("call_mode", "map_batches"))
        ex = _ACTOR_POOLS.get(key)
        if ex is None:
            ex = build_executor(payload, execution_backend, num_workers=num_workers, num_gpus=num_gpus)
            _ACTOR_POOLS[key] = ex
        return ex.map(table, batch_size=batch_size)  # resident: no per-call shutdown
    ex = build_executor(payload, execution_backend, num_workers=num_workers, num_gpus=num_gpus)
    try:
        return ex.map(table, batch_size=batch_size)
    finally:
        if hasattr(ex, "shutdown"):
            ex.shutdown()
