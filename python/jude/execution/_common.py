"""Batching + call-mode helpers shared by all execution backends.

Mirrors the role of Vane's duckdb/execution/_common.py + _udf_runtime.py: turn a
user callable into a batch-in/batch-out function according to its call mode, and
provide row/byte-based rebatching.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pyarrow as pa

CALL_MODES = ("map_batches", "map_batches_rows", "flat_map", "map")


def coerce_table(result: Any) -> pa.Table:
    """Normalize a UDF result (Table / RecordBatch / dict / iterator) to a Table."""
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatch):
        return pa.Table.from_batches([result])
    if isinstance(result, dict):
        return pa.table(result)
    if hasattr(result, "__iter__") and not isinstance(result, (bytes, str)):
        parts = [coerce_table(r) for r in result]
        return pa.concat_tables(parts) if parts else pa.table({})
    raise TypeError(f"UDF returned unsupported type {type(result)!r}")


def rechunk(table: pa.Table, batch_size: int | None) -> list[pa.Table]:
    """Split a table into row-count batches (whole table if batch_size falsy)."""
    if not batch_size or batch_size <= 0:
        return [table]
    out = []
    n = table.num_rows
    for start in range(0, n, batch_size):
        out.append(table.slice(start, min(batch_size, n - start)))
    return out or [table]


def apply_callable(fn: Callable, table: pa.Table, call_mode: str) -> pa.Table:
    """Apply a user callable to one batch according to its call mode."""
    if call_mode in ("map_batches", "map_batches_rows", "flat_map"):
        return coerce_table(fn(table))
    if call_mode == "map":
        # scalar per-row over the first column
        col = table.column(0).to_pylist()
        return pa.table({"result": pa.array([fn(v) for v in col])})
    raise ValueError(f"unknown call_mode {call_mode!r}")


def iter_apply(fn: Callable, table: pa.Table, call_mode: str, batch_size: int | None) -> Iterator[pa.Table]:
    """Streaming apply: yield one output table per input batch (generator)."""
    for chunk in rechunk(table, batch_size):
        yield apply_callable(fn, chunk, call_mode)


def load_callable(payload: dict) -> Callable:
    """Unpickle a UDF from a serialize_udf payload; instantiate actor classes."""
    import cloudpickle

    fn = cloudpickle.loads(bytes.fromhex(payload["fn_hex"]))
    if payload.get("is_class") and isinstance(fn, type):
        fn = fn()
    return fn
