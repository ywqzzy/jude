"""User-defined aggregate UDFs via group-apply.

DuckDB (through the stock ``duckdb`` crate) exposes no way to register a native
Python aggregate function, so jude implements aggregate UDFs at the
materialization boundary: DuckDB does the grouping (fast, in-engine) by
collecting each group's rows with ``list(...)``; jude then hands each group's
rows to the Python function and reduces to one row per group.

``fn`` receives a ``pyarrow.Table`` of the group's rows (the requested columns)
and returns either a scalar (becomes ``result_name``) or a ``dict``/mapping of
output-column -> scalar. Global aggregation (no ``group_by``) yields one row.
"""

from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa


def apply_group_aggregate(
    grouped: pa.Table,
    list_cols: list[str],
    group_cols: list[str],
    columns: list[str],
    fn: Callable[[pa.Table], Any],
    result_name: str,
) -> pa.Table:
    """Reduce a ``list()``-collected grouped table to one row per group.

    ``grouped`` has the group-key columns plus one ``list<...>`` column per
    requested input column (aligned with ``list_cols`` / ``columns``). For each
    group row we rebuild a small ``pyarrow.Table`` of that group's rows and call
    ``fn`` on it.
    """
    n = grouped.num_rows
    # Collect group-key column values (python objects), preserved for output.
    key_values: dict[str, list[Any]] = {g: grouped.column(g).to_pylist() for g in group_cols}
    # Per-group list columns.
    group_lists = {c: grouped.column(lc).to_pylist() for c, lc in zip(columns, list_cols)}

    results: list[Any] = []
    extra_cols: dict[str, list[Any]] = {}
    for i in range(n):
        group_table = pa.table({c: pa.array(group_lists[c][i] or []) for c in columns})
        out = fn(group_table)
        if isinstance(out, dict):
            for k, v in out.items():
                extra_cols.setdefault(k, [None] * n)
                extra_cols[k][i] = v
            results.append(None)
        else:
            results.append(out)

    data: dict[str, Any] = {g: key_values[g] for g in group_cols}
    if extra_cols:
        for k, vals in extra_cols.items():
            data[k] = vals
    else:
        data[result_name] = results
    return pa.table(data) if data else pa.table({result_name: results})
