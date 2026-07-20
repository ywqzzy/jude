"""Generator / table UDFs — a Python function produces a table of rows that
becomes a jude relation.

A native SQL-callable table function (`FROM myfn(args)`) would need a DuckDB VTab
bridge (static types, DuckDB-native vector output); jude instead runs the
generator at the materialization boundary and normalizes whatever it returns to
Arrow. Accepts: a pyarrow Table/RecordBatch/RecordBatchReader, a pandas/polars
frame (anything with an Arrow C stream), a list of dicts, or a list/iterable of
tuples/lists (with an explicit ``schema`` of column names).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def normalize_to_arrow(result: Any, schema: Any = None) -> pa.Table:
    # Already Arrow-native.
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatch):
        return pa.Table.from_batches([result])
    if isinstance(result, pa.RecordBatchReader):
        return result.read_all()
    # Anything exposing the Arrow C stream (pandas via pyarrow, polars, etc.).
    if hasattr(result, "__arrow_c_stream__"):
        return pa.table(result)
    # Materialize a generator/iterator of rows.
    if hasattr(result, "__iter__") and not isinstance(result, (list, tuple, dict)):
        result = list(result)
    if isinstance(result, dict):
        return pa.table(result)
    if isinstance(result, (list, tuple)):
        rows = list(result)
        if not rows:
            names = _schema_names(schema)
            return pa.table({n: pa.array([]) for n in names}) if names else pa.table({})
        if isinstance(rows[0], dict):
            return pa.Table.from_pylist(rows)
        # Rows are tuples/lists -> need column names.
        names = _schema_names(schema)
        if not names:
            names = [f"column{i}" for i in range(len(rows[0]))]
        cols = {names[i]: pa.array([r[i] for r in rows]) for i in range(len(names))}
        return pa.table(cols)
    raise TypeError(f"table UDF returned an unsupported type: {type(result).__name__}")


def _schema_names(schema: Any) -> list[str]:
    if schema is None:
        return []
    if isinstance(schema, pa.Schema):
        return list(schema.names)
    if isinstance(schema, dict):
        return list(schema.keys())
    if isinstance(schema, (list, tuple)):
        return [str(s) for s in schema]
    return []
