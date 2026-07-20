"""DuckDB-style replacement scans.

Lets ``conn.sql("SELECT * FROM my_df")`` resolve ``my_df`` from the caller's
Python scope when it isn't a registered table — matching DuckDB / Vane, where a
pandas DataFrame, pyarrow Table/RecordBatch, or polars DataFrame in scope can be
queried by its variable name.

Because jude relations are lazy (the catalog error would surface far from the
call site), we resolve *eagerly* at ``sql()`` time: scan the query for bare
identifiers, look them up in the caller's frames, and register any Arrow-like
object found under that name as a temp view on the connection.
"""

from __future__ import annotations

import re
import sys
from typing import Any

# Bare identifiers that could be table names. We only *register* ones that (a)
# resolve to an Arrow-like object in scope and (b) are not already a real table,
# so this never shadows SQL keywords or existing tables.
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

_SQL_KEYWORDS = frozenset(
    {
        "select", "from", "where", "group", "by", "order", "limit", "offset",
        "join", "inner", "left", "right", "full", "outer", "cross", "on", "as",
        "and", "or", "not", "in", "is", "null", "distinct", "union", "all",
        "except", "intersect", "having", "with", "values", "case", "when",
        "then", "else", "end", "asc", "desc", "using", "count", "sum", "avg",
        "min", "max", "cast", "create", "table", "view", "insert", "into",
        "true", "false", "between", "like", "exists", "over", "partition",
    }
)


def _is_arrow_like(obj: Any) -> bool:
    import pyarrow as pa

    # pyarrow Table / RecordBatch / RecordBatchReader
    if isinstance(obj, (pa.Table, pa.RecordBatch)):
        return True
    if hasattr(obj, "to_batches") or hasattr(obj, "read_all"):
        return True
    tn = type(obj).__module__ or ""
    # pandas / polars dataframes
    if tn.startswith("pandas") and type(obj).__name__ == "DataFrame":
        return True
    if tn.startswith("polars") and type(obj).__name__ in ("DataFrame", "LazyFrame"):
        return True
    return False


def _is_relation(obj: Any) -> bool:
    """A jude Relation (DuckDBPyRelation) in scope can be scanned by name."""
    import jude

    return isinstance(obj, jude.Relation)


def _scannable(obj: Any) -> bool:
    return _is_arrow_like(obj) or _is_relation(obj)


def _to_arrow(obj: Any) -> Any:
    import pyarrow as pa

    if _is_relation(obj):
        return obj.to_arrow_table()
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    if hasattr(obj, "read_all"):  # RecordBatchReader
        return obj.read_all()
    tn = type(obj).__module__ or ""
    if tn.startswith("polars"):
        # polars DataFrame / LazyFrame
        df = obj.collect() if type(obj).__name__ == "LazyFrame" else obj
        return df.to_arrow()
    if tn.startswith("pandas"):
        return pa.Table.from_pandas(obj)
    return pa.table(obj)


def register_scan_candidates(
    conn: Any, query: str, depth: int = 1, all_frames: bool = False
) -> None:
    """Register any in-scope Arrow-like / jude-Relation variables referenced by
    name in `query` as temp views.

    `conn` is the jude Connection. `depth` is how many Python frames up to start
    looking; Rust callers add no Python frame, so depth=1 is the user's frame
    (the caller of conn.sql/execute). By DuckDB semantics only that single frame
    is scanned unless `all_frames` is set, in which case we walk outward.
    """
    idents = {m.group(1) for m in _IDENT_RE.finditer(query)}
    if not idents:
        return

    seen: dict[str, Any] = {}
    frame = None
    try:
        frame = sys._getframe(depth)
    except ValueError:
        return
    while frame is not None and idents:
        for name in list(idents):
            if name in seen:
                continue
            val = frame.f_locals.get(name)
            if val is None:
                val = frame.f_globals.get(name)
            if val is not None and _scannable(val):
                seen[name] = val
        if not all_frames:
            break
        frame = frame.f_back

    for name, val in seen.items():
        try:
            conn.register(name, _to_arrow(val))
        except Exception:
            # best-effort: if registration fails, let the SQL error surface
            pass
