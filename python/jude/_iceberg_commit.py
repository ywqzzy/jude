"""jude._iceberg_commit — the Iceberg commit metadata step.

jude writes the *data* files (Parquet) in Rust via DuckDB `COPY TO` (the heavy,
parallelizable path). This module does only the table-format **commit**: register
those already-written Parquet files into an Iceberg table's metadata (manifest +
snapshot) via pyiceberg's `add_files`. It creates the table (inferring the schema
from the written Parquet) on first write. No data is rewritten here — the files
Rust produced are committed as-is.

Kept thin and separate because the commit protocol (atomic metadata swap, snapshot
lineage, optimistic-concurrency conflict handling) is exactly what pyiceberg
already implements correctly; jude's job is to hand it the file list.
"""

from __future__ import annotations

import os
from typing import Any


def _catalog(warehouse: str) -> Any:
    from pyiceberg.catalog.sql import SqlCatalog

    os.makedirs(warehouse, exist_ok=True)
    return SqlCatalog(
        "jude",
        uri=f"sqlite:///{os.path.join(warehouse, 'catalog.db')}",
        warehouse=f"file://{warehouse}",
    )


def commit(warehouse: str, table_ident: str, parquet_files: list[str], mode: str) -> str:
    """Commit `parquet_files` (already written by jude/Rust) into the Iceberg
    table `table_ident` under `warehouse`. `mode` is 'append' or 'overwrite'.
    Returns the new metadata-location so the caller can read it back.
    """
    import pyarrow.parquet as pq

    if "." not in table_ident:
        table_ident = f"default.{table_ident}"
    namespace = table_ident.rsplit(".", 1)[0]

    cat = _catalog(warehouse)
    try:
        cat.create_namespace(namespace)
    except Exception:
        pass  # already exists

    schema = pq.read_schema(parquet_files[0])

    if cat.table_exists(table_ident):
        tbl = cat.load_table(table_ident)
        if mode == "overwrite":
            # Drop existing data files, then add the new ones (a replace commit).
            tbl.delete()
    else:
        tbl = cat.create_table(table_ident, schema=schema)

    # add_files registers the Parquet files jude already wrote — no rewrite.
    tbl.add_files([os.path.abspath(f) for f in parquet_files])
    return tbl.metadata_location


__all__ = ["commit"]
