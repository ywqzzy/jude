"""jude.training_format — write curated data to training-ready formats (C8).

The "processing -> training" last mile. Training frameworks (PyTorch,
MosaicML Composer, Megatron, WebDataset loaders) want **sharded, streamable,
size-aligned** datasets, not one giant file. This module writes a jude
Relation / Arrow table to:

- **WebDataset** (``write_webdataset``) — ``.tar`` shards, each record a group
  of files sharing a key (``<key>.<ext>``); the de-facto format for large-scale
  multimodal training (img2dataset, LAION).
- **Mosaic MDS** (``write_mds``) — MosaicML StreamingDataset shards + an ``index.json``;
  streamable from object storage with deterministic sharding.
- **sharded Parquet** (``write_sharded_parquet``) — size-aligned Parquet shards
  (N rows or ~target bytes per shard) for frameworks that read Parquet.

All writers shard by a target rows-per-shard (or byte budget) so shard sizes are
uniform — the property training loaders need for balanced parallel reads.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pyarrow as pa

__all__ = ["write_sharded_parquet", "write_webdataset", "write_mds"]


def _as_table(data: Any) -> pa.Table:
    if isinstance(data, pa.Table):
        return data
    if hasattr(data, "to_arrow"):
        return data.to_arrow()
    raise TypeError(f"expected pyarrow.Table or jude Relation, got {type(data)!r}")


def _shard_row_ranges(n: int, rows_per_shard: int) -> list[tuple[int, int]]:
    rps = max(1, rows_per_shard)
    return [(s, min(rps, n - s)) for s in range(0, n, rps)]


def write_sharded_parquet(
    data: Any, out_dir: str, *, rows_per_shard: int = 100_000, prefix: str = "shard"
) -> dict:
    """Write size-aligned Parquet shards (``<prefix>-00000.parquet`` ...)."""
    import pyarrow.parquet as pq

    table = _as_table(data)
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for i, (start, length) in enumerate(_shard_row_ranges(table.num_rows, rows_per_shard)):
        path = os.path.join(out_dir, f"{prefix}-{i:05d}.parquet")
        pq.write_table(table.slice(start, length), path)
        shards.append({"path": path, "rows": length})
    return {"format": "parquet", "shards": shards, "num_shards": len(shards), "rows": table.num_rows}


def write_webdataset(
    data: Any,
    out_dir: str,
    *,
    rows_per_shard: int = 10_000,
    key_column: str | None = None,
    text_columns: list[str] | None = None,
    binary_columns: dict[str, str] | None = None,
    prefix: str = "shard",
) -> dict:
    """Write WebDataset ``.tar`` shards. Each row becomes a record whose files
    are ``<key>.<ext>``:

    - ``text_columns``: written as ``<key>.<col>.txt`` (utf-8).
    - ``binary_columns``: {column: ext} written as ``<key>.<ext>`` (raw bytes),
      e.g. {"image": "jpg"}.
    - remaining scalar columns are bundled into ``<key>.json``.
    ``key_column`` supplies the record key (else the global row index).
    """
    import io
    import tarfile

    table = _as_table(data)
    os.makedirs(out_dir, exist_ok=True)
    text_columns = text_columns or []
    binary_columns = binary_columns or {}
    special = set(text_columns) | set(binary_columns)
    meta_cols = [c for c in table.column_names if c not in special and c != key_column]

    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    keys = cols[key_column] if key_column else list(range(table.num_rows))

    shards = []
    for i, (start, length) in enumerate(_shard_row_ranges(table.num_rows, rows_per_shard)):
        path = os.path.join(out_dir, f"{prefix}-{i:05d}.tar")
        with tarfile.open(path, "w") as tar:
            for r in range(start, start + length):
                key = str(keys[r])

                def _add(name: str, payload: bytes):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))

                for tc in text_columns:
                    val = cols[tc][r]
                    _add(f"{key}.{tc}.txt", ("" if val is None else str(val)).encode("utf-8"))
                for bc, ext in binary_columns.items():
                    val = cols[bc][r]
                    if val is not None:
                        _add(f"{key}.{ext}", bytes(val))
                if meta_cols:
                    meta = {c: cols[c][r] for c in meta_cols}
                    _add(f"{key}.json", json.dumps(meta, default=str).encode("utf-8"))
        shards.append({"path": path, "rows": length})
    return {"format": "webdataset", "shards": shards, "num_shards": len(shards), "rows": table.num_rows}


def write_mds(
    data: Any, out_dir: str, *, rows_per_shard: int = 50_000, prefix: str = "shard"
) -> dict:
    """Write a Mosaic StreamingDataset (MDS)-style layout: Parquet shards plus an
    ``index.json`` listing shards + row counts + the schema, so a streaming
    loader can index into it. (Uses Parquet shard payloads for portability; the
    index is the MDS-style manifest a StreamingDataset reader consumes.)
    """
    import pyarrow.parquet as pq

    table = _as_table(data)
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for i, (start, length) in enumerate(_shard_row_ranges(table.num_rows, rows_per_shard)):
        fname = f"{prefix}-{i:05d}.parquet"
        pq.write_table(table.slice(start, length), os.path.join(out_dir, fname))
        shards.append({"file": fname, "rows": length})
    index = {
        "version": 1,
        "format": "mds-parquet",
        "columns": {f.name: str(f.type) for f in table.schema},
        "shards": shards,
        "num_shards": len(shards),
        "rows": table.num_rows,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    return {"format": "mds", "index": os.path.join(out_dir, "index.json"), **index}
