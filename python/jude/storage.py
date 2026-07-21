"""jude.storage — object-store IO via fsspec (S3 / GCS / local / in-memory).

Curation data lives on object storage (S3/GCS/MinIO), not just local disk. This
is a thin fsspec-backed layer so every source/sink can take a URL:

    s3://bucket/data/*.parquet   gs://bucket/...   file:///abs/path   memory://tmp

Backends resolve through fsspec: ``file`` / ``memory`` are built in (used by the
tests, zero-dep); ``s3`` needs ``s3fs`` and MinIO/S3 creds via ``storage_options``
(``endpoint_url`` points at a local MinIO). Arrow IO bridges fsspec to pyarrow
via ``PyFileSystem(FSSpecHandler(...))`` so parquet/csv/json read straight off
the store.

    fs, path = resolve("s3://b/x.parquet", endpoint_url="http://localhost:9000",
                       key="minio", secret="minio123")
    tbl = read_arrow("memory://d/x.parquet", fmt="parquet")
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def resolve(url: str, **storage_options: Any):
    """Return (fsspec filesystem, path-within-fs) for ``url``. Extra kwargs are
    fsspec ``storage_options`` (e.g. S3 ``endpoint_url``/``key``/``secret`` for
    MinIO)."""
    import fsspec

    return fsspec.core.url_to_fs(url, **storage_options)


def _pa_fs(fs: Any):
    """Wrap an fsspec filesystem as a pyarrow filesystem for Arrow readers."""
    from pyarrow.fs import FSSpecHandler, PyFileSystem

    return PyFileSystem(FSSpecHandler(fs))


def write_bytes(url: str, data: bytes, **storage_options: Any) -> None:
    fs, path = resolve(url, **storage_options)
    parent = path.rsplit("/", 1)[0]
    if parent and parent != path:
        fs.makedirs(parent, exist_ok=True)
    fs.pipe_file(path, data)


def read_bytes(url: str, **storage_options: Any) -> bytes:
    fs, path = resolve(url, **storage_options)
    return fs.cat_file(path)


def exists(url: str, **storage_options: Any) -> bool:
    fs, path = resolve(url, **storage_options)
    return fs.exists(path)


def glob(url: str, **storage_options: Any) -> list[str]:
    """List files matching a glob URL (returns protocol-qualified paths)."""
    fs, path = resolve(url, **storage_options)
    proto = (url.split("://", 1)[0] + "://") if "://" in url else ""
    return [proto + p for p in fs.glob(path)]


def write_parquet(table: pa.Table, url: str, **storage_options: Any) -> dict:
    """Write an Arrow table as parquet to any fsspec URL (S3/MinIO/local/memory)."""
    import pyarrow.parquet as pq

    fs, path = resolve(url, **storage_options)
    parent = path.rsplit("/", 1)[0]
    if parent and parent != path:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as f:
        pq.write_table(table, f)
    return {"url": url, "rows": table.num_rows}


def read_arrow(url: str, *, fmt: str = "parquet", columns: Any = None,
               **storage_options: Any) -> pa.Table:
    """Read parquet/csv/json (or a glob of them) off any fsspec store into Arrow.
    For ``lance``, use ``jude._lance`` with the URL + storage_options directly."""
    import pyarrow.parquet as pq

    fs, path = resolve(url, **storage_options)
    paths = fs.glob(path) if any(c in path for c in "*?[") else [path]
    if not paths:
        raise FileNotFoundError(f"no files match {url!r}")
    pafs = _pa_fs(fs)
    tables = []
    for p in paths:
        if fmt == "parquet":
            tables.append(pq.read_table(p, columns=columns, filesystem=pafs))
        elif fmt in ("csv", "json"):
            import pyarrow.csv as pacsv
            import pyarrow.json as pajson

            # fsspec transparently decompresses .gz/.bz2/.zst by extension so
            # compressed shards (.jsonl.gz, .csv.gz — common in crawl dumps) read
            # directly.
            with fs.open(p, "rb", compression="infer") as f:
                t = pacsv.read_csv(f) if fmt == "csv" else pajson.read_json(f)
            tables.append(t.select(columns) if columns else t)
        else:
            raise ValueError(f"unsupported fmt {fmt!r}")
    out = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    return out.combine_chunks()
