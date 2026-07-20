"""L0.1: object-store IO via fsspec — tested with memory:// and file:// (the
same code path serves s3://MinIO once s3fs + endpoint_url are configured)."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

from jude import storage


def test_bytes_roundtrip_memory():
    storage.write_bytes("memory://d/hello.txt", b"hi there")
    assert storage.read_bytes("memory://d/hello.txt") == b"hi there"
    assert storage.exists("memory://d/hello.txt")
    assert not storage.exists("memory://d/nope.txt")


def test_parquet_roundtrip_memory():
    t = pa.table({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    storage.write_parquet(t, "memory://p/data.parquet")
    back = storage.read_arrow("memory://p/data.parquet", fmt="parquet")
    assert back.num_rows == 3
    assert back.column("text").to_pylist() == ["a", "b", "c"]


def test_parquet_glob_and_projection():
    for i in range(3):
        storage.write_parquet(pa.table({"id": [i], "g": [i * 10]}),
                              f"memory://g/part{i}.parquet")
    out = storage.read_arrow("memory://g/*.parquet", fmt="parquet", columns=["id"])
    assert out.column_names == ["id"]
    assert set(out.column("id").to_pylist()) == {0, 1, 2}


def test_file_backend_roundtrip():
    d = tempfile.mkdtemp()
    t = pa.table({"x": [1, 2]})
    storage.write_parquet(t, f"file://{d}/x.parquet")
    assert storage.read_arrow(f"file://{d}/x.parquet").num_rows == 2


def test_glob_lists_qualified_paths():
    for i in range(2):
        storage.write_bytes(f"memory://gl/f{i}.bin", b"x")
    got = storage.glob("memory://gl/*.bin")
    assert len(got) == 2 and all(g.startswith("memory://") for g in got)


def test_read_missing_raises():
    with pytest.raises(FileNotFoundError):
        storage.read_arrow("memory://empty/*.parquet", fmt="parquet")
