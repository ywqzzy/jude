"""L0.4: transparent gz decompression for compressed csv/json shards."""

from __future__ import annotations

import gzip
import tempfile

from jude import storage


def test_read_csv_gz():
    d = tempfile.mkdtemp()
    with gzip.open(f"{d}/data.csv.gz", "wt") as f:
        f.write("id,g\n1,10\n2,20\n3,30\n")
    out = storage.read_arrow(f"file://{d}/data.csv.gz", fmt="csv")
    assert out.num_rows == 3
    assert out.column("id").to_pylist() == [1, 2, 3]


def test_read_jsonl_gz():
    d = tempfile.mkdtemp()
    with gzip.open(f"{d}/docs.jsonl.gz", "wt") as f:
        f.write('{"text":"hello"}\n{"text":"world"}\n')
    out = storage.read_arrow(f"file://{d}/docs.jsonl.gz", fmt="json")
    assert out.num_rows == 2
    assert out.column("text").to_pylist() == ["hello", "world"]
