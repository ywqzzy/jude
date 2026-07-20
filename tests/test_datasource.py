"""Streaming DataSource API: bounded-memory generator-backed scan + distributed."""

from __future__ import annotations

import pyarrow as pa
import pytest

import jude
from jude import datasource as ds


SCHEMA = pa.schema([("x", pa.int64()), ("y", pa.string())])


# --- module-level task fns (picklable for the distributed path) --------------
def _shard(start, n):
    def gen():
        # emit in small chunks to prove streaming (bounded memory)
        for i in range(start, start + n, 2):
            hi = min(i + 2, start + n)
            xs = list(range(i, hi))
            yield pa.record_batch({"x": xs, "y": [f"v{v}" for v in xs]}, schema=SCHEMA)
    return gen


class RangeSource(ds.DataSource):
    """N shards, each yielding `per` rows in 2-row chunks."""

    def __init__(self, shards: int, per: int):
        self.shards = shards
        self.per = per

    def schema(self):
        return SCHEMA

    def tasks(self):
        return [ds._FnTask(_shard(s * self.per, self.per)) for s in range(self.shards)]


def test_read_stream_is_bounded_and_ordered():
    src = RangeSource(shards=2, per=6)
    batches = list(ds.read_stream(src))
    # every batch is small (<=2 rows) — proves chunked streaming
    assert all(b.num_rows <= 2 for b in batches)
    tbl = pa.Table.from_batches(batches, schema=SCHEMA)
    assert tbl.num_rows == 12
    assert tbl.column("x").to_pylist() == list(range(12))


def test_read_stream_rechunk():
    src = RangeSource(shards=1, per=10)
    batches = list(ds.read_stream(src, batch_rows=3))
    assert all(b.num_rows <= 3 for b in batches)
    assert sum(b.num_rows for b in batches) == 10


def test_read_local_relation():
    src = RangeSource(shards=3, per=4)
    rel = ds.read(src)
    assert rel.num_rows == 12
    got = rel.to_arrow().column("x").to_pylist()
    assert sorted(got) == list(range(12))


def test_empty_source_keeps_schema():
    src = RangeSource(shards=0, per=0)
    rel = ds.read(src)
    assert rel.num_rows == 0
    assert set(rel.to_arrow().schema.names) == {"x", "y"}


def test_generator_source_helper():
    def s0():
        yield pa.record_batch({"x": [1, 2], "y": ["a", "b"]}, schema=SCHEMA)

    def s1():
        yield {"x": [3], "y": ["c"]}  # dict chunk normalization

    src = ds.GeneratorSource(SCHEMA, [s0, s1])
    rel = ds.read(src)
    assert rel.num_rows == 3
    assert sorted(rel.to_arrow().column("y").to_pylist()) == ["a", "b", "c"]


def test_distributed_read_matches_local():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    src = RangeSource(shards=4, per=5)
    local = ds.read(src).to_arrow()
    dist = ds.read(RangeSource(shards=4, per=5), distributed=True).to_arrow()
    assert local.num_rows == dist.num_rows == 20
    assert sorted(dist.column("x").to_pylist()) == sorted(local.column("x").to_pylist())
