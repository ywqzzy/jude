"""L0.3: streaming write sink — write each output shard as produced (bounded
memory end to end)."""

from __future__ import annotations

import glob
import tempfile

import pyarrow as pa
import pytest

from jude.datasource import GeneratorSource
from jude.pipeline._multimodal import RelationPipeline

_SCHEMA = pa.schema([("x", pa.int64())])


def _source(n_tasks=5, per_task=4):
    def make(sid):
        def gen():
            for j in range(per_task):
                yield pa.record_batch({"x": [sid * 100 + j]})
        return gen
    return GeneratorSource(schema=_SCHEMA, task_fns=[make(s) for s in range(n_tasks)])


def test_write_streaming_parquet_shards():
    d = tempfile.mkdtemp() + "/out"
    p = RelationPipeline.from_datasource(_source()).map_batches(lambda t: t)
    manifest = p.write_streaming(f"file://{d}", fmt="parquet")
    assert manifest["shards"] >= 1
    assert manifest["rows"] == 20                       # 5 tasks * 4 rows
    # every input row landed in some shard file, exactly once
    import pyarrow.parquet as pq
    files = sorted(glob.glob(d + "/part-*.parquet"))
    allx = []
    for f in files:
        allx.extend(pq.read_table(f).column("x").to_pylist())
    assert sorted(allx) == sorted(s * 100 + j for s in range(5) for j in range(4))


def test_write_streaming_lance():
    lance = pytest.importorskip("lance")
    import jude

    path = tempfile.mkdtemp() + "/out.lance"
    p = RelationPipeline.from_datasource(_source(3, 3)).map_batches(lambda t: t)
    manifest = p.write_streaming(path, fmt="lance")
    assert manifest["rows"] == 9
    back = jude._lance.read_table(path)
    assert back.num_rows == 9                            # all shards appended into one dataset
