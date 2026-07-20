"""D3: RelationPipeline.from_datasource must consume the source LAZILY (bounded
input memory), not list() it into one monolithic table. Verifies the source is
never materialized and run_streaming() yields output shards without pulling the
whole source first.
"""

from __future__ import annotations

import pyarrow as pa

from jude.datasource import GeneratorSource
from jude.pipeline._multimodal import RelationPipeline

_SCHEMA = pa.schema([("x", pa.int64())])


def _counting_source(n_tasks=5, per_task=3, pulled=None):
    """A GeneratorSource whose tasks record every batch they emit into `pulled`,
    so a test can prove how much of the source was actually consumed."""
    def make(sid):
        def gen():
            for j in range(per_task):
                if pulled is not None:
                    pulled.append((sid, j))
                yield pa.record_batch({"x": [sid * 100 + j]})
        return gen
    return GeneratorSource(schema=_SCHEMA, task_fns=[make(s) for s in range(n_tasks)])


def test_from_datasource_not_materialized():
    p = RelationPipeline.from_datasource(_counting_source()).map_batches(lambda t: t)
    # the streaming source is a lazy thunk, NOT a materialized input table
    assert p._input_table is None
    assert p._input_shard_iter is not None


def test_run_streaming_is_lazy():
    pulled: list = []
    p = RelationPipeline.from_datasource(_counting_source(n_tasks=5, per_task=3, pulled=pulled))
    p = p.map_batches(lambda t: t)
    gen = p.run_streaming()
    first = next(gen)                       # pull exactly one output shard
    assert first.num_rows >= 1
    # a lazy pipeline pulls only what's needed for the first shard — NOT all 15
    assert len(pulled) < 15, f"pulled {len(pulled)} batches to yield the first shard"
    gen.close()


def test_streaming_run_matches_materialized():
    # full run() over a streaming source returns every row, once
    p = RelationPipeline.from_datasource(_counting_source(n_tasks=4, per_task=3))
    p = p.map_batches(lambda t: t)
    out = p.run()
    assert p._input_table is None           # still never materialized
    got = sorted(out.column("x").to_pylist())
    expected = sorted(s * 100 + j for s in range(4) for j in range(3))
    assert got == expected


def test_run_streaming_filter_and_explode():
    # a filter stage (drops rows) + explode-style map, fully streamed
    def keep_even(t: pa.Table) -> pa.Table:
        mask = [v % 2 == 0 for v in t.column("x").to_pylist()]
        return t.filter(pa.array(mask))

    p = RelationPipeline.from_datasource(_counting_source(n_tasks=6, per_task=4))
    p = p.map_batches(keep_even)
    rows = sorted(pa.concat_tables(list(p.run_streaming())).column("x").to_pylist())
    expected = sorted(v for s in range(6) for j in range(4) if (v := s * 100 + j) % 2 == 0)
    assert rows == expected
