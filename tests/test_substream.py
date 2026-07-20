"""Sub-batch streaming (③): row-wise ops stream batch-by-batch via Ray streaming
generators; streaming two-phase aggregation streams partials per batch."""
import os

import pytest

ray = pytest.importorskip("ray")
import jude


@pytest.fixture(scope="module", autouse=True)
def _ray():
    os.environ["JUDE_RUNNER"] = "ray"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    jude.runners._reset_runner()
    yield


def test_streaming_transform_filter_project():
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(1000)").repartition(4)
    r = jude.runners.get_or_create_runner()
    import pyarrow as pa
    out = pa.concat_tables(list(r.streaming_transform(rel, "SELECT x, x*2 AS y FROM part WHERE x % 2 = 0", batch_size=64)))
    rows = sorted((t[0], t[1]) for t in zip(out.column("x").to_pylist(), out.column("y").to_pylist()))
    assert rows == [(x, x * 2) for x in range(0, 1000, 2)]


def test_streaming_transform_is_incremental():
    # More than one batch is yielded (proves sub-batch streaming, not one blob).
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(500)").repartition(2)
    r = jude.runners.get_or_create_runner()
    batches = list(r.streaming_transform(rel, "SELECT * FROM part", batch_size=50))
    assert len(batches) > 1
    assert sum(b.num_rows for b in batches) == 500


def test_streaming_aggregate_sum():
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(1000)").repartition(4)
    r = jude.runners.get_or_create_runner()
    out = r.streaming_aggregate(
        rel,
        partial_sql="SELECT sum(x) AS s, count(*) AS c FROM part",
        final_sql="SELECT sum(s) AS total, sum(c) AS n FROM partials",
        batch_size=64,
    )
    assert out.column("total").to_pylist()[0] == sum(range(1000))
    assert out.column("n").to_pylist()[0] == 1000
