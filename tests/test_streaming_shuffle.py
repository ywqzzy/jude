"""Distributed streaming shuffle (#40): worker-side pipelined hash-exchange.
Each partition is bucketized on its worker; per-bucket shards flow
worker -> object store -> reducer directly (no driver materialization), so map
and reduce overlap. Correctness is checked vs a single-node join. Ray-gated."""
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


def test_streaming_shuffle_join_matches_single_node():
    c = jude.connect()
    left = c.sql("SELECT range AS k, range*10 AS lv FROM range(200)").repartition(4)
    right = c.sql("SELECT range AS k, range*100 AS rv FROM range(0, 400, 2)").repartition(3)
    r = jude.runners.get_or_create_runner()
    assert type(r).__name__ == "RayRunner"
    out = r.distributed_join_streaming(left, right, ["k"], "inner")
    got = sorted((row["k"], row["lv"], row["rv"]) for row in out.to_pylist())

    # Single-node reference.
    ref = c.sql(
        "SELECT l.k, l.k*10 AS lv, l.k*100 AS rv FROM range(200) l(k) "
        "JOIN (SELECT range AS k FROM range(0,400,2)) r ON l.k = r.k"
    )
    want = sorted((x[0], x[1], x[2]) for x in ref.fetchall())
    assert got == want
    assert len(got) == 100  # even keys 0..198


def test_streaming_shuffle_empty_side():
    c = jude.connect()
    left = c.sql("SELECT range AS k FROM range(10)").repartition(2)
    right = c.sql("SELECT range AS k FROM range(100, 110)").repartition(2)  # disjoint keys
    r = jude.runners.get_or_create_runner()
    out = r.distributed_join_streaming(left, right, ["k"], "inner")
    assert out.num_rows == 0
