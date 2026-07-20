"""Distributed operators beyond agg/join: sort, distinct, top-k — proving many
operators are distributable (local op per partition + merge/shuffle)."""
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


def test_distributed_sort():
    c = jude.connect()
    rel = c.sql("SELECT (1000-range) AS x FROM range(1000)").repartition(4)
    out = jude.runners.get_or_create_runner().distributed_sort(rel, ["x"])
    xs = out.column("x").to_pylist()
    assert xs == sorted(xs) and len(xs) == 1000


def test_distributed_distinct():
    c = jude.connect()
    rel = c.sql("SELECT range % 7 AS g FROM range(500)").repartition(4)
    out = jude.runners.get_or_create_runner().distributed_distinct(rel)
    assert sorted(out.column("g").to_pylist()) == list(range(7))


def test_distributed_top_k():
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(1000)").repartition(4)
    out = jude.runners.get_or_create_runner().distributed_top_k(rel, ["x DESC"], 5)
    assert sorted(out.column("x").to_pylist(), reverse=True) == [999, 998, 997, 996, 995]
