"""collect(): auto-distributing executor — inspect the stage DAG and route to
the right distributed strategy (sort / distinct / parallel scan)."""
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


def _rel(sql):
    # A self-contained relation (range source) so no cross-connection table is needed.
    return jude.connect().sql(sql)


def test_collect_routes_order():
    rel = _rel("SELECT range AS id, (1000-range) AS v FROM range(200)").order("v").repartition(4)
    out = jude.runners.get_or_create_runner().collect(rel)
    assert out.column("v").to_pylist() == sorted(out.column("v").to_pylist())
    assert out.num_rows == 200


def test_collect_routes_distinct():
    rel = _rel("SELECT range % 4 AS g FROM range(200)").distinct().repartition(4)
    out = jude.runners.get_or_create_runner().collect(rel)
    assert sorted(out.column("g").to_pylist()) == [0, 1, 2, 3]


def test_collect_parallel_scan_filter():
    rel = _rel("SELECT range AS id, range % 4 AS g FROM range(200)").filter("g = 1").repartition(4)
    out = jude.runners.get_or_create_runner().collect(rel)
    assert out.num_rows == 50


def test_collect_routes_aggregate():
    rel = _rel("SELECT range % 4 AS g, range AS v FROM range(400)").aggregate("sum(v) AS s, count(*) AS n", "g").repartition(4)
    out = jude.runners.get_or_create_runner().collect(rel)
    rows = sorted((r["g"], int(r["s"]), int(r["n"])) for r in out.to_pylist())
    ref = sorted((g, sum(v for v in range(400) if v % 4 == g), 100) for g in range(4))
    assert rows == ref


def test_collect_routes_join():
    c = jude.connect()
    left = c.sql("SELECT range AS k, range*10 AS lv FROM range(50)").repartition(3)
    right = c.sql("SELECT range AS k FROM range(0, 100, 2)").repartition(2)
    out = jude.runners.get_or_create_runner().collect(left.join(right, "k"))
    assert out.num_rows == 25  # even keys 0..48
