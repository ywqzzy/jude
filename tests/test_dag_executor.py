"""General streaming stage-DAG executor: nested shuffles run distributed and
match single-node ground truth. Gated on Ray being importable.

Plans are built through the *relational* API (.aggregate/.join/.order/...), which
produces the structured LogicalPlan the distributed executor decomposes. (A raw
con.sql("...") string is an opaque RawSql leaf — the executor treats it as one
scan stage, which is correct but not what these nested-shuffle tests exercise.)
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import jude

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _runner():
    from jude.runners.ray import RayRunner

    return RayRunner(num_workers=4)


def _table(con, name):
    return con.sql(f"SELECT * FROM {name}")


def _sorted_rows(t: pa.Table) -> list:
    d = t.to_pydict()
    cols = list(d.keys())
    rows = list(zip(*[d[c] for c in cols]))
    return sorted(rows, key=lambda r: tuple((x is None, x) for x in r))


def _assert_same(got: pa.Table, want: pa.Table):
    assert set(got.column_names) == set(want.column_names), (got.column_names, want.column_names)
    want2 = want.select(got.column_names)
    assert _sorted_rows(got) == _sorted_rows(want2)


def test_dist_step_nested_shuffle_detected():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2, 3], "v": [10, 20, 30, 40]}))
    rel = _table(con, "t").aggregate("SUM(v) AS s", "g").order("s")
    r = _runner()
    assert r._plan_is_nested_shuffle(rel) is True


def test_single_shuffle_not_nested():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2], "v": [1, 2, 3]}))
    rel = _table(con, "t").aggregate("SUM(v) AS s", "g")
    r = _runner()
    assert r._plan_is_nested_shuffle(rel) is False


def test_aggregate_then_order():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2, 3, 3, 3], "v": [10, 20, 30, 40, 50, 60]}))
    rel = _table(con, "t").aggregate("SUM(v) AS s", "g").order("s DESC")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_aggregate_then_join():
    con = jude.connect()
    con.register("sales", pa.table({"region": ["a", "a", "b", "c"], "amt": [10, 20, 30, 40]}))
    con.register("names", pa.table({"region": ["a", "b", "c"], "label": ["A", "B", "C"]}))
    agg = _table(con, "sales").aggregate("SUM(amt) AS total", "region")
    rel = agg.join(_table(con, "names"), "region")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_join_then_aggregate():
    con = jude.connect()
    con.register("orders", pa.table({"cust": [1, 1, 2, 2, 3], "amt": [5, 15, 25, 35, 45]}))
    con.register("cust", pa.table({"cust": [1, 2, 3], "tier": ["x", "y", "x"]}))
    joined = _table(con, "orders").join(_table(con, "cust"), "cust")
    rel = joined.aggregate("SUM(amt) AS spend", "tier")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_distinct_over_aggregate():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2, 2, 3], "v": [1, 1, 2, 2, 2]}))
    agg = _table(con, "t").aggregate("COUNT(*) AS c", "g")
    rel = agg.distinct()
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_filter_above_aggregate_pushable():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2, 3, 3], "v": [10, 20, 5, 40, 50]}))
    agg = _table(con, "t").aggregate("SUM(v) AS s", "g")
    rel = agg.filter("s > 15").order("g")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_collect_routes_nested_to_dag():
    con = jude.connect()
    con.register("t", pa.table({"g": [1, 1, 2, 3], "v": [10, 20, 30, 40]}))
    rel = _table(con, "t").aggregate("SUM(v) AS s", "g").order("s")
    r = _runner()
    _assert_same(r.collect(rel), rel.to_arrow())


def test_three_level_agg_join_order():
    con = jude.connect()
    con.register("sales", pa.table({"region": ["a", "a", "b", "c", "c"], "amt": [10, 20, 30, 40, 50]}))
    con.register("names", pa.table({"region": ["a", "b", "c"], "label": ["A", "B", "C"]}))
    agg = _table(con, "sales").aggregate("SUM(amt) AS total", "region")
    joined = agg.join(_table(con, "names"), "region")
    rel = joined.order("total DESC")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())


def test_union_then_aggregate():
    con = jude.connect()
    con.register("a", pa.table({"g": [1, 2], "v": [10, 20]}))
    con.register("b", pa.table({"g": [2, 3], "v": [30, 40]}))
    unioned = _table(con, "a").union_all(_table(con, "b"))
    rel = unioned.aggregate("SUM(v) AS s", "g")
    r = _runner()
    _assert_same(r.execute_dag(rel), rel.to_arrow())
