"""B4: salted skew-join. A hot join key must not pile all its rows onto one
reducer bucket; the salted inner-join spreads the hot key's left rows across all
buckets and replicates its right rows — producing the SAME result as a single-
node join. (No spill; skew handling only.)"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import jude

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _runner(n=4):
    from jude.runners.ray import RayRunner
    return RayRunner(num_workers=n)


def _single_node_join(left, right, key, how="inner"):
    con = jude.connect()
    con.register("l", left)
    con.register("r", right)
    return con.sql(
        f"SELECT l.*, r.val AS rval FROM l JOIN r ON l.{key} = r.{key}"
    ).to_arrow()


def test_salted_inner_join_matches_single_node_under_skew():
    # 90% of left rows share ONE hot key; the rest are spread — classic skew.
    n = 4000
    rng = np.random.default_rng(0)
    keys = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 200, n))
    left = pa.table({"k": keys.tolist(), "lid": list(range(n))})
    # right: one row per distinct key (dimension table), incl the hot key 0
    rkeys = sorted(set(keys.tolist()))
    right = pa.table({"k": rkeys, "val": [k * 10 for k in rkeys]})

    r = _runner()
    con = jude.connect()
    # call the salted distributed_join API directly so the salted path runs
    dist = r.distributed_join(con.from_arrow(left), con.from_arrow(right), ["k"], how="inner")
    got = dist.to_pydict()

    exp = _single_node_join(left, right, "k")
    assert dist.num_rows == exp.num_rows
    # distributed_join keeps both sides' columns; join value column is "val"
    d = dict(zip(got["lid"], got["val"]))
    e = dict(zip(exp.column("lid").to_pylist(), exp.column("rval").to_pylist()))
    assert d == e


def test_salt_helper_spreads_hot_key_across_buckets():
    # unit-test the salting helper directly: the hot key's left rows land in
    # MULTIPLE buckets, and its right row is replicated to every bucket.
    n = 2000
    left = pa.table({"k": [0] * 1800 + list(range(1, 201)), "lid": list(range(n))})
    right = pa.table({"k": list(range(0, 201)), "val": list(range(0, 201))})
    r = _runner()
    con = jude.connect()
    b = 8
    res = r._salt_skewed_inner(con, left, right, "k", b)
    assert res is not None, "a 90%-hot key should trigger salting"
    lb, rb = res
    # the hot key (0) appears in more than one left bucket
    hot_buckets = sum(1 for t in lb if 0 in t.column("k").to_pylist())
    assert hot_buckets > 1
    # the hot key's right row is replicated to every bucket
    assert all(0 in t.column("k").to_pylist() for t in rb)
    # no rows lost: union of left buckets == original left row count
    assert sum(t.num_rows for t in lb) == n


def test_no_skew_no_salting():
    # a uniform join has no hot key -> helper returns None (plain path, no overhead)
    n = 2000
    rng = np.random.default_rng(1)
    left = pa.table({"k": rng.integers(0, 500, n).tolist(), "lid": list(range(n))})
    right = pa.table({"k": list(range(500)), "val": list(range(500))})
    r = _runner()
    con = jude.connect()
    assert r._salt_skewed_inner(con, left, right, "k", 8) is None
