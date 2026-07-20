"""B2: distributed final merge for GROUP BY. A high-cardinality GROUP BY must not
funnel every group onto one reducer — the partials are shuffled by group key and
merged per bucket. Results must still match a single-node aggregate exactly.
"""

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


def _data(n=20000, groups=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table({"g": rng.integers(0, groups, n).tolist(),
                     "x": rng.standard_normal(n).tolist()})


def _cmp(sql, t):
    from jude.runners.ray import RayRunner
    r = RayRunner(num_workers=4)
    con = jude.connect(); con.register("t", t)
    got = r.collect(con.sql(sql)).to_pydict()
    c2 = jude.connect(); c2.register("t", t)
    exp = c2.sql(sql).to_arrow().to_pydict()
    return got, exp


def test_high_cardinality_groupby_matches_single_node():
    t = _data()
    got, exp = _cmp("SELECT g, sum(x) s, count(*) c, avg(x) a FROM t GROUP BY g", t)
    # same set of groups, same aggregates per group
    g_got = dict(zip(got["g"], zip(got["s"], got["c"], got["a"])))
    g_exp = dict(zip(exp["g"], zip(exp["s"], exp["c"], exp["a"])))
    assert set(g_got) == set(g_exp)
    for g in g_exp:
        s0, c0, a0 = g_got[g]
        s1, c1, a1 = g_exp[g]
        assert c0 == c1
        assert abs(s0 - s1) < 1e-9 and abs(a0 - a1) < 1e-9


def test_distributed_aggregate_distributes_when_grouped():
    # directly exercise distributed_aggregate with group_keys (distributed merge)
    from jude.runners._agg import build_two_phase
    from jude.runners.ray import RayRunner

    t = _data(n=8000, groups=1000)
    partial_sql, final_sql = build_two_phase(["g"], ["sum(x)", "count(*)"])
    r = RayRunner(num_workers=4)
    con = jude.connect()
    out = r.distributed_aggregate(con.from_arrow(t), partial_sql, final_sql, ["g"])
    # every group present exactly once (per-bucket merge, then concat)
    gs = out.column("g").to_pylist()
    assert len(gs) == len(set(gs))
    c2 = jude.connect(); c2.register("t", t)
    exp = c2.sql("SELECT count(DISTINCT g) FROM t").to_arrow().column(0)[0].as_py()
    assert len(gs) == exp


def test_stddev_high_cardinality_exact():
    t = _data(n=12000, groups=800)
    got, exp = _cmp("SELECT g, stddev(x) s FROM t GROUP BY g", t)
    m_got = dict(zip(got["g"], got["s"]))
    m_exp = dict(zip(exp["g"], exp["s"]))
    assert set(m_got) == set(m_exp)
    for g in m_exp:
        a, b = m_got[g], m_exp[g]
        if a is None:
            assert b is None
        else:
            assert abs(a - b) < 1e-6
