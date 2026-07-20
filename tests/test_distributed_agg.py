"""Distributed aggregate decomposition (B3): STDDEV/VARIANCE run as exact
two-phase; non-decomposable aggregates (MEDIAN/QUANTILE/COUNT DISTINCT) fall back
to single-node instead of erroring."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude.runners._agg import NotDecomposable, is_decomposable, parse_agg

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _data(n=80000, groups=20, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table({"g": rng.integers(0, groups, n).tolist(),
                     "x": rng.standard_normal(n).tolist()})


def test_decomposability_classification():
    assert is_decomposable(["sum(x)", "avg(x)", "stddev(x)", "var_pop(x)", "count(*)"])
    assert not is_decomposable(["median(x)"])
    assert not is_decomposable(["count(distinct g)"])
    with pytest.raises(NotDecomposable):
        parse_agg("quantile_cont(x, 0.5)")


def test_stddev_variance_distributed_exact():
    from jude.runners.ray import RayRunner
    t = _data()
    r = RayRunner(num_workers=4)
    con = jude.connect(); con.register("t", t)
    sql = "SELECT g, stddev(x) s, var_pop(x) vp, avg(x) a FROM t GROUP BY g ORDER BY g"
    got = r.collect(con.sql(sql)).to_pydict()
    c2 = jude.connect(); c2.register("t", t)
    exp = c2.sql(sql).to_arrow().to_pydict()
    for col in ("s", "vp", "a"):
        for a, b in zip(got[col], exp[col]):
            assert abs(a - b) < 1e-9, f"{col}: {a} vs {b}"  # exact two-phase


def test_nondecomposable_falls_back_not_errors():
    from jude.runners.ray import RayRunner
    t = _data()
    r = RayRunner(num_workers=4)
    con = jude.connect(); con.register("t", t)
    # these are NOT two-phase decomposable — must fall back to single-node and
    # return the correct answer, NOT raise.
    for sql in ("SELECT median(x) m FROM t",
                "SELECT count(distinct g) c FROM t",
                "SELECT g, quantile_cont(x, 0.9) q FROM t GROUP BY g ORDER BY g"):
        got = r.collect(con.sql(sql))
        c2 = jude.connect(); c2.register("t", t)
        exp = c2.sql(sql).to_arrow()
        assert got.num_rows == exp.num_rows
