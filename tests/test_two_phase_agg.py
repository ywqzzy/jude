"""Two-phase aggregate decomposition (_agg): the partial->merge SQL must be
EXACT vs a single-node aggregate for decomposable aggregates, and correctly
classify non-decomposable ones. Executes the generated SQL directly (no Ray) so
it's fast and deterministic — verifying the decomposition math itself.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude.runners._agg import NotDecomposable, build_two_phase, is_decomposable, parse_agg


def _data(n=4000, groups=7, seed=0):
    rng = np.random.default_rng(seed)
    return pa.table({"g": rng.integers(0, groups, n).tolist(),
                     "x": rng.standard_normal(n).tolist()})


def _two_phase_result(t, group_by, aggs, nparts=3):
    """Run partial per partition -> union -> final, mirroring the distributed
    executor but on one machine (so we test the SQL, not Ray)."""
    partial_sql, final_sql = build_two_phase(group_by, aggs)
    n = t.num_rows
    step = (n + nparts - 1) // nparts
    partials = []
    for i in range(0, n, step):
        part = t.slice(i, step)
        con = jude.connect()
        con.register("part", part)
        partials.append(con.sql(partial_sql).to_arrow())
    merged = pa.concat_tables(partials)
    con = jude.connect()
    con.register("partials", merged)
    return con.sql(final_sql).to_arrow()


def _single(t, sql):
    con = jude.connect()
    con.register("part", t)
    return con.sql(sql).to_arrow()


# --- classification ----------------------------------------------------------

def test_parse_agg_canonicalizes_aliases():
    assert parse_agg("stddev(x)").func == "STDDEV_SAMP"
    assert parse_agg("variance(x)").func == "VAR_SAMP"
    assert parse_agg("COUNT(*)").func == "COUNT"


def test_is_decomposable_classification():
    assert is_decomposable(["sum(x)", "avg(x)", "min(x)", "max(x)", "count(*)",
                            "stddev(x)", "var_pop(x)"])
    assert not is_decomposable(["median(x)"])
    assert not is_decomposable(["count(distinct g)"])
    with pytest.raises(NotDecomposable):
        parse_agg("quantile_cont(x, 0.5)")
    with pytest.raises(NotDecomposable):
        parse_agg("string_agg(x, ',')")


# --- exactness of the decomposition ------------------------------------------

@pytest.mark.parametrize("agg", ["sum(x)", "avg(x)", "min(x)", "max(x)",
                                 "stddev(x)", "stddev_pop(x)", "var_samp(x)", "var_pop(x)"])
def test_two_phase_exact_grouped(agg):
    t = _data()
    got = _two_phase_result(t, ["g"], [agg]).to_pydict()
    exp = _single(t, f"SELECT g, {agg} AS a FROM part GROUP BY g ORDER BY g").to_pydict()
    # align by group key
    got_map = dict(zip(got["g"], got[list(k for k in got if k != "g")[0]]))
    for g, a in zip(exp["g"], exp["a"]):
        b = got_map[g]
        if a is None:
            assert b is None
        else:
            assert abs(a - b) < 1e-9, f"{agg} group {g}: {a} vs {b}"


def test_two_phase_global_no_groupby():
    t = _data()
    got = _two_phase_result(t, [], ["sum(x)", "count(*)"]).to_pydict()
    exp = _single(t, "SELECT sum(x) s, count(*) c FROM part").to_pydict()
    assert abs(got["sum_x"][0] - exp["s"][0]) < 1e-9
    assert got["count__"][0] == exp["c"][0]


def test_count_star_sums_partials():
    t = _data(n=1000)
    got = _two_phase_result(t, ["g"], ["count(*)"]).to_pydict()
    total = sum(got["count__"])
    assert total == 1000  # every row counted exactly once across partitions
