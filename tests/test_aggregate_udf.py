"""User-defined aggregate UDFs via group-apply (jude differentiator — neither
DuckDB-Python nor Vane exposes a Python aggregate UDF)."""

import statistics

import pytest

import jude


@pytest.fixture
def rel():
    c = jude.connect()
    c.execute("create table s(g int, v int)")
    c.execute("insert into s values (1,10),(1,20),(1,30),(2,5),(2,15)")
    return c.table("s")


def test_grouped_scalar_aggregate(rel):
    out = rel.aggregate_udf(lambda t: statistics.median(t.column("v").to_pylist()), ["v"], group_by=["g"], result_name="median")
    assert sorted(out.fetchall()) == [(1, 20.0), (2, 10.0)]


def test_global_aggregate(rel):
    out = rel.aggregate_udf(lambda t: statistics.median(t.column("v").to_pylist()), ["v"])
    assert out.fetchall() == [(15,)]


def test_dict_multi_output(rel):
    def stats(t):
        vals = t.column("v").to_pylist()
        return {"mn": min(vals), "mx": max(vals)}

    out = rel.aggregate_udf(stats, ["v"], group_by=["g"])
    assert sorted(out.fetchall()) == [(1, 10, 30), (2, 5, 15)]


def test_multi_column_input(rel):
    # fn sees a table of the requested columns for each group.
    def span(t):
        return t.column("v").to_pylist()[-1] - t.column("v").to_pylist()[0]

    out = rel.aggregate_udf(span, ["v"], group_by=["g"], result_name="span")
    assert sorted(out.fetchall()) == [(1, 20), (2, 10)]


def test_empty_columns_raises(rel):
    with pytest.raises(jude.InvalidInputException):
        rel.aggregate_udf(lambda t: 0, [])
