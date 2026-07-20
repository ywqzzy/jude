"""Lazy-output boundary pipelining (#39): a UDF/kernel boundary's output is
lowered to a re-scannable jude_scan() table function (no temp-table copy) that
downstream SQL reads batch-by-batch. Verifies correctness + re-scannability."""
import pyarrow.compute as pc

import jude


def _mapped(c, n=1000):
    rel = c.sql(f"SELECT range AS i FROM range({n})")
    return rel.map_batches(lambda b: b.set_column(0, "i", pc.add(b.column("i"), 1)),
                           execution_backend="in_process")


def test_boundary_output_composes_with_sql():
    c = jude.connect()
    out = _mapped(c).filter("i % 2 = 0").aggregate("count(*), sum(i)")
    assert out.fetchall() == [(500, sum(range(2, 1001, 2)))]


def test_boundary_rescanned_via_union():
    # UNION ALL references the boundary output twice in ONE query -> the
    # jude_scan source is scanned twice; each scan re-inits its cursor.
    c = jude.connect()
    m = _mapped(c, 10)
    doubled = m.union_all(m)
    assert len(doubled.fetchall()) == 20
    assert doubled.aggregate("sum(i)").fetchone()[0] == 2 * sum(range(1, 11))


def test_aggregate_udf_output_composes():
    c = jude.connect()
    c.execute("create table s(g int, v int)")
    c.execute("insert into s values (1,10),(1,20),(2,5),(3,100)")
    agg = c.table("s").aggregate_udf(lambda t: sum(t.column("v").to_pylist()), ["v"], group_by=["g"])
    assert sorted(agg.filter("result >= 30").fetchall()) == [(1, 30), (3, 100)]
