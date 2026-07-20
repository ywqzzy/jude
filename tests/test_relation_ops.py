"""Relation-algebra correctness tests for jude.

These assert on the *results* of filter/project/aggregate/join/order/distinct/
set-ops, which used to be no-op stubs. Cases are adapted from Vane's
tests/fast/test_expression.py and the DuckDB relational API test shape
(VALUES-based fixtures, tuple results).
"""

import pytest

import jude
from jude import col, lit


@pytest.fixture
def filter_rel():
    con = jude.connect()
    rel = con.sql(
        """
        select * from (VALUES
            (1, 'a'),
            (2, 'b'),
            (1, 'b'),
            (3, 'c'),
            (4, 'a')
        ) tbl(a, b)
        """
    )
    return rel


class TestFilter:
    def test_filter_string(self, filter_rel):
        assert filter_rel.filter("a > 1").num_rows == 3

    def test_filter_expression(self, filter_rel):
        assert filter_rel.filter(col("a") > lit(1)).num_rows == 3

    def test_filter_and(self, filter_rel):
        r = filter_rel.filter((col("a") > lit(1)) & (col("a") < lit(4)))
        assert r.num_rows == 2

    def test_filter_string_eq(self, filter_rel):
        assert filter_rel.filter("b = 'a'").num_rows == 2

    def test_filter_chained(self, filter_rel):
        r = filter_rel.filter("a >= 1").filter("a <= 2")
        assert r.num_rows == 3  # rows with a in {1,2}: (1,a),(2,b),(1,b)


class TestProject:
    def test_project_alias(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
        r = rel.project(["a * 10 AS x"])
        assert r.columns == ["x"]
        assert r.fetchall() == [(10,)]

    def test_project_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT 5 AS a")
        r = rel.select((col("a") + lit(1)).alias("b"))
        assert r.fetchall() == [(6,)]

    def test_project_multiple(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b")
        r = rel.select(["a", "b"])
        assert r.columns == ["a", "b"]
        assert r.fetchall() == [(1, 2)]


class TestAggregate:
    def test_count_star(self, filter_rel):
        r = filter_rel.aggregate("COUNT(*) AS c")
        assert r.fetchall() == [(5,)]

    def test_group_by(self, filter_rel):
        r = filter_rel.aggregate("COUNT(*) AS c", "b")
        rows = dict(r.fetchall())
        assert rows == {"a": 2, "b": 2, "c": 1}

    def test_sum_grouped(self, filter_rel):
        r = filter_rel.aggregate("SUM(a) AS s", "b")
        rows = dict(r.fetchall())
        assert rows == {"a": 5, "b": 3, "c": 3}

    def test_agg_shortcut_sum(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        r = con.sql("SELECT * FROM t").sum("n")
        assert r.fetchall() == [(45,)]


class TestJoin:
    def test_inner_join(self):
        con = jude.connect()
        a = con.sql("SELECT * FROM (VALUES (1,'x'),(2,'y')) t(k,v)")
        b = con.sql("SELECT * FROM (VALUES (1,'p'),(3,'q')) t(k,w)")
        r = a.join(b, "lhs.k = rhs.k")
        assert r.num_rows == 1
        assert r.fetchall() == [(1, "x", 1, "p")]

    def test_left_join(self):
        con = jude.connect()
        a = con.sql("SELECT * FROM (VALUES (1),(2)) t(k)")
        b = con.sql("SELECT * FROM (VALUES (1)) t(k)")
        r = a.join(b, "lhs.k = rhs.k", how="left")
        assert r.num_rows == 2

    def test_cross_join(self):
        con = jude.connect()
        a = con.sql("SELECT * FROM (VALUES (1),(2)) t(k)")
        b = con.sql("SELECT * FROM (VALUES (10),(20),(30)) t(v)")
        assert a.cross(b).num_rows == 6


class TestSetOps:
    def test_union_all(self):
        con = jude.connect()
        a = con.sql("SELECT 1 AS v")
        b = con.sql("SELECT 1 AS v")
        assert a.union(b).num_rows == 2

    def test_distinct(self):
        con = jude.connect()
        rel = con.sql("SELECT * FROM (VALUES (1),(1),(2),(3),(3)) t(v)")
        assert rel.distinct().num_rows == 3

    def test_intersect(self):
        con = jude.connect()
        a = con.sql("SELECT * FROM (VALUES (1),(2),(3)) t(v)")
        b = con.sql("SELECT * FROM (VALUES (2),(3),(4)) t(v)")
        assert a.intersect(b).num_rows == 2

    def test_except(self):
        con = jude.connect()
        a = con.sql("SELECT * FROM (VALUES (1),(2),(3)) t(v)")
        b = con.sql("SELECT * FROM (VALUES (2),(3)) t(v)")
        assert a.except_(b).fetchall() == [(1,)]


class TestOrderLimit:
    def test_order_desc(self):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(5) t(n)")
        assert rel.order("n DESC").limit(1).fetchall() == [(4,)]

    def test_order_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(5) t(n)")
        assert rel.order(col("n").desc()).limit(1).fetchall() == [(4,)]

    def test_limit_offset(self):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(100) t(n)")
        r = rel.order("n").limit(5, 10)
        assert r.fetchall() == [(10,), (11,), (12,), (13,), (14,)]


class TestLazyChaining:
    def test_full_pipeline(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        r = (
            con.sql("SELECT * FROM t")
            .filter("n >= 10")
            .filter("n < 20")
            .project(["n", "n * n AS sq"])
            .order("n DESC")
        )
        assert r.num_rows == 10
        assert r.limit(1).fetchall() == [(19, 361)]

    def test_zero_row_schema(self):
        # Schema is available even for empty results (read from Arrow stream).
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2.5 AS b WHERE FALSE")
        assert rel.columns == ["a", "b"]
        assert rel.num_rows == 0


class TestArrowInterchange:
    def test_from_arrow_roundtrip(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        src = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        rel = con.from_arrow(src)
        assert rel.num_rows == 3
        assert rel.columns == ["x", "y"]
        assert rel.filter("y = 'b'").fetchall() == [(2, "b")]

    def test_to_arrow(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n, CAST(n AS VARCHAR) AS s FROM range(4) t(n)")
        tbl = con.sql("SELECT * FROM t").to_arrow()
        assert tbl.num_rows == 4
        assert tbl.column_names == ["n", "s"]

    def test_register(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        src = pa.table({"v": [10, 20, 30]})
        con.register("myview", src)
        assert con.sql("SELECT SUM(v) AS s FROM myview").fetchall() == [(60,)]

    def test_map_batches_then_chain(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        src = pa.table({"x": [1, 2, 3, 4]})
        rel = con.from_arrow(src)
        mapped = rel.map_batches(lambda b: b)  # identity
        assert mapped.filter("x > 2").num_rows == 2


class TestTypes:
    """Value extraction must cover Decimal, temporal, and nested types
    (these used to silently return None)."""

    def test_sum_returns_decimal(self):
        import decimal

        con = jude.connect()
        r = con.sql("SELECT SUM(n) AS s FROM range(10) t(n)")
        assert r.fetchall() == [(decimal.Decimal(45),)]

    def test_decimal_scale_applied(self):
        import decimal

        con = jude.connect()
        r = con.sql("SELECT CAST(3.14 AS DECIMAL(5,2)) AS d")
        assert r.fetchall() == [(decimal.Decimal("3.14"),)]

    def test_date(self):
        import datetime

        con = jude.connect()
        r = con.sql("SELECT DATE '2024-03-15' AS d")
        assert r.fetchall() == [(datetime.date(2024, 3, 15),)]

    def test_timestamp_naive(self):
        import datetime

        con = jude.connect()
        r = con.sql("SELECT TIMESTAMP '2024-01-01 12:30:45.123456' AS t")
        val = r.fetchall()[0][0]
        assert val == datetime.datetime(2024, 1, 1, 12, 30, 45, 123456)
        assert val.tzinfo is None

    def test_list(self):
        con = jude.connect()
        r = con.sql("SELECT [1, 2, 3] AS l")
        assert r.fetchall() == [([1, 2, 3],)]


class TestIO:
    def test_to_parquet_roundtrip(self, tmp_path):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        out = str(tmp_path / "out.parquet")
        con.sql("SELECT * FROM t").to_parquet(out)
        assert con.sql(f"SELECT COUNT(*) c FROM read_parquet('{out}')").fetchall() == [(10,)]

    def test_to_table(self):
        con = jude.connect()
        con.sql("SELECT 1 AS a UNION ALL SELECT 2").to_table("newtab")
        assert con.sql("SELECT COUNT(*) c FROM newtab").fetchall() == [(2,)]

    def test_create_view(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(5) t(n)")
        con.sql("SELECT n FROM t WHERE n > 2").to_view("v")
        assert con.sql("SELECT COUNT(*) c FROM v").fetchall() == [(2,)]
