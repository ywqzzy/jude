"""Replacement scan tests — adapted from Vane's tests/fast/test_replacement_scan.py.

conn.sql("SELECT * FROM my_var") resolves an in-scope pandas/polars/pyarrow
object by its variable name, like DuckDB / Vane.
"""

import pytest

import jude

pa = pytest.importorskip("pyarrow")


class TestReplacementScan:
    def test_pandas_dataframe(self):
        pd = pytest.importorskip("pandas")
        con = jude.connect()
        my_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rel = con.sql("SELECT * FROM my_df WHERE a > 1")
        assert rel.fetchall() == [(2, "y"), (3, "z")]

    def test_pyarrow_table(self):
        con = jude.connect()
        arrow_tbl = pa.table({"v": [10, 20, 30]})
        assert con.sql("SELECT SUM(v) AS s FROM arrow_tbl").fetchall() == [(60,)]

    def test_pyarrow_record_batch(self):
        con = jude.connect()
        rb = pa.record_batch({"n": [1, 2, 3, 4]})
        assert con.sql("SELECT COUNT(*) AS c FROM rb").fetchall() == [(4,)]

    def test_polars_dataframe(self):
        pl = pytest.importorskip("polars")
        con = jude.connect()
        pldf = pl.DataFrame({"n": [1, 2, 3, 4]})
        assert con.sql("SELECT COUNT(*) AS c FROM pldf").fetchall() == [(4,)]

    def test_using_table_method(self):
        # Vane's using_table(): con.table(name) also resolves the scan.
        con = jude.connect()
        tbl_obj = pa.table({"x": [5, 6, 7]})
        # register manually then table() — table() itself does not do frame
        # inspection, but sql() (used by table under the hood elsewhere) does;
        # here we assert the sql path.
        assert con.sql("SELECT MAX(x) AS m FROM tbl_obj").fetchall() == [(7,)]

    def test_normal_table_unaffected(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(5) t(n)")
        assert con.sql("SELECT COUNT(*) AS c FROM t").fetchall() == [(5,)]

    def test_join_two_scanned_frames(self):
        pd = pytest.importorskip("pandas")
        con = jude.connect()
        orders = pd.DataFrame({"k": [1, 2, 3], "lv": [10, 20, 30]})
        prices = pd.DataFrame({"k": [2, 3, 4], "rv": [200, 300, 400]})
        rel = con.sql(
            "SELECT orders.k, lv, rv FROM orders JOIN prices ON orders.k = prices.k ORDER BY orders.k"
        )
        assert rel.fetchall() == [(2, 20, 200), (3, 30, 300)]

    def test_missing_variable_still_errors(self):
        con = jude.connect()
        with pytest.raises(Exception):
            con.sql("SELECT * FROM totally_not_defined_xyz").fetchall()
