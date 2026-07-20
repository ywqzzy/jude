"""DuckDB/Vane compatibility tests — adapted from Vane's tests/fast.

Covers DBAPI execute()->fetch, transactions, create_function, and DuckDB-style
cursor semantics that jude now supports.
"""

import pytest

import jude


class TestConnectionTransaction:
    """Adapted from vane tests/fast/test_transaction.py."""

    def test_transaction(self):
        con = jude.connect()
        con.execute("create table t (i integer)")
        con.execute("insert into t values (1)")

        con.begin()
        con.execute("insert into t values (1)")
        assert con.execute("select count(*) from t").fetchone()[0] == 2
        con.rollback()
        assert con.execute("select count(*) from t").fetchone()[0] == 1
        con.begin()
        con.execute("insert into t values (1)")
        assert con.execute("select count(*) from t").fetchone()[0] == 2
        con.commit()
        assert con.execute("select count(*) from t").fetchone()[0] == 2


class TestDBAPIFetch:
    def test_execute_fetchone(self):
        con = jude.connect()
        assert con.execute("SELECT 42").fetchone() == (42,)

    def test_execute_fetchall(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(3) t(n)")
        assert con.execute("SELECT n FROM t ORDER BY n").fetchall() == [(0,), (1,), (2,)]

    def test_execute_fetchmany(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        rows = con.execute("SELECT n FROM t ORDER BY n").fetchmany(3)
        assert rows == [(0,), (1,), (2,)]

    def test_execute_with_params(self):
        con = jude.connect()
        con.execute("CREATE TABLE t(i INTEGER, name VARCHAR)")
        con.execute("INSERT INTO t VALUES (?, ?)", [1, "Alice"])
        assert con.execute("SELECT name FROM t WHERE i = ?", [1]).fetchone() == ("Alice",)

    def test_execute_fetchdf(self):
        pd = pytest.importorskip("pandas")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(5) t(n)")
        df = con.execute("SELECT * FROM t").fetchdf()
        assert len(df) == 5


class TestCreateFunction:
    """Adapted from DuckDB's create_function API used across vane tests."""

    def test_scalar_udf(self):
        con = jude.connect()
        con.create_function("add_one", lambda x: x + 1, parameters=["INTEGER"], return_type="INTEGER")
        assert con.execute("SELECT add_one(41)").fetchone() == (42,)

    def test_udf_on_table(self):
        con = jude.connect()
        con.execute("CREATE TABLE nums AS SELECT CAST(n AS INTEGER) n FROM range(5) t(n)")
        con.create_function("dbl", lambda x: x * 2, parameters=["INTEGER"], return_type="INTEGER")
        assert con.execute("SELECT dbl(n) FROM nums ORDER BY n").fetchall() == [(0,), (2,), (4,), (6,), (8,)]

    def test_remove_function_is_noop_safe(self):
        con = jude.connect()
        con.create_function("f", lambda x: x, parameters=["INTEGER"], return_type="INTEGER")
        con.remove_function("f")  # must not raise


class TestCursor:
    def test_cursor_shares_database(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        cur = con.cursor()
        assert cur.execute("SELECT COUNT(*) FROM t").fetchone() == (10,)

    def test_context_manager(self):
        con = jude.connect()
        con.execute("CREATE TABLE t(i INTEGER)")
        with con:
            con.execute("INSERT INTO t VALUES (1)")
        assert con.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)
