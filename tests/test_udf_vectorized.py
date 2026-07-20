"""Vectorized (arrow-native) scalar UDFs via create_function(type="arrow").

The vectorized path hands the UDF whole pyarrow columns and calls it once per
chunk (GIL acquired once per ~2048-row vector, not once per row), and inherits
full Arrow<->DuckDB type coverage — including types the row-by-row path only
string-coerced.
"""

import pytest

import jude

pc = pytest.importorskip("pyarrow.compute")


class TestVectorizedUDF:
    def test_arrow_udf_basic(self):
        con = jude.connect()

        def add10(col):
            return pc.add(col, 10)

        con.create_function("add10", add10, ["BIGINT"], "BIGINT", type="arrow")
        assert con.sql("SELECT add10(v) AS y FROM (VALUES (1),(2),(3)) t(v) ORDER BY y").fetchall() == [
            (11,),
            (12,),
            (13,),
        ]

    def test_vectorized_kwarg_alias(self):
        con = jude.connect()

        def dbl(col):
            return pc.multiply(col, 2.0)

        # DOUBLE is a type the row-by-row path only handled via the string
        # fallback; the arrow path gets it right.
        con.create_function("dbl", dbl, ["DOUBLE"], "DOUBLE", vectorized=True)
        assert con.sql("SELECT dbl(1.5) AS y").fetchall() == [(3.0,)]

    def test_null_passthrough(self):
        con = jude.connect()

        def inc(col):
            return pc.add(col, 1)

        con.create_function("inc", inc, ["BIGINT"], "BIGINT", type="arrow")
        assert con.sql("SELECT inc(v) AS y FROM (VALUES (1),(NULL),(3)) t(v)").fetchall() == [
            (1 + 1,),
            (None,),
            (3 + 1,),
        ]

    def test_two_arg_arrow_udf(self):
        con = jude.connect()

        def addcols(a, b):
            return pc.add(a, b)

        con.create_function("addcols", addcols, ["BIGINT", "BIGINT"], "BIGINT", type="arrow")
        assert con.sql("SELECT addcols(v, w) AS y FROM (VALUES (1,10),(2,20)) t(v,w) ORDER BY y").fetchall() == [
            (11,),
            (22,),
        ]

    def test_row_path_still_works(self):
        # The non-vectorized (row-by-row) path is unchanged for non-vectorizable fns.
        con = jude.connect()
        con.create_function("sq", lambda x: x * x, ["BIGINT"], "BIGINT")
        assert con.sql("SELECT sq(5) AS y").fetchall() == [(25,)]


class TestUDFExceptionHandling:
    """exception_handling='return_null' turns a throwing row/chunk into SQL NULL
    (a corrupt input doesn't abort the whole scan); the default re-raises."""

    def test_row_return_null(self):
        con = jude.connect()

        def recip(x):
            return 100 // x  # ZeroDivisionError on 0

        con.create_function("recip", recip, ["BIGINT"], "BIGINT", exception_handling="return_null")
        assert con.sql("SELECT recip(v) AS y FROM (VALUES (4),(0),(5)) t(v)").fetchall() == [
            (25,),
            (None,),
            (20,),
        ]

    def test_forward_default_raises(self):
        con = jude.connect()
        con.create_function("recip2", lambda x: 100 // x, ["BIGINT"], "BIGINT")
        with pytest.raises(Exception):
            con.sql("SELECT recip2(0)").fetchall()

    def test_arrow_return_null(self):
        con = jude.connect()

        def bad(col):
            raise ValueError("boom")

        con.create_function("bad", bad, ["BIGINT"], "BIGINT", type="arrow", exception_handling="return_null")
        assert con.sql("SELECT bad(v) AS y FROM (VALUES (1),(2)) t(v)").fetchall() == [(None,), (None,)]


class TestUDFNullHandling:
    """null_handling='default' skips rows with any NULL argument (UDF not called,
    result NULL); 'special' (default) passes NULLs to the UDF as Python None."""

    def test_row_default_skips_null_args(self):
        con = jude.connect()
        seen = []

        def f(x):
            seen.append(x)  # must never be None under 'default'
            return x * 2

        con.create_function("f_nd", f, ["BIGINT"], "BIGINT", null_handling="default")
        assert con.sql("SELECT f_nd(v) AS y FROM (VALUES (2),(NULL),(5)) t(v)").fetchall() == [
            (4,),
            (None,),
            (10,),
        ]
        assert None not in seen  # the NULL row never reached the UDF

    def test_row_special_sees_null(self):
        con = jude.connect()

        def g(x):
            return -1 if x is None else x * 2

        con.create_function("g_ns", g, ["BIGINT"], "BIGINT")  # special (default)
        assert con.sql("SELECT g_ns(v) AS y FROM (VALUES (2),(NULL),(5)) t(v)").fetchall() == [
            (4,),
            (-1,),
            (10,),
        ]

    def test_arrow_default_nulls_out(self):
        con = jude.connect()

        def af(col):
            return pc.multiply(col, 2)

        con.create_function("af_nd", af, ["BIGINT"], "BIGINT", type="arrow", null_handling="default")
        assert con.sql("SELECT af_nd(v) AS y FROM (VALUES (2),(NULL),(5)) t(v)").fetchall() == [
            (4,),
            (None,),
            (10,),
        ]


class TestUDFSideEffects:
    """side_effects=True marks the UDF volatile so DuckDB re-evaluates it per row
    instead of folding repeated calls (needed for nondeterministic UDFs)."""

    def test_volatile_called_per_row(self):
        con = jude.connect()
        n = {"c": 0}

        def counter():
            n["c"] += 1
            return 42

        con.create_function("cnt", counter, [], "BIGINT", side_effects=True)
        rows = con.sql("SELECT cnt() AS y FROM range(5)").fetchall()
        assert len(rows) == 5
        assert all(r == (42,) for r in rows)
        assert n["c"] == 5  # called once per row, not folded to a single call


