"""Expression-class tests — adapted from Vane's tests/fast/test_expression.py.

jude exposes DuckDB/Vane-compatible ColumnExpression / ConstantExpression /
FunctionExpression / CaseExpression / CoalesceOperator / StarExpression / Value.
"""

import jude
from jude import (
    CaseExpression,
    CoalesceOperator,
    ColumnExpression,
    ConstantExpression,
    FunctionExpression,
    StarExpression,
    Value,
)


class TestExpressionClasses:
    def test_constant_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
        constant = ConstantExpression(Value(5, "INTEGER"))
        assert rel.select(constant).fetchall() == [(5,)]

    def test_column_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
        assert rel.select(ColumnExpression("a")).fetchall() == [(1,)]

    def test_function_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT 'hello' AS s")
        assert rel.select(FunctionExpression("upper", ColumnExpression("s"))).fetchall() == [("HELLO",)]

    def test_coalesce_first_non_null(self):
        con = jude.connect()
        rel = con.sql("SELECT 'unused'")
        assert rel.select(CoalesceOperator(ConstantExpression(None), ConstantExpression(42))).fetchall() == [(42,)]

    def test_coalesce_all_null(self):
        con = jude.connect()
        rel = con.sql("SELECT 'unused'")
        assert rel.select(
            CoalesceOperator(ConstantExpression(None), ConstantExpression(None))
        ).fetchone() == (None,)

    def test_coalesce_requires_argument(self):
        import pytest

        with pytest.raises(ValueError):
            CoalesceOperator()

    def test_coalesce_over_column(self):
        con = jude.connect()
        con.execute(
            "CREATE TABLE exprtest(a INTEGER, b INTEGER); "
            "INSERT INTO exprtest VALUES (42, 10), (43, 100), (NULL, 1), (45, 0)"
        )
        rel = con.table("exprtest")
        res = rel.select(CoalesceOperator(ColumnExpression("a"))).fetchall()
        assert res == [(42,), (43,), (None,), (45,)]

    def test_case_expression(self):
        con = jude.connect()
        tbl = con.sql("SELECT * FROM range(5) t(n)")
        case = CaseExpression(jude.col("n") > jude.lit(2), jude.lit("big")).otherwise(jude.lit("small"))
        assert tbl.select(case.alias("sz")).fetchall() == [
            ("small",),
            ("small",),
            ("small",),
            ("big",),
            ("big",),
        ]

    def test_case_multiple_when(self):
        con = jude.connect()
        tbl = con.sql("SELECT * FROM range(3) t(n)")
        case = (
            CaseExpression(jude.col("n") == jude.lit(0), jude.lit("zero"))
            .when(jude.col("n") == jude.lit(1), jude.lit("one"))
            .otherwise(jude.lit("many"))
        )
        assert tbl.select(case.alias("w")).fetchall() == [("zero",), ("one",), ("many",)]

    def test_value_typed_cast(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a")
        # Value with a float type casts the constant.
        out = rel.select(ConstantExpression(Value(3, "DOUBLE")).alias("v")).fetchone()
        assert out[0] == 3.0

    def test_star_expression(self):
        con = jude.connect()
        rel = con.sql("SELECT 1 AS a, 2 AS b")
        assert rel.select(StarExpression()).fetchall() == [(1, 2)]
