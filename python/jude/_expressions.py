"""DuckDB / Vane-compatible expression constructors.

Vane (via DuckDB-Python) exposes ``ColumnExpression``, ``ConstantExpression``,
``FunctionExpression``, ``CaseExpression``, ``CoalesceOperator``,
``StarExpression``, and ``Value`` for building relational-algebra expressions
programmatically, e.g.::

    rel.select(ColumnExpression("a"), ConstantExpression(5).cast(int))

jude already has a native ``Expression`` (SQL-fragment backed). These are thin
constructors that produce jude ``Expression`` objects, so they compose with the
relation API (``select`` / ``filter`` / ``order`` accept jude Expressions).
"""

from __future__ import annotations

from typing import Any

from .jude import Expression, col, lit, sql_expr

__all__ = [
    "ColumnExpression",
    "ConstantExpression",
    "FunctionExpression",
    "CaseExpression",
    "CoalesceOperator",
    "StarExpression",
    "DefaultExpression",
    "Value",
    "SQLExpression",
]


# ---------------------------------------------------------------------------
# Value + type helpers
# ---------------------------------------------------------------------------


class Value:
    """A typed constant, like DuckDB's ``Value(5, INTEGER)``.

    The type is advisory (used to CAST when rendered); the raw value drives the
    literal.
    """

    def __init__(self, value: Any, sqltype: Any = None):
        self.value = value
        self.sqltype = sqltype

    def to_expression(self) -> Expression:
        base = _const(self.value)
        if self.sqltype is not None:
            return base.cast(_type_to_sql(self.sqltype))
        return base


def _type_to_sql(t: Any) -> str:
    """Map a Python type or type name to a DuckDB SQL type string."""
    if isinstance(t, str):
        return t
    if t is int:
        return "BIGINT"
    if t is float:
        return "DOUBLE"
    if t is str:
        return "VARCHAR"
    if t is bool:
        return "BOOLEAN"
    if t is bytes:
        return "BLOB"
    # DuckDBPyType-like: str() yields the type name
    return str(t)


def _const(value: Any) -> Expression:
    if isinstance(value, Value):
        return value.to_expression()
    if isinstance(value, Expression):
        return value
    return lit(value)


# ---------------------------------------------------------------------------
# Expression constructors
# ---------------------------------------------------------------------------


def ColumnExpression(name: str) -> Expression:
    """Reference a column by name."""
    return col(name)


def ConstantExpression(value: Any) -> Expression:
    """A constant literal (or a typed Value)."""
    return _const(value)


def StarExpression() -> Expression:
    """``*`` — all columns."""
    return sql_expr("*")


def DefaultExpression() -> Expression:
    return sql_expr("DEFAULT")


def SQLExpression(text: str) -> Expression:
    """A raw SQL fragment."""
    return sql_expr(text)


def FunctionExpression(name: str, *args: Any) -> Expression:
    """A scalar function call, e.g. ``FunctionExpression("upper", col("s"))``."""
    rendered = ", ".join(_to_sql(a) for a in args)
    return sql_expr(f"{name}({rendered})")


def CoalesceOperator(*args: Any) -> Expression:
    """``COALESCE(a, b, ...)``. Requires at least one argument."""
    if not args:
        raise ValueError("Please provide at least one argument to COALESCE")
    rendered = ", ".join(_to_sql(a) for a in args)
    return sql_expr(f"COALESCE({rendered})")


class CaseExpression:
    """A CASE WHEN builder: ``CaseExpression(cond, val).when(c2, v2).otherwise(v)``."""

    def __init__(self, condition: Any, value: Any):
        self._whens: list[tuple[str, str]] = [(_to_sql(condition), _to_sql(value))]
        self._else: str | None = None

    def when(self, condition: Any, value: Any) -> "CaseExpression":
        self._whens.append((_to_sql(condition), _to_sql(value)))
        return self

    def otherwise(self, value: Any) -> Expression:
        self._else = _to_sql(value)
        return self._build()

    def _build(self) -> Expression:
        parts = " ".join(f"WHEN {c} THEN {v}" for c, v in self._whens)
        tail = f" ELSE {self._else}" if self._else is not None else ""
        return sql_expr(f"CASE {parts}{tail} END")

    def to_sql(self) -> str:
        return self._build().to_sql()

    def __str__(self) -> str:
        return self.to_sql()


def _to_sql(x: Any) -> str:
    if isinstance(x, Expression):
        return x.to_sql()
    if isinstance(x, CaseExpression):
        return x.to_sql()
    if isinstance(x, Value):
        return x.to_expression().to_sql()
    return _const(x).to_sql()
