"""jude.sqltypes — DuckDB SQL type constants.

Mirrors ``duckdb.sqltypes``: each name is a jude type token usable wherever the
relation/UDF API takes a SQL type (e.g. ``map_batches(fn, schema={"x": BIGINT})``,
``tensor_type(FLOAT, shape)``). Values are the DuckDB SQL type strings.
"""

from __future__ import annotations


class SQLType(str):
    """A SQL type token — a str subclass so it works anywhere a type string does,
    but carries a distinct type for isinstance checks."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SQLType({str.__str__(self)!r})"


def _t(name: str) -> SQLType:
    return SQLType(name)


BOOLEAN = _t("BOOLEAN")
TINYINT = _t("TINYINT")
SMALLINT = _t("SMALLINT")
INTEGER = _t("INTEGER")
BIGINT = _t("BIGINT")
HUGEINT = _t("HUGEINT")
UTINYINT = _t("UTINYINT")
USMALLINT = _t("USMALLINT")
UINTEGER = _t("UINTEGER")
UBIGINT = _t("UBIGINT")
UHUGEINT = _t("UHUGEINT")
FLOAT = _t("FLOAT")
DOUBLE = _t("DOUBLE")
DATE = _t("DATE")
TIME = _t("TIME")
TIME_TZ = _t("TIME WITH TIME ZONE")
TIMESTAMP = _t("TIMESTAMP")
TIMESTAMP_MS = _t("TIMESTAMP_MS")
TIMESTAMP_NS = _t("TIMESTAMP_NS")
TIMESTAMP_S = _t("TIMESTAMP_S")
TIMESTAMP_TZ = _t("TIMESTAMP WITH TIME ZONE")
INTERVAL = _t("INTERVAL")
VARCHAR = _t("VARCHAR")
BLOB = _t("BLOB")
BIT = _t("BIT")
UUID = _t("UUID")
VARIANT = _t("VARIANT")
SQLNULL = _t("NULL")

__all__ = [
    "SQLType",
    "BOOLEAN", "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DATE", "TIME", "TIME_TZ", "TIMESTAMP", "TIMESTAMP_MS",
    "TIMESTAMP_NS", "TIMESTAMP_S", "TIMESTAMP_TZ", "INTERVAL", "VARCHAR",
    "BLOB", "BIT", "UUID", "VARIANT", "SQLNULL",
]
