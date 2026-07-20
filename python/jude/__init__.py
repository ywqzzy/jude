"""jude — A distributed, multimodal data engine built on DuckDB."""

import enum as _enum

from .jude import *  # noqa: F401,F403
from .jude import (  # noqa: F401
    Connection,
    Relation,
    Expression,
    Config,
    EnvRegistry,
    connect,
    sql,
    col,
    lit,
    sql_expr,
    configure,
    current_config,
    make_env,
    attach_function,
    detach_function,
)
from .expression_udf import func, cls  # noqa: F401
from .exceptions import *  # noqa: F401,F403
from ._expressions import (  # noqa: F401
    ColumnExpression,
    ConstantExpression,
    FunctionExpression,
    CaseExpression,
    CoalesceOperator,
    StarExpression,
    DefaultExpression,
    SQLExpression,
    Value,
)
from . import runners  # noqa: F401
from . import pipeline  # noqa: F401
from . import catalog  # noqa: F401
from . import observe  # noqa: F401
from . import datasource  # noqa: F401
from . import curate  # noqa: F401
from . import curate_mm  # noqa: F401
from . import curate_dist  # noqa: F401
from . import curate_flow  # noqa: F401
from . import cluster  # noqa: F401
from . import training_format  # noqa: F401
from . import vector  # noqa: F401
from .vector_search import VectorSearch  # noqa: F401
from . import lance  # noqa: F401
from . import structured  # noqa: F401
from . import types  # noqa: F401
from . import sqltypes  # noqa: F401
from ._mm_expr import mm, MultimodalExpr  # noqa: F401

# DuckDB-compatible type aliases (Vane/DuckDB tests use these names).
DuckDBPyRelation = Relation
DuckDBPyConnection = Connection

# DuckDB exposes its execution vector size as a module constant.
__standard_vector_size__ = 2048


class CSVLineTerminator(_enum.Enum):
    """DuckDB-compatible CSV line-terminator enum (compat shim)."""

    LINE_FEED = "\n"
    CARRIAGE_RETURN_LINE_FEED = "\r\n"


# Spark compatibility
def getSparkSession():
    from .experimental.spark.sql.session import SparkSession
    return SparkSession

__all__ = [
    "Connection", "Relation", "Expression", "Config", "EnvRegistry",
    "connect", "sql", "col", "lit", "sql_expr",
    "configure", "current_config", "make_env",
    "attach_function", "detach_function", "func", "cls",
    "__version__", "apilevel", "paramstyle", "threadsafety",
    "getSparkSession", "runners", "pipeline", "types",
    "ColumnExpression", "ConstantExpression", "FunctionExpression",
    "CaseExpression", "CoalesceOperator", "StarExpression",
    "DefaultExpression", "SQLExpression", "Value",
    "DuckDBPyRelation", "DuckDBPyConnection",
    "mm", "MultimodalExpr",
]
