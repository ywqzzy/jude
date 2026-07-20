"""SparkSession — PySpark-compatible API backed by jude/DuckDB."""

import jude


class SparkSession:
    """A PySpark-compatible SparkSession backed by DuckDB."""

    def __init__(self, conn=None):
        self._conn = conn or jude.connect()
        self._conf = {}

    @property
    def conf(self):
        return RuntimeConfig(self._conn)

    def sql(self, sqlQuery):
        rel = self._conn.sql(sqlQuery)
        from .dataframe import DataFrame
        return DataFrame(rel, self)

    def table(self, tableName):
        rel = self._conn.table(tableName)
        from .dataframe import DataFrame
        return DataFrame(rel, self)

    def range(self, start, end=None, step=1, numPartitions=None):
        if end is None:
            end = start
            start = 0
        rel = self._conn.sql(f"SELECT * FROM range({start}, {end}, {step}) AS t(id)")
        from .dataframe import DataFrame
        return DataFrame(rel, self)

    def createDataFrame(self, data, schema=None):
        import pyarrow as pa
        from .dataframe import DataFrame

        if isinstance(data, list):
            if data and isinstance(data[0], (list, tuple)):
                col_count = len(data[0])
                col_names = list(schema) if (schema and isinstance(schema, list)) else [f"_c{i}" for i in range(col_count)]
                columns = {name: [row[i] for row in data] for i, name in enumerate(col_names)}
                table = pa.table(columns)
                rel = self._conn.from_arrow(table)
            elif data and isinstance(data[0], dict):
                col_names = list(schema) if (schema and isinstance(schema, list)) else list(data[0].keys())
                columns = {name: [row.get(name) for row in data] for name in col_names}
                table = pa.table(columns)
                rel = self._conn.from_arrow(table)
            else:
                table = pa.table({"value": data})
                rel = self._conn.from_arrow(table)
        elif hasattr(data, "column") or hasattr(data, "to_batches"):
            # Already a pyarrow Table / RecordBatch-like object.
            rel = self._conn.from_arrow(data)
        else:
            table = pa.table(data)
            rel = self._conn.from_arrow(table)

        return DataFrame(rel, self)

    def _format_val(self, v):
        if v is None:
            return "NULL"
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    def newSession(self):
        return SparkSession()

    @classmethod
    def getActiveSession(cls):
        return getattr(cls, "_active", None)

    def stop(self):
        pass

    @property
    def version(self):
        return "3.5.0"

    class Builder:
        def __init__(self):
            self._config = {}
            self._master = None
            self._app_name = None

        def master(self, master):
            self._master = master
            return self

        def appName(self, name):
            self._app_name = name
            return self

        def config(self, key, value):
            self._config[key] = value
            return self

        def enableHiveSupport(self):
            return self

        def remote(self, url):
            return self

        def getOrCreate(self):
            session = SparkSession()
            session._conf = self._config
            SparkSession._active = session
            return session

    builder = Builder()


class RuntimeConfig:
    def __init__(self, conn):
        self._conn = conn

    def set(self, key, value):
        try:
            self._conn.execute_batch(f"SET {key} = '{value}'")
        except Exception:
            pass

    def get(self, key, default=None):
        try:
            rel = self._conn.sql(f"SELECT current_setting('{key}')")
            rows = rel.fetchall()
            return rows[0][0] if rows else default
        except Exception:
            return default

    def getOrElse(self, key, default):
        return self.get(key, default)
