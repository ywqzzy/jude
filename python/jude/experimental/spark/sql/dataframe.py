"""DataFrame — PySpark-compatible DataFrame backed by jude/DuckDB Relation."""

import jude


class DataFrame:
    """A PySpark-compatible DataFrame wrapping a jude Relation."""

    def __init__(self, relation, session):
        self._rel = relation
        self._session = session
        self._schema = None

    @property
    def schema(self):
        if self._schema is None:
            cols = self._rel.columns
            types = self._rel.types
            self._schema = list(zip(cols, types))
        return self._schema

    @property
    def columns(self):
        return self._rel.columns

    @property
    def dtypes(self):
        return [(c, t) for c, t in self.schema]

    def show(self, n=20, **kwargs):
        limited = self._rel.limit(n)
        limited.show()

    def toPandas(self):
        return self._rel.to_df()

    def toArrow(self):
        return self._rel.to_arrow()

    def collect(self):
        return self._rel.fetchall()

    def count(self):
        return self._rel.num_rows

    def first(self):
        rows = self._rel.limit(1).fetchall()
        return rows[0] if rows else None

    def head(self, n=1):
        return self._rel.limit(n).fetchall()

    def take(self, num):
        return self.head(num)

    def printSchema(self):
        print("root")
        for col_name, col_type in self.schema:
            print(f" |-- {col_name}: {col_type}")

    def select(self, *cols):
        col_names = [c if isinstance(c, str) else str(c) for c in cols]
        rel = self._rel.select(col_names)
        return DataFrame(rel, self._session)

    def filter(self, condition):
        rel = self._rel.filter(condition)
        return DataFrame(rel, self._session)

    def where(self, condition):
        return self.filter(condition)

    def sort(self, *cols):
        order_cols = [c if isinstance(c, str) else str(c) for c in cols]
        rel = self._rel.order(", ".join(order_cols))
        return DataFrame(rel, self._session)

    def orderBy(self, *cols):
        return self.sort(*cols)

    def limit(self, n):
        rel = self._rel.limit(n)
        return DataFrame(rel, self._session)

    def join(self, other, on=None, how="inner"):
        if isinstance(on, str):
            on = [on]
        if on and isinstance(on, list):
            condition = " AND ".join(f"lhs.{c} = rhs.{c}" for c in on)
        else:
            condition = str(on) if on else "TRUE"
        rel = self._rel.join(other._rel, condition, how)
        return DataFrame(rel, self._session)

    def crossJoin(self, other):
        rel = self._rel.cross(other._rel)
        return DataFrame(rel, self._session)

    def union(self, other):
        rel = self._rel.union(other._rel)
        return DataFrame(rel, self._session)

    def unionByName(self, other):
        return self.union(other)

    def intersect(self, other):
        rel = self._rel.intersect(other._rel)
        return DataFrame(rel, self._session)

    def exceptAll(self, other):
        rel = self._rel.except_(other._rel)
        return DataFrame(rel, self._session)

    def distinct(self):
        rel = self._rel.distinct()
        return DataFrame(rel, self._session)

    def dropDuplicates(self):
        return self.distinct()

    def withColumn(self, colName, col):
        rel = self._session.sql(
            f"SELECT *, {col if isinstance(col, str) else str(col)} AS {colName} FROM ({self._sql()})"
        )
        return DataFrame(rel, self._session)

    def withColumns(self, colsMap):
        parts = list(self._rel.columns)
        for name, expr in colsMap.items():
            parts.append(f"{expr if isinstance(expr, str) else str(expr)} AS {name}")
        rel = self._session.sql(f"SELECT {', '.join(parts)} FROM ({self._sql()})")
        return DataFrame(rel, self._session)

    def withColumnRenamed(self, existing, new):
        cols = self._rel.columns
        parts = [new if c == existing else c for c in cols]
        rel = self._rel.select(parts)
        return DataFrame(rel, self._session)

    def drop(self, *cols):
        current = self._rel.columns
        keep = [c for c in current if c not in cols]
        rel = self._rel.select(keep)
        return DataFrame(rel, self._session)

    def alias(self, name):
        rel = self._rel.set_alias(name)
        return DataFrame(rel, self._session)

    def cache(self):
        return self

    def createOrReplaceTempView(self, name):
        self._session._conn.execute_batch(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM ({self._sql()})")

    def createGlobalTempView(self, name):
        self._session._conn.execute_batch(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM ({self._sql()})")

    def toDF(self, *cols):
        return self.withColumns({old: new for old, new in zip(self._rel.columns, cols)})

    def _sql(self):
        return self._rel.sql_query() if hasattr(self._rel, "sql_query") else "SELECT * FROM self"

    def __repr__(self):
        return f"DataFrame[{', '.join(f'{c}:{t}' for c, t in self.schema)}]"
