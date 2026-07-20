"""Test suite for jude — aligned with vane's test structure."""

import pytest
import jude


class TestConnection:
    def test_connect_memory(self):
        conn = jude.connect()
        assert conn is not None

    def test_connect_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = jude.connect(db_path)
        assert conn is not None

    def test_sql_select(self):
        conn = jude.connect()
        rel = conn.sql("SELECT 42 as answer")
        assert rel.num_rows == 1
        assert rel.columns == ["answer"]

    def test_execute_create_table(self):
        conn = jude.connect()
        conn.execute("CREATE TABLE t(id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO t VALUES (?, ?)", [1, "Alice"])
        rel = conn.sql("SELECT * FROM t")
        assert rel.num_rows == 1
        assert rel.columns == ["id", "name"]

    def test_execute_batch(self):
        conn = jude.connect()
        conn.execute_batch("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        rel = conn.sql("SELECT COUNT(*) as cnt FROM t")
        rel.show()

    def test_transaction(self):
        conn = jude.connect()
        conn.execute("CREATE TABLE t(id INTEGER)")
        conn.begin()
        conn.execute("INSERT INTO t VALUES (1)")
        conn.rollback()
        rel = conn.sql("SELECT COUNT(*) as cnt FROM t")
        # Should be 0 after rollback

    def test_context_manager(self):
        conn = jude.connect()
        conn.execute("CREATE TABLE t(id INTEGER)")
        with conn:
            conn.execute("INSERT INTO t VALUES (1)")
        # After context, should be committed


class TestRelation:
    def setup_method(self):
        self.conn = jude.connect()
        self.conn.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")

    def test_show(self):
        rel = self.conn.sql("SELECT * FROM t LIMIT 5")
        rel.show()

    def test_columns(self):
        rel = self.conn.sql("SELECT * FROM t")
        assert "n" in rel.columns

    def test_num_rows(self):
        rel = self.conn.sql("SELECT * FROM t")
        assert rel.num_rows == 100

    def test_shape(self):
        rel = self.conn.sql("SELECT * FROM t")
        shape = rel.shape
        assert shape[0] == 100

    def test_select(self):
        rel = self.conn.sql("SELECT n, n*2 as double_n FROM t")
        selected = rel.select(["n"])
        assert selected.columns == ["n"]

    def test_limit(self):
        rel = self.conn.sql("SELECT * FROM t")
        limited = rel.limit(10)
        assert limited.num_rows == 10

    def test_limit_offset(self):
        rel = self.conn.sql("SELECT * FROM t")
        limited = rel.limit(5, 10)
        assert limited.num_rows == 5

    def test_fetchall(self):
        rel = self.conn.sql("SELECT * FROM t LIMIT 3")
        rows = rel.fetchall()
        assert len(rows) == 3

    def test_fetchone(self):
        rel = self.conn.sql("SELECT * FROM t LIMIT 1")
        row = rel.fetchone()
        assert row is not None

    def test_to_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        rel = self.conn.sql("SELECT * FROM t LIMIT 5")
        table = rel.to_arrow()
        assert table.num_rows == 5

    def test_to_df(self):
        pd = pytest.importorskip("pandas")
        rel = self.conn.sql("SELECT * FROM t LIMIT 5")
        df = rel.to_df()
        assert len(df) == 5


class TestExpressions:
    def test_col(self):
        expr = jude.col("my_col")
        assert "my_col" in expr.to_sql()

    def test_lit_int(self):
        expr = jude.lit(42)
        assert "42" in expr.to_sql()

    def test_lit_string(self):
        expr = jude.lit("hello")
        assert "hello" in expr.to_sql()

    def lit_bool(self):
        expr = jude.lit(True)
        assert "TRUE" in expr.to_sql()

    def test_sql_expr(self):
        expr = jude.sql_expr("COUNT(*) > 5")
        assert "COUNT" in expr.to_sql()

    def test_alias(self):
        expr = jude.col("my_col").alias("renamed")
        sql = expr.to_sql()
        assert "AS" in sql or "as" in sql

    # ---- Arithmetic operators ----

    def test_add(self):
        expr = jude.col("a") + jude.col("b")
        sql = expr.to_sql()
        assert "+" in sql
        assert "a" in sql
        assert "b" in sql

    def test_sub(self):
        expr = jude.col("a") - jude.lit(1)
        assert "-" in expr.to_sql()

    def test_mul(self):
        expr = jude.col("price") * jude.lit(1.1)
        sql = expr.to_sql()
        assert "*" in sql

    def test_div(self):
        expr = jude.col("total") / jude.lit(2)
        assert "/" in expr.to_sql()

    def test_neg(self):
        expr = -jude.col("value")
        assert "-" in expr.to_sql()

    # ---- Comparison operators ----

    def test_eq(self):
        expr = jude.col("status") == jude.lit("active")
        assert "=" in expr.to_sql()

    def test_ne(self):
        expr = jude.col("status") != jude.lit("inactive")
        assert "<>" in expr.to_sql()

    def test_lt(self):
        expr = jude.col("age") < jude.lit(18)
        assert "<" in expr.to_sql()

    def test_gt(self):
        expr = jude.col("age") > jude.lit(18)
        assert ">" in expr.to_sql()

    def test_le(self):
        expr = jude.col("age") <= jude.lit(18)
        assert "<=" in expr.to_sql()

    def test_ge(self):
        expr = jude.col("age") >= jude.lit(18)
        assert ">=" in expr.to_sql()

    # ---- Logical operators ----

    def test_and(self):
        expr = (jude.col("a") > jude.lit(0)) & (jude.col("b") < jude.lit(10))
        assert "AND" in expr.to_sql()

    def test_or(self):
        expr = (jude.col("a") > jude.lit(0)) | (jude.col("b") < jude.lit(10))
        assert "OR" in expr.to_sql()

    def test_invert(self):
        expr = ~jude.col("flag")
        assert "NOT" in expr.to_sql()

    # ---- Expression methods ----

    def test_asc(self):
        expr = jude.col("name").asc()
        assert "ASC" in expr.to_sql()

    def test_desc(self):
        expr = jude.col("name").desc()
        assert "DESC" in expr.to_sql()

    def test_cast(self):
        expr = jude.col("value").cast("INTEGER")
        assert "CAST" in expr.to_sql()
        assert "INTEGER" in expr.to_sql()

    def test_isnull(self):
        expr = jude.col("name").isnull()
        assert "IS NULL" in expr.to_sql()

    def test_isnotnull(self):
        expr = jude.col("name").isnotnull()
        assert "IS NOT NULL" in expr.to_sql()

    def test_between(self):
        expr = jude.col("age").between(jude.lit(18), jude.lit(65))
        assert "BETWEEN" in expr.to_sql()

    def test_isin(self):
        expr = jude.col("status").isin([jude.lit("a"), jude.lit("b")])
        assert "IN" in expr.to_sql()

    def test_isnotin(self):
        expr = jude.col("status").isnotin([jude.lit("a"), jude.lit("b")])
        assert "NOT IN" in expr.to_sql()

    def test_get_name(self):
        expr = jude.col("my_col")
        assert expr.get_name() == "my_col"

    def test_nulls_first(self):
        expr = jude.col("name").nulls_first()
        assert "NULLS FIRST" in expr.to_sql()

    def test_nulls_last(self):
        expr = jude.col("name").nulls_last()
        assert "NULLS LAST" in expr.to_sql()


class TestConfig:
    def test_configure(self):
        cfg = jude.configure(runner="local")
        assert cfg.runner == "local"

    def test_current_config(self):
        cfg = jude.current_config()
        assert hasattr(cfg, "runner")

    def test_invalid_runner(self):
        with pytest.raises(Exception):
            jude.configure(runner="invalid")


class TestEnv:
    def test_env(self):
        env = jude.make_env()
        assert hasattr(env, "runner")
        assert hasattr(env, "udf_parallel")
        assert hasattr(env, "local_exchange_buffer")

    def test_env_as_dict(self):
        env = jude.make_env()
        d = env.as_dict()
        assert "runner" in d


class TestAI:
    def test_ai_import(self):
        from jude.ai import embed_text, classify_text, prompt
        assert embed_text is not None
        assert classify_text is not None
        assert prompt is not None

    def test_token_metrics(self):
        from jude.ai import get_token_metrics, reset_token_metrics
        metrics = get_token_metrics()
        assert isinstance(metrics, list)
        reset_token_metrics()

    def test_load_provider(self):
        from jude.ai import load_provider
        assert load_provider is not None


class TestUDF:
    def test_attach_scalar(self):
        from jude.expression_udf import attach_function, detach_function

        conn = jude.connect()

        def upper(s):
            return s.upper()

        attach_function(upper, alias="my_upper", connection=conn, parameters=["VARCHAR"], return_dtype="VARCHAR")
        result = conn.sql("SELECT my_upper('hello')")
        result.show()

    def test_udf_integer(self):
        from jude.expression_udf import attach_function

        conn = jude.connect()

        def add_one(x):
            return x + 1

        attach_function(add_one, connection=conn, parameters=["INTEGER"], return_dtype="INTEGER")
        result = conn.sql("SELECT add_one(42)")
        rows = result.fetchall()
        assert rows[0][0] == 43

    def test_udf_double(self):
        from jude.expression_udf import attach_function

        conn = jude.connect()

        def double(x):
            return x * 2

        attach_function(double, connection=conn, parameters=["DOUBLE"], return_dtype="DOUBLE")
        result = conn.sql("SELECT double(3.14)")
        rows = result.fetchall()
        assert abs(rows[0][0] - 6.28) < 0.01

    def test_udf_two_params(self):
        from jude.expression_udf import attach_function

        conn = jude.connect()

        def concat(a, b):
            return a + b

        attach_function(concat, connection=conn, parameters=["VARCHAR", "VARCHAR"], return_dtype="VARCHAR")
        result = conn.sql("SELECT concat('hello', 'world')")
        rows = result.fetchall()
        assert rows[0][0] == "helloworld"

    def test_udf_on_table(self):
        from jude.expression_udf import attach_function

        conn = jude.connect()
        conn.execute("CREATE TABLE nums AS SELECT CAST(n AS INTEGER) as n FROM range(5) t(n)")

        def add_one(x):
            return x + 1

        attach_function(add_one, connection=conn, parameters=["INTEGER"], return_dtype="INTEGER")
        result = conn.sql("SELECT add_one(n) FROM nums")
        rows = result.fetchall()
        assert len(rows) == 5
        assert rows[0][0] == 1
        assert rows[4][0] == 5

    def test_func_decorator(self):
        @jude.func(return_dtype="VARCHAR")
        def my_udf(x):
            return x.upper()

        assert hasattr(my_udf, "_jude_is_func")
        assert my_udf._jude_return_dtype == "VARCHAR"

    def test_cls_decorator(self):
        @jude.cls(actor_number=1, return_dtype="VARCHAR")
        class MyClass:
            def __call__(self, x):
                return x

        assert hasattr(MyClass, "_jude_is_cls")
        assert MyClass._jude_actor_number == 1

    def test_cls_batch_decorator(self):
        @jude.cls.batch(schema={"result": "VARCHAR"})
        class MyBatch:
            def __call__(self, table):
                return table

        assert hasattr(MyBatch, "_jude_is_cls_batch")
        assert MyBatch._jude_schema == {"result": "VARCHAR"}


class TestExceptions:
    def test_exception_hierarchy(self):
        from jude.exceptions import (
            Error, DatabaseError, DataError, OperationalError,
            ProgrammingError, BinderException, ParserException,
        )
        assert issubclass(DataError, DatabaseError)
        assert issubclass(DatabaseError, Error)
        assert issubclass(OperationalError, DatabaseError)
        assert issubclass(BinderException, DatabaseError)
        assert issubclass(ParserException, DatabaseError)


class TestSpark:
    def test_spark_session(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        assert spark is not None

    def test_spark_sql(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.sql("SELECT 42 as answer")
        assert df.columns == ["answer"]
        rows = df.collect()
        assert len(rows) == 1

    def test_spark_range(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.range(0, 10)
        assert df.count() == 10

    def test_spark_create_dataframe(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], schema=["id", "name"])
        assert df.count() == 2
        assert "name" in df.columns

    def test_spark_select(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.sql("SELECT 1 as a, 2 as b")
        selected = df.select("a")
        assert selected.columns == ["a"]

    def test_spark_filter(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark._conn.execute("CREATE TABLE nums AS SELECT * FROM range(100) t(n)")
        df = spark.table("nums")
        filtered = df.filter("CAST(n AS BIGINT) > 50")
        assert filtered.count() == 49

    def test_spark_limit(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark._conn.execute("CREATE TABLE nums AS SELECT * FROM range(100) t(n)")
        df = spark.table("nums")
        limited = df.limit(10)
        assert limited.count() == 10

    def test_spark_distinct(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.sql("SELECT 1 as v UNION SELECT 1 UNION SELECT 2")
        distinct = df.distinct()
        assert distinct.count() == 2

    def test_spark_show(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.sql("SELECT 1 as a")
        df.show()

    def test_spark_print_schema(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.sql("SELECT 1 as a, 'hello' as b")
        df.printSchema()

    def test_spark_conf(self):
        from jude.experimental.spark.sql.session import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("memory_limit", "1GB")
        val = spark.conf.get("memory_limit")
        assert val is not None
