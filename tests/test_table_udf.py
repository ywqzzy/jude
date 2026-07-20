"""Generator / table UDFs: a Python function produces rows -> a jude relation."""
import pyarrow as pa
import jude


def test_list_of_dicts():
    c = jude.connect()
    out = c.table_function_udf(lambda n: [{"i": k, "sq": k * k} for k in range(n)], 4)
    assert out.fetchall() == [(0, 0), (1, 1), (2, 4), (3, 9)]


def test_generator_of_tuples_with_schema():
    c = jude.connect()
    def rows():
        for k in range(3):
            yield (k, f"v{k}")
    out = c.table_function_udf(rows, schema=["id", "name"])
    assert out.fetchall() == [(0, "v0"), (1, "v1"), (2, "v2")]


def test_arrow_result_composes_with_sql():
    c = jude.connect()
    rel = c.table_function_udf(lambda: pa.table({"x": [10, 20, 30]}))
    assert rel.filter("x > 15").fetchall() == [(20,), (30,)]


def test_empty_with_schema():
    c = jude.connect()
    out = c.table_function_udf(lambda: [], schema=["a", "b"])
    assert out.fetchall() == []
