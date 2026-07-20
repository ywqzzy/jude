"""UDF call-mode + batching tests: scalar map, byte-based dynamic batching."""

import pytest

import jude

pa = pytest.importorskip("pyarrow")


class TestScalarMap:
    def test_map_named_column(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(20) t(n)")
        out = con.sql("SELECT * FROM t").map(lambda x: x * x, "n", output_column="sq")
        assert out.columns == ["n", "sq"]
        assert out.num_rows == 20
        assert dict(out.fetchall())[5] == 25

    def test_map_default_first_column(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n FROM range(5) t(n)")
        out = con.sql("SELECT n FROM t").map(lambda x: x + 100)
        rows = dict(out.fetchall())
        assert rows[3] == 103

    def test_map_row_preserving(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n, n*2 AS d FROM range(4) t(n)")
        out = con.sql("SELECT * FROM t").map(lambda x: x + 1, "n", output_column="np1")
        # original columns preserved + new column appended, same row count
        assert out.columns == ["n", "d", "np1"]
        assert out.num_rows == 4
        assert out.order("n").fetchall() == [(0, 0, 1), (1, 2, 2), (2, 4, 3), (3, 6, 4)]

    def test_map_batch_size(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(17) t(n)")
        out = con.sql("SELECT * FROM t").map(lambda x: x, "n", batch_size=4)
        assert out.num_rows == 17


def _add1(tbl):
    import pyarrow as pa

    return tbl.append_column("m", pa.array([v + 1 for v in tbl["n"].to_pylist()]))


class TestByteBatching:
    def test_max_batch_bytes_inprocess_correct(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(50) t(n)")
        # byte batching parameter is accepted; result is correct regardless of chunking
        out = con.sql("SELECT * FROM t").map_batches(_add1, max_batch_bytes=128)
        assert out.num_rows == 50
        assert "m" in out.columns

    def test_max_batch_bytes_subprocess(self):
        pytest.importorskip("cloudpickle")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(40) t(n)")
        out = con.sql("SELECT * FROM t").map_batches(
            _add1, max_batch_bytes=64, execution_backend="subprocess", num_workers=2
        )
        assert out.num_rows == 40
        vals = sorted(r[-1] for r in out.fetchall())
        assert vals == list(range(1, 41))
        jude.shutdown_udf_pools()

    def test_env_byte_target(self, monkeypatch):
        monkeypatch.setenv("JUDE_UDF_TARGET_MAX_BATCH_BYTES", "128")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(30) t(n)")
        # env-driven byte target; correctness preserved
        out = con.sql("SELECT * FROM t").map_batches(_add1)
        assert out.num_rows == 30
