"""Tests for map_batches/flat_map, repartition, and the jude.runners surface.

Adapted from the shape of Vane's tests/fast/test_local_e2e.py and the
map_batches examples in examples/querying_images.py.
"""

import pytest

import jude


@pytest.fixture(autouse=True)
def _reset_runner():
    jude.runners._reset_runner()
    yield
    jude.runners._reset_runner()


class TestMapBatches:
    def test_identity(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(50) t(n)")
        rel = con.sql("SELECT * FROM t")
        out = rel.map_batches(lambda b: b)
        assert out.num_rows == 50

    def test_append_column(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")
        rel = con.sql("SELECT * FROM t")

        def add(tbl):
            return tbl.append_column("n2", pa.array([v * 2 for v in tbl["n"].to_pylist()]))

        out = rel.map_batches(add, batch_size=4)
        assert out.columns == ["n", "n2"]
        assert out.num_rows == 10
        rows = dict(out.fetchall())
        assert rows[5] == 10

    def test_map_then_filter(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(20) t(n)")
        rel = con.sql("SELECT * FROM t")
        out = rel.map_batches(lambda b: b).filter("n >= 10")
        assert out.num_rows == 10

    def test_flat_map_one_to_many(self):
        pa = pytest.importorskip("pyarrow")
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(3) t(n)")

        def explode(tbl):
            vals = tbl["n"].to_pylist()
            return pa.table({"n": vals + vals})

        out = rel.flat_map(explode)
        assert out.num_rows == 6


class TestRepartition:
    def test_repartition_preserves_rows(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(8)
        assert rel.num_rows == 100
        assert rel.num_partitions == 8

    def test_repartition_then_filter(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4).filter("n < 30")
        assert rel.num_rows == 30

    def test_local_exchange(self):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(10) t(n)").local_exchange(2)
        assert rel.num_partitions == 2
        assert rel.num_rows == 10


class TestRunners:
    def test_default_runner_available(self):
        r = jude.runners.get_or_create_runner()
        # ray falls back to local until Phase 4; either is a valid Runner.
        assert r.name in ("ray", "local")

    def test_set_runner_local(self):
        jude.runners.set_runner_local(num_workers=2)
        r = jude.runners.get_or_create_runner()
        assert r.name == "local"

    def test_run_iter_tables_partitions(self):
        pa = pytest.importorskip("pyarrow")
        jude.runners.set_runner_local()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4)
        r = jude.runners.get_or_create_runner()
        tables = list(r.run_iter_tables(rel))
        assert len(tables) == 4
        assert sum(t.num_rows for t in tables) == 100

    def test_run_iter_materialized_results(self):
        pa = pytest.importorskip("pyarrow")
        jude.runners.set_runner_local()
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(10) t(n)").repartition(2)
        r = jude.runners.get_or_create_runner()
        results = list(r.run_iter(rel))
        assert len(results) == 2
        total = sum(res.metadata().num_rows for res in results)
        assert total == 10

    def test_runner_type_env(self, monkeypatch):
        jude.runners._reset_runner()
        monkeypatch.setenv("JUDE_RUNNER", "local")
        assert jude.runners.get_or_infer_runner_type() == "local"
        jude.runners._reset_runner()
        monkeypatch.setenv("JUDE_RUNNER", "ray")
        assert jude.runners.get_or_infer_runner_type() == "ray"
