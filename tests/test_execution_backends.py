"""Tests for the multi-backend execution engine (jude.execution).

Aligns with Vane's duckdb/execution: ray_task (stateless) / ray_actor (stateful)
backends + streaming. Subprocess backends are covered by test_udf_subprocess.
Skipped if Ray is missing.
"""

import os

import pytest

ray = pytest.importorskip("ray")
pa = pytest.importorskip("pyarrow")
pytest.importorskip("cloudpickle")

import jude
from jude.execution import build_executor, run_ray_map, serialize_udf


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    os.environ["JUDE_RUNNER"] = "ray"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _add_sq(tbl):
    import pyarrow as pa

    return tbl.append_column("sq", pa.array([v * v for v in tbl["n"].to_pylist()]))


def _table(n=100):
    con = jude.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM range({n}) t(n)")
    return con.sql("SELECT * FROM t").to_arrow()


class TestRayTaskBackend:
    def test_map_eager(self):
        payload = serialize_udf(_add_sq)
        ex = build_executor(payload, "ray_task")
        out = ex.map(_table(100), batch_size=25)
        assert out.num_rows == 100
        assert out.column("sq")[5].as_py() == 25

    def test_imap_streaming(self):
        payload = serialize_udf(_add_sq)
        ex = build_executor(payload, "ray_task")
        total = sum(t.num_rows for t in ex.imap(_table(80), batch_size=20))
        assert total == 80


class TestRayActorBackend:
    def test_map_pool(self):
        payload = serialize_udf(_add_sq)
        ex = build_executor(payload, "ray_actor", num_workers=2)
        try:
            out = ex.map(_table(60), batch_size=15)
            assert out.num_rows == 60
            assert out.column("sq")[7].as_py() == 49
        finally:
            ex.shutdown()

    def test_stateful_actor_state_persists(self):
        # An actor class that counts calls; with 1 worker, state accumulates.
        class Counter:
            def __init__(self):
                self.seen = 0

            def __call__(self, tbl):
                import pyarrow as pa

                base = self.seen
                self.seen += tbl.num_rows
                return tbl.append_column(
                    "idx", pa.array([base + i for i in range(tbl.num_rows)])
                )

        payload = serialize_udf(Counter, is_class=True)
        ex = build_executor(payload, "ray_actor", num_workers=1)
        try:
            out = ex.map(_table(20), batch_size=5)
            idxs = sorted(out.column("idx").to_pylist())
            assert idxs == list(range(20))  # continuous -> state persisted
        finally:
            ex.shutdown()


class TestRunRayMap:
    def test_helper_roundtrip(self):
        payload = serialize_udf(_add_sq)
        out = run_ray_map(payload, _table(50), "ray_task", batch_size=10)
        assert out.num_rows == 50
        assert "sq" in out.column_names


class TestRelationBackends:
    def test_relation_ray_task(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(40) t(n)")
        out = con.sql("SELECT * FROM t").map_batches(
            _add_sq, batch_size=10, execution_backend="ray_task"
        )
        assert out.num_rows == 40
        assert "sq" in out.columns

    def test_relation_ray_actor(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(40) t(n)")
        out = con.sql("SELECT * FROM t").map_batches(
            _add_sq, batch_size=10, execution_backend="ray_actor", num_workers=2
        )
        assert out.num_rows == 40
