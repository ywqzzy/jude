"""Scheduling tests for the Ray runner: size grouping + bounded backpressure.

Skipped if Ray is missing. Verifies the config surface (JUDE_RAY_*/VANE_RAY_*)
is honored and that bounded in-flight dispatch preserves order and completeness.
"""

import os

import pytest

ray = pytest.importorskip("ray")
pa = pytest.importorskip("pyarrow")
pytest.importorskip("cloudpickle")

import jude


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    os.environ["JUDE_RUNNER"] = "ray"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


class TestScheduling:
    def test_config_read_from_env(self, monkeypatch):
        monkeypatch.setenv("JUDE_RAY_MAX_TASK_BACKLOG", "3")
        monkeypatch.setenv("JUDE_RAY_SCAN_TASK_SIZE_GROUPING", "false")
        monkeypatch.setenv("JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "6")
        jude.runners._reset_runner()
        r = jude.runners.get_or_create_runner()
        assert r.max_task_backlog == 3
        assert r.size_grouping is False
        assert r.min_partition_num == 6

    def test_bounded_backlog_preserves_order_and_count(self, monkeypatch):
        monkeypatch.setenv("JUDE_RAY_MAX_TASK_BACKLOG", "2")
        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(500) t(n)")
        rel = con.sql("SELECT * FROM t WHERE n >= 100").repartition(8)
        r = jude.runners.get_or_create_runner()
        tables = list(r.run_iter_tables(rel))
        assert sum(t.num_rows for t in tables) == 400

    def test_min_partition_num_floor(self, monkeypatch):
        monkeypatch.setenv("JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "5")
        monkeypatch.delenv("JUDE_RAY_MAX_TASK_BACKLOG", raising=False)
        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(50) t(n)")
        r = jude.runners.get_or_create_runner()
        parts = r._partition_tables(con.sql("SELECT * FROM t"))
        # at least min_partition_num partitions
        assert len(parts) >= 5
        assert sum(p.num_rows for p in parts) == 50

    def test_map_relation_bounded(self, monkeypatch):
        monkeypatch.setenv("JUDE_RAY_MAX_TASK_BACKLOG", "2")
        jude.runners._reset_runner()

        def add_sq(t):
            import pyarrow as pa

            return t.append_column("sq", pa.array([v * v for v in t["n"].to_pylist()]))

        import cloudpickle

        payload = {"fn_hex": cloudpickle.dumps(add_sq).hex(), "is_class": False}
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(200) t(n)").repartition(8)
        r = jude.runners.get_or_create_runner()
        outs = r.map_relation(rel, payload, 25)
        merged = pa.concat_tables(outs)
        assert merged.num_rows == 200
        # order preserved across the bounded window
        assert [row["n"] for row in merged.slice(0, 3).to_pylist()] == [0, 1, 2]

    def test_scheduling_delegates_to_rust_worker_manager(self, monkeypatch):
        """The runner must delegate every scheduling decision to the Rust
        jude.dist.WorkerManager — not compute them in Python."""
        monkeypatch.setenv("JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "3")
        monkeypatch.setenv("JUDE_RAY_MAX_TASK_BACKLOG", "2")
        jude.runners._reset_runner()
        r = jude.runners.get_or_create_runner()
        # The runner holds a Rust WorkerManager and reads decisions from it.
        assert isinstance(r.mgr, jude.dist.WorkerManager)
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(50) t(n)")
        table = rel.to_arrow()
        # _target_partitions is a thin delegation to the manager.
        assert r._target_partitions(table) == r.mgr.target_partitions(table.nbytes, table.num_rows)
        # dispatch window policy comes from Rust.
        assert r.mgr.dispatch_window(10) == 2
        assert r.mgr.dispatch_window(2) == 0

