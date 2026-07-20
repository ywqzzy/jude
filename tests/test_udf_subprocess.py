"""Tests for the out-of-process UDF execution engine (subprocess pool).

The key property: UDFs run in separate worker processes (own interpreter, own
GIL), so N workers give real parallelism. Correctness and process-isolation are
asserted here; the performance win is demonstrated in the benchmarks.
"""

import os

import pytest

import jude

pa = pytest.importorskip("pyarrow")


@pytest.fixture(autouse=True)
def _shutdown_pools():
    yield
    jude.shutdown_udf_pools()


def _append_double(tbl):
    import pyarrow as pa

    return tbl.append_column("d", pa.array([v * 2 for v in tbl["n"].to_pylist()]))


def _append_pid(tbl):
    import os

    import pyarrow as pa

    return tbl.append_column("pid", pa.array([os.getpid()] * tbl.num_rows))


class TestSubprocessUDF:
    def test_correctness_matches_inprocess(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(500) t(n)")
        rel = con.sql("SELECT * FROM t")
        expected = sorted(rel.map_batches(_append_double, batch_size=64).fetchall())
        got = sorted(
            rel.map_batches(
                _append_double, batch_size=64, execution_backend="subprocess", num_workers=4
            ).fetchall()
        )
        assert got == expected

    def test_runs_in_worker_processes(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(400) t(n)")
        rel = con.sql("SELECT * FROM t")
        out = rel.map_batches(
            _append_pid, batch_size=50, execution_backend="subprocess", num_workers=4
        )
        pids = {r[-1] for r in out.fetchall()}
        # UDF never runs in the main process; multiple workers participate.
        assert os.getpid() not in pids
        assert len(pids) >= 2

    def test_row_order_preserved(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(300) t(n)")
        rel = con.sql("SELECT * FROM t")
        out = rel.map_batches(
            _append_double, batch_size=32, execution_backend="subprocess", num_workers=3
        )
        ns = [r[0] for r in out.fetchall()]
        assert ns == list(range(300))

    def test_chain_after_subprocess(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        rel = con.sql("SELECT * FROM t")
        out = rel.map_batches(
            _append_double, batch_size=25, execution_backend="subprocess", num_workers=2
        ).filter("d >= 100")
        # d = n*2 >= 100 -> n >= 50 -> 50 rows
        assert out.num_rows == 50

    def test_pool_reused_across_calls(self):
        # Two calls with the same UDF should reuse the cached pool (same PIDs).
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(200) t(n)")
        rel = con.sql("SELECT * FROM t")
        pids1 = {
            r[-1]
            for r in rel.map_batches(
                _append_pid, batch_size=50, execution_backend="subprocess", num_workers=2
            ).fetchall()
        }
        pids2 = {
            r[-1]
            for r in rel.map_batches(
                _append_pid, batch_size=50, execution_backend="subprocess", num_workers=2
            ).fetchall()
        }
        assert pids1 == pids2  # same persistent workers


class TestStatefulActor:
    def test_cls_batch_state_persists(self):
        @jude.cls.batch(schema={"n": "BIGINT", "seen": "BIGINT"})
        class Counter:
            def __init__(self):
                self.count = 0

            def __call__(self, tbl):
                import pyarrow as pa

                base = self.count
                self.count += tbl.num_rows
                return pa.table(
                    {"n": tbl["n"], "seen": pa.array([base + i for i in range(tbl.num_rows)])}
                )

        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(20) t(n)")
        rel = con.sql("SELECT * FROM t")
        # single worker so the actor accumulates state deterministically
        out = rel.map_batches(
            Counter(), batch_size=5, execution_backend="subprocess", num_workers=1
        )
        rows = out.fetchall()
        assert len(rows) == 20
        # State persisted across batches: the running counter reaches 19.
        assert max(r[1] for r in rows) == 19
