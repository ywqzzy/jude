"""Tests for the Ray distributed runner (partition-level orchestration).

Skipped entirely if Ray is not installed. Ray init is slow, so these are kept
minimal and share one cluster for the module.
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
    jude.runners._reset_runner()


@pytest.fixture
def runner():
    jude.runners._reset_runner()
    r = jude.runners.get_or_create_runner()
    assert r.name == "ray"
    return r


def _add_sq(tbl):
    import pyarrow as pa

    return tbl.append_column("sq", pa.array([v * v for v in tbl["n"].to_pylist()]))


class TestRayRunner:
    def test_run_iter_tables_roundtrip(self, runner):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(1000) t(n)")
        rel = con.sql("SELECT * FROM t WHERE n >= 100").repartition(4)
        tables = list(runner.run_iter_tables(rel))
        assert sum(t.num_rows for t in tables) == 900

    def test_run_iter_materialized(self, runner):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(50) t(n)").repartition(3)
        results = list(runner.run_iter(rel))
        assert sum(r.metadata().num_rows for r in results) == 50

    def test_run_write_counts_rows(self, runner):
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(120) t(n)").repartition(4)
        info = runner.run_write(rel)
        assert info["rows_written"] == 120


class TestDistributedAggregate:
    def test_two_phase_group_by(self):
        from jude.runners._agg import build_two_phase

        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n%5 AS g, n AS v FROM range(1000) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4)
        r = jude.runners.get_or_create_runner()

        ref = {
            row[0]: (row[1], row[2], row[3], row[4])
            for row in con.sql(
                "SELECT g, COUNT(*) c, SUM(v) s, MIN(v) mn, MAX(v) mx FROM t GROUP BY g"
            ).fetchall()
        }
        partial, final = build_two_phase(
            ["g"], ["COUNT(*) AS c", "SUM(v) AS s", "MIN(v) AS mn", "MAX(v) AS mx"]
        )
        tbl = r.distributed_aggregate(rel, partial, final)
        got = {row["g"]: (row["c"], row["s"], row["mn"], row["mx"]) for row in tbl.to_pylist()}
        assert got == ref

    def test_distributed_avg(self):
        from jude.runners._agg import build_two_phase

        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n%5 AS g, n AS v FROM range(1000) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4)
        r = jude.runners.get_or_create_runner()
        p, f = build_two_phase(["g"], ["AVG(v) AS a"])
        tbl = r.distributed_aggregate(rel, p, f)
        got = {row["g"]: round(row["a"], 4) for row in tbl.to_pylist()}
        ref = {
            row[0]: round(row[1], 4)
            for row in con.sql("SELECT g, AVG(v) a FROM t GROUP BY g").fetchall()
        }
        assert got == ref

    def test_global_aggregate_no_group(self):
        from jude.runners._agg import build_two_phase

        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(100) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4)
        r = jude.runners.get_or_create_runner()
        p, f = build_two_phase([], ["SUM(n) AS s", "COUNT(*) AS c"])
        tbl = r.distributed_aggregate(rel, p, f)
        row = tbl.to_pylist()[0]
        assert row["s"] == 4950
        assert row["c"] == 100


class TestRayMapBatches:
    def test_distributed_map(self):
        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(200) t(n)")
        rel = con.sql("SELECT * FROM t").repartition(4)
        out = rel.map_batches(_add_sq, batch_size=50, execution_backend="ray")
        assert out.num_rows == 200
        assert "sq" in out.columns
        rows = dict((r[0], r[1]) for r in out.fetchall())
        assert rows[10] == 100
        assert rows[15] == 225

    def test_distributed_map_then_sql(self):
        jude.runners._reset_runner()
        con = jude.connect()
        rel = con.sql("SELECT * FROM range(100) t(n)").repartition(2)
        out = rel.map_batches(_add_sq, execution_backend="ray").filter("sq >= 100")
        # sq = n*n >= 100 -> n >= 10 -> 90 rows
        assert out.num_rows == 90


class TestDistributedJoin:
    def test_hash_join_inner(self):
        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE a AS SELECT n AS k, n*10 AS av FROM range(100) t(n)")
        con.execute("CREATE TABLE b AS SELECT n AS k, n*100 AS bv FROM range(50,150) t(n)")
        r = jude.runners.get_or_create_runner()
        tbl = r.distributed_join(
            con.sql("SELECT * FROM a"), con.sql("SELECT * FROM b"), keys=["k"], num_buckets=4
        )
        # 50 matching keys (50..99); output schema de-dupes the join key
        assert tbl.column_names == ["k", "av", "bv"]
        assert tbl.num_rows == 50
        ref = con.sql(
            "SELECT COUNT(*) c, SUM(a.av) sa, SUM(b.bv) sb FROM a JOIN b ON a.k=b.k"
        ).fetchone()
        rows = tbl.to_pylist()
        assert tbl.num_rows == ref[0]
        assert sum(row["av"] for row in rows) == ref[1]
        assert sum(row["bv"] for row in rows) == ref[2]

    def test_hash_join_no_matches(self):
        jude.runners._reset_runner()
        con = jude.connect()
        con.execute("CREATE TABLE a AS SELECT n AS k FROM range(10) t(n)")
        con.execute("CREATE TABLE b AS SELECT n AS k FROM range(100,110) t(n)")
        r = jude.runners.get_or_create_runner()
        tbl = r.distributed_join(
            con.sql("SELECT * FROM a"), con.sql("SELECT * FROM b"), keys=["k"], num_buckets=2
        )
        assert tbl.num_rows == 0
