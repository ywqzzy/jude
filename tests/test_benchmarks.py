"""Benchmark suite for jude — aligned with vane's benchmarking structure."""

import time
import pytest
import jude


class TestSQLBenchmarks:
    """SQL query benchmarks."""

    def setup_method(self):
        self.conn = jude.connect()
        # Create TPC-H-like data
        self.conn.execute("""
            CREATE TABLE lineitem AS
            SELECT
                range as l_orderkey,
                range as l_partkey,
                range % 100 as l_suppkey,
                range % 10 as l_linenumber,
                range % 1000 as l_quantity,
                range % 100 as l_extendedprice,
                range % 10 as l_discount,
                range % 5 as l_tax,
                'N' as l_returnflag,
                'O' as l_linestatus,
                '1998-09-02' as l_shipdate,
                '1998-09-02' as l_commitdate,
                '1998-09-02' as l_receiptdate,
                'TRUCK' as l_shipinstruct,
                'COLLECT COD' as l_shipmode,
                'comment' as l_comment
            FROM range(100000) t(range)
        """)

    @pytest.mark.benchmark
    def test_bench_select(self):
        """Benchmark simple SELECT."""
        start = time.perf_counter()
        rel = self.conn.sql("SELECT * FROM lineitem LIMIT 10000")
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  SELECT 10k rows: {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_aggregate(self):
        """Benchmark aggregation."""
        start = time.perf_counter()
        rel = self.conn.sql("SELECT l_suppkey, COUNT(*) as cnt, SUM(l_quantity) as qty FROM lineitem GROUP BY l_suppkey")
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  GROUP BY aggregate: {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_filter(self):
        """Benchmark filter + limit."""
        start = time.perf_counter()
        rel = self.conn.sql("SELECT * FROM lineitem WHERE l_quantity > 500 LIMIT 100")
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Filter + LIMIT: {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_order_by(self):
        """Benchmark ORDER BY."""
        start = time.perf_counter()
        rel = self.conn.sql("SELECT * FROM lineitem ORDER BY l_quantity DESC LIMIT 100")
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  ORDER BY + LIMIT: {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_join(self):
        """Benchmark self-join."""
        start = time.perf_counter()
        rel = self.conn.sql("""
            SELECT a.l_orderkey, b.l_orderkey
            FROM lineitem a JOIN lineitem b ON a.l_partkey = b.l_partkey
            LIMIT 1000
        """)
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Self-join: {elapsed*1000:.2f}ms")


class TestRelationBenchmarks:
    """Relation operation benchmarks."""

    def setup_method(self):
        self.conn = jude.connect()
        self.conn.execute("CREATE TABLE t AS SELECT * FROM range(50000) t(n)")

    @pytest.mark.benchmark
    def test_bench_limit(self):
        rel = self.conn.sql("SELECT * FROM t")
        start = time.perf_counter()
        limited = rel.limit(1000)
        _ = limited.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Relation.limit(1000): {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_select_columns(self):
        rel = self.conn.sql("SELECT n, n*2 as double_n FROM t")
        start = time.perf_counter()
        selected = rel.select(["n"])
        _ = selected.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Relation.select(['n']): {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_fetchall(self):
        rel = self.conn.sql("SELECT * FROM t LIMIT 1000")
        start = time.perf_counter()
        rows = rel.fetchall()
        elapsed = time.perf_counter() - start
        print(f"\n  Relation.fetchall() 1000 rows: {elapsed*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_bench_to_arrow(self):
        pyarrow = pytest.importorskip("pyarrow")
        rel = self.conn.sql("SELECT * FROM t LIMIT 5000")
        start = time.perf_counter()
        table = rel.to_arrow()
        _ = table.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Relation.to_arrow() 5000 rows: {elapsed*1000:.2f}ms")


class TestUDFBenchmarks:
    """UDF execution benchmarks."""

    def setup_method(self):
        self.conn = jude.connect()
        self.conn.execute("CREATE TABLE t AS SELECT 'hello' as text FROM range(1000)")

    @pytest.mark.benchmark
    def test_bench_scalar_udf(self):
        """Benchmark scalar UDF on 1000 rows."""
        from jude.expression_udf import attach_function

        def upper(s):
            return s.upper()

        attach_function(upper, alias="bench_upper", connection=self.conn, parameters=["VARCHAR"], return_dtype="VARCHAR")

        start = time.perf_counter()
        rel = self.conn.sql("SELECT bench_upper(text) FROM t")
        _ = rel.num_rows
        elapsed = time.perf_counter() - start
        print(f"\n  Scalar UDF 1000 rows: {elapsed*1000:.2f}ms")


def _cpu_heavy_batch(tbl):
    import pyarrow as pa

    out = []
    for v in tbl["n"].to_pylist():
        x = 0
        for i in range(20000):
            x = (x + v * i) % 999983
        out.append(x)
    return tbl.append_column("h", pa.array(out))


class TestUDFParallelismBenchmarks:
    """Compare in-process (GIL-bound) vs out-of-process (GIL-free) UDF execution."""

    @pytest.mark.benchmark
    def test_bench_cpu_udf_inprocess_vs_subprocess(self):
        pytest.importorskip("pyarrow")
        pytest.importorskip("cloudpickle")
        conn = jude.connect()
        conn.execute("CREATE TABLE t AS SELECT * FROM range(4000) t(n)")
        rel = conn.sql("SELECT * FROM t")

        # Warm the persistent worker pool so spawn cost is not counted.
        _ = rel.limit(1).map_batches(
            _cpu_heavy_batch, execution_backend="subprocess", num_workers=8
        ).num_rows

        t0 = time.perf_counter()
        _ = rel.map_batches(_cpu_heavy_batch, batch_size=200).num_rows
        t_in = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = rel.map_batches(
            _cpu_heavy_batch, batch_size=200, execution_backend="subprocess", num_workers=8
        ).num_rows
        t_sub = time.perf_counter() - t0

        print(f"\n  CPU UDF in-process:  {t_in*1000:8.1f} ms")
        print(f"  CPU UDF subprocess8: {t_sub*1000:8.1f} ms  (warm pool)")
        print(f"  speedup:             {t_in/max(t_sub,1e-6):.2f}x")
        jude.shutdown_udf_pools()


class TestAIBenchmarks:
    """AI function benchmarks (requires API keys)."""

    @pytest.mark.skipif(True, reason="Requires OPENAI_API_KEY")
    def test_bench_embed_text(self):
        """Benchmark embedding 100 texts."""
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY")

        conn = jude.connect()
        conn.execute("CREATE TABLE texts AS SELECT 'hello world ' || CAST(n AS VARCHAR) as text FROM range(100) t(n)")
        rel = conn.sql("SELECT * FROM texts")

        start = time.perf_counter()
        from jude.ai import embed_text
        result = embed_text(rel, "text", provider="openai")
        elapsed = time.perf_counter() - start
        print(f"\n  embed_text 100 texts: {elapsed*1000:.2f}ms")
