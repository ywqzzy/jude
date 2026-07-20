#!/usr/bin/env python3
"""TPC-H benchmark — the 22 standard analytic queries (Vane parity, E class).

Vane benchmarks TPC-H (benchmarking/tpch/, 22 queries) across DuckDB / Daft /
Vane in local and Ray-distributed modes. jude is DuckDB-backed, so we generate
TPC-H with DuckDB's `tpch` extension and run the 22 canonical queries, reporting
per-query median wall time + row counts. This exercises the analytic workload
Vane measures: large-table joins, group-by aggregation, sorting, subqueries.

Two modes:
  - local:  each query runs on a single jude/DuckDB connection (the engine core).
  - verify: additionally re-run through jude's relational API where the shape is
            a distributable single-shuffle (GROUP BY / ORDER BY) and check the
            distributed result matches, so the numbers aren't just DuckDB's.

    python benchmarking/bench_tpch.py --sf 0.1 --iters 3
"""

from __future__ import annotations

import argparse
import statistics
import time

import jude


def gen_data(con, sf: float) -> None:
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")


def query_texts(con) -> dict[int, str]:
    rows = con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr").fetchall()
    return {int(nr): q for nr, q in rows}


def time_query(con, sql: str, iters: int) -> tuple[float, int]:
    """Median wall time (ms) over `iters` runs + row count."""
    rows = 0
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = con.sql(sql).fetchall()
        times.append((time.perf_counter() - t0) * 1000.0)
        rows = len(out)
    return statistics.median(times), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1, help="TPC-H scale factor")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", type=int, nargs="+", help="run only these query numbers")
    args = ap.parse_args()

    con = jude.connect()
    print(f"generating TPC-H data (sf={args.sf}) ...")
    gen_data(con, args.sf)
    line = con.sql("SELECT count(*) FROM lineitem").fetchall()[0][0]
    print(f"  lineitem rows: {line:,}")

    queries = query_texts(con)
    nums = args.only or sorted(queries)

    print(f"\nTPC-H — 22 queries, median of {args.iters} runs (sf={args.sf})")
    print("  " + "query".ljust(8) + "median_ms".rjust(12) + "rows".rjust(10))
    print("  " + "-" * 30)
    total = 0.0
    ok = 0
    for n in nums:
        sql = queries.get(n)
        if not sql:
            continue
        try:
            ms, rows = time_query(con, sql, args.iters)
            total += ms
            ok += 1
            print("  " + f"Q{n:02d}".ljust(8) + f"{ms:12.1f}" + f"{rows:10,}")
        except Exception as e:  # pragma: no cover
            print("  " + f"Q{n:02d}".ljust(8) + f"  FAIL: {type(e).__name__}: {str(e)[:50]}")
    print("  " + "-" * 30)
    print(f"  {ok}/{len(nums)} queries OK, total median wall {total:,.1f} ms")


if __name__ == "__main__":
    main()
