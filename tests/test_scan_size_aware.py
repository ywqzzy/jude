"""E3: size-aware split assignment. The Rust WorkerManager balances total bytes
per worker (worst-fit bin-packing) instead of an even file-count split, and the
distributed scan still reads every row correctly under size skew."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _runner(n=4):
    from jude.runners.ray import RayRunner
    return RayRunner(num_workers=n)


def test_worker_manager_assign_by_size_balances():
    from jude.jude import dist

    m = dist.WorkerManager(3)
    sizes = [100, 90, 80, 70, 60, 50]
    a = m.assign_by_size(sizes)
    assert len(a) == 6 and all(0 <= w < 3 for w in a)
    load = [0, 0, 0]
    for s, w in zip(sizes, a):
        load[w] += s
    assert max(load) - min(load) <= 30   # balanced (even split would be worse)


def test_assign_by_size_isolates_one_giant_file():
    from jude.jude import dist

    m = dist.WorkerManager(2)
    sizes = [1000] + [1] * 10
    a = m.assign_by_size(sizes)
    giant = a[0]
    other_load = sum(s for s, w in zip(sizes, a) if w != giant)
    assert other_load == 10   # tiny files pile on the other worker, not the giant's


def test_distributed_scan_size_skew_reads_all_rows():
    # files of very different sizes: correctness must hold under size-aware sharding
    d = tempfile.mkdtemp()
    total = 0
    for i in range(6):
        rows = 10 if i < 5 else 5000        # one much bigger file
        pq.write_table(pa.table({"id": list(range(total, total + rows))}),
                       f"{d}/part{i}.parquet")
        total += rows
    out = _runner().distributed_scan("parquet", f"{d}/*.parquet")
    assert out.num_rows == total
    assert set(out.column("id").to_pylist()) == set(range(total))   # every row once


def test_distributed_scan_projection_pushdown_still_works():
    d = tempfile.mkdtemp()
    for i in range(4):
        pq.write_table(pa.table({"id": list(range(i * 50, i * 50 + 50)), "g": [i] * 50}),
                       f"{d}/f{i}.parquet")
    out = _runner().distributed_scan("parquet", f"{d}/*.parquet", columns=["id"], where="g >= 2")
    assert out.column_names == ["id"]
    assert out.num_rows == 100
