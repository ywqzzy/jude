"""Distributed, worker-side sharded scan: each worker reads its own shard of a
source (parquet/csv/json by file, lance by fragment); a single file/fragment runs
single-node. Verifies correctness vs a single-node read + pushdown. (Fixes B1 for
the scan path — data never funnels through the driver.)"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jude

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _runner(n=4):
    from jude.runners.ray import RayRunner
    return RayRunner(num_workers=n)


def test_parquet_multifile_shard_matches_union():
    d = tempfile.mkdtemp()
    total = 0
    for i in range(5):
        pq.write_table(pa.table({"id": list(range(i * 100, i * 100 + 100)), "g": [i] * 100}),
                       f"{d}/part{i}.parquet")
        total += 100
    out = _runner().distributed_scan("parquet", f"{d}/*.parquet")
    assert out.num_rows == total
    assert set(out.column("id").to_pylist()) == set(range(total))  # every row present, once


def test_parquet_projection_and_predicate_pushdown():
    d = tempfile.mkdtemp()
    for i in range(4):
        pq.write_table(pa.table({"id": list(range(i * 100, i * 100 + 100)), "g": [i] * 100}),
                       f"{d}/p{i}.parquet")
    out = _runner().distributed_scan("parquet", f"{d}/*.parquet", columns=["id"], where="g >= 2")
    assert out.column_names == ["id"]      # projection pushed down
    assert out.num_rows == 200             # predicate pushed down (g in {2,3})


def test_single_file_runs_single_node():
    d = tempfile.mkdtemp()
    pq.write_table(pa.table({"id": list(range(100))}), f"{d}/only.parquet")
    # a single file isn't worth distributing — still correct
    out = _runner().distributed_scan("parquet", f"{d}/only.parquet")
    assert out.num_rows == 100


def test_csv_multifile_shard():
    d = tempfile.mkdtemp()
    for i in range(3):
        with open(f"{d}/c{i}.csv", "w") as f:
            f.write("id,g\n")
            for j in range(50):
                f.write(f"{i*50+j},{i}\n")
    out = _runner().distributed_scan("csv", f"{d}/*.csv")
    assert out.num_rows == 150


def test_lance_fragment_shard():
    lance = pytest.importorskip("lance")
    p = tempfile.mkdtemp() + "/ds"
    jude._lance.write(pa.table({"id": list(range(500)), "g": [i % 4 for i in range(500)]}),
                      p, mode="create")
    out = _runner().distributed_scan("lance", p, columns=["id"], where="g = 1")
    assert out.column_names == ["id"]
    assert out.num_rows == len([i for i in range(500) if i % 4 == 1])
