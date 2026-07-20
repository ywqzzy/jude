"""Hive-partitioned read (key=value/ layout; partition columns from paths),
single-machine and distributed (files split across Ray workers)."""
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jude


def _hive_dataset():
    base = tempfile.mkdtemp()
    for dt, region, vals in [("2024-01", "us", [1, 2]), ("2024-01", "eu", [3]), ("2024-02", "us", [4, 5])]:
        d = os.path.join(base, f"dt={dt}", f"region={region}")
        os.makedirs(d)
        pq.write_table(pa.table({"v": vals}), os.path.join(d, "part.parquet"))
    return base


def test_read_hive_partition_columns():
    base = _hive_dataset()
    r = jude.connect().read_hive(f"{base}/**/*.parquet")
    assert r.columns == ["v", "dt", "region"]
    assert sorted(x[0] for x in r.fetchall()) == [1, 2, 3, 4, 5]


def test_read_hive_partition_filter():
    base = _hive_dataset()
    c = jude.connect()
    us = c.read_hive(f"{base}/**/*.parquet").filter("region = 'us'")
    assert sorted(x[0] for x in us.fetchall()) == [1, 2, 4, 5]
    feb = c.read_hive(f"{base}/**/*.parquet").filter("dt = '2024-02'")
    assert sorted(x[0] for x in feb.fetchall()) == [4, 5]


def test_distributed_hive_read():
    ray = pytest.importorskip("ray")
    os.environ["JUDE_RUNNER"] = "ray"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    jude.runners._reset_runner()
    base = _hive_dataset()
    r = jude.runners.get_or_create_runner()
    tbl = r.distributed_read_hive(f"{base}/**/*.parquet")
    assert sorted(tbl.column("v").to_pylist()) == [1, 2, 3, 4, 5]
    assert set(tbl.column_names) == {"v", "dt", "region"}
