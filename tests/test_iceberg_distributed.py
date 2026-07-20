"""Distributed Iceberg write on Ray — each worker writes its partition's Parquet
file, the driver commits the file list as one snapshot. Gated on Ray + pyiceberg.
"""

import os
import tempfile

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("pyiceberg")
pytest.importorskip("pyarrow")

import jude


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    os.environ["JUDE_RUNNER"] = "ray"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    jude.runners._reset_runner()
    yield


class TestDistributedIcebergWrite:
    def test_distributed_write_roundtrip(self):
        wh = tempfile.mkdtemp()
        con = jude.connect()
        rel = con.sql("SELECT range AS id, ('v'||range::VARCHAR) AS name FROM range(2000)").repartition(4)
        r = jude.runners.get_or_create_runner()
        assert type(r).__name__ == "RayRunner"
        meta = r.distributed_write_iceberg(rel, wh, "db.dt", mode="append")
        back = jude.read_iceberg(meta)
        assert len(back.fetchall()) == 2000
        # value-correct, not just count-correct
        assert back.aggregate("sum(id)").fetchone()[0] == 1999 * 2000 // 2

    def test_distributed_write_overwrite(self):
        wh = tempfile.mkdtemp()
        con = jude.connect()
        r = jude.runners.get_or_create_runner()
        r.distributed_write_iceberg(con.sql("SELECT range AS id FROM range(500)").repartition(4), wh, "db.ow", "append")
        meta = r.distributed_write_iceberg(con.sql("SELECT range AS id FROM range(5)"), wh, "db.ow", "overwrite")
        assert len(jude.read_iceberg(meta).fetchall()) == 5
