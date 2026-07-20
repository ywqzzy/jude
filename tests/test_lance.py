"""Lance format read + write (single-machine) and distributed write on Ray.
Lance's writer is Rust-backed; jude orchestrates. Gated on pylance."""
import os
import tempfile

import pytest

pytest.importorskip("lance")
import jude


@pytest.fixture
def path():
    return os.path.join(tempfile.mkdtemp(), "ds.lance")


def test_write_read_roundtrip(path):
    c = jude.connect()
    rel = c.sql("SELECT range AS id, ('v'||range::VARCHAR) AS name FROM range(100)")
    rel.write_lance(path, mode="create")
    back = jude.read_lance(path)
    assert len(back.fetchall()) == 100
    assert back.aggregate("sum(id)").fetchone()[0] == 99 * 100 // 2


def test_column_and_filter_pushdown(path):
    c = jude.connect()
    c.sql("SELECT range AS id, range*2 AS v FROM range(50)").write_lance(path, mode="create")
    proj = jude.read_lance(path, columns=["id"], filter="id < 5")
    assert proj.fetchall() == [(0,), (1,), (2,), (3,), (4,)]


def test_append(path):
    c = jude.connect()
    c.sql("SELECT range AS id FROM range(10)").write_lance(path, mode="create")
    c.sql("SELECT range+10 AS id FROM range(5)").write_lance(path, mode="append")
    assert len(jude.read_lance(path).fetchall()) == 15


class TestDistributedLanceWrite:
    @pytest.fixture(scope="class", autouse=True)
    def _ray(self):
        ray = pytest.importorskip("ray")
        os.environ["JUDE_RUNNER"] = "ray"
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
        jude.runners._reset_runner()
        yield

    def test_distributed_write_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), "dist.lance")
        c = jude.connect()
        rel = c.sql("SELECT range AS id FROM range(2000)").repartition(4)
        r = jude.runners.get_or_create_runner()
        assert type(r).__name__ == "RayRunner"
        meta = r.distributed_write_lance(rel, path, mode="overwrite")
        assert meta["fragments"] >= 1
        back = jude.read_lance(path)
        assert len(back.fetchall()) == 2000
        assert back.aggregate("sum(id)").fetchone()[0] == 1999 * 2000 // 2


class TestDistributedLanceVectorIndex:
    @pytest.fixture(scope="class", autouse=True)
    def _ray(self):
        ray = pytest.importorskip("ray")
        os.environ["JUDE_RUNNER"] = "ray"
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
        jude.runners._reset_runner()
        yield

    def test_distributed_write_builds_global_index(self):
        import numpy as np
        import pyarrow as pa

        path = os.path.join(tempfile.mkdtemp(), "vec.lance")
        c = jude.connect()
        n = 1000
        vecs = [np.random.rand(8).astype("float32") for _ in range(n)]
        t = pa.table({"id": list(range(n)), "emb": pa.array(vecs, type=pa.list_(pa.float32(), 8))})
        rel = c.from_arrow(t).repartition(4)
        r = jude.runners.get_or_create_runner()
        meta = r.distributed_write_lance(
            rel, path, mode="overwrite",
            vector_index={"column": "emb", "num_partitions": 4, "num_sub_vectors": 2},
        )
        assert meta["fragments"] >= 1 and meta.get("vector_index") == "emb"
        # a GLOBAL index now spans all fragments -> ANN over the whole dataset
        from jude import _lance
        assert any(ix["type"] in ("IVF_PQ", "IVF") for ix in _lance.list_indices(path))
        res = c.lance_vector_search(path, "emb", list(vecs[7]), k=5)
        assert res.order("_distance").fetchone()[0] == 7
        assert len(res.fetchall()) == 5
