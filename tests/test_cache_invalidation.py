"""A1: resident caches must invalidate after a write.

Before this fix, `_lance._DS_CACHE` (dataset handles), `vector._RESIDENT_VEC`
(in-RAM vector matrix) and the actor-side `_vec_cache` kept serving the
pre-write snapshot after append/delete/merge_insert/add_columns — a silent
correctness bug (query returns stale rows / misses new ones / resurrects deleted
ones). Every mutation now bumps a per-path epoch and drops the cached handle; the
resident matrix is stamped with the epoch and reloads when it advances.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude import _lance

lance = pytest.importorskip("lance")


def _tbl(ids, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(ids), dim)).astype("float32")
    return pa.table({"id": pa.array(list(ids), type=pa.int64()),
                     "vec": pa.array(list(vecs), type=pa.list_(pa.float32(), dim))})


def test_epoch_bumps_on_every_mutation():
    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(100)), p, mode="create")
    e0 = _lance.epoch(p)
    _lance.write(_tbl(range(100, 200), seed=1), p, mode="append")
    e1 = _lance.epoch(p)
    _lance.delete(p, "id >= 150")
    e2 = _lance.epoch(p)
    assert e0 < e1 < e2  # each write advances the epoch


def test_dataset_cached_reflects_append():
    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(100)), p, mode="create")
    # populate the handle cache with a read
    assert _lance.dataset_cached(p).count_rows() == 100
    # append via the public writer -> must invalidate the cached handle
    _lance.write(_tbl(range(100, 250), seed=2), p, mode="append")
    assert _lance.dataset_cached(p).count_rows() == 250  # not the stale 100


def test_dataset_cached_reflects_delete():
    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(200)), p, mode="create")
    assert _lance.dataset_cached(p).count_rows() == 200
    _lance.delete(p, "id >= 120")
    assert _lance.dataset_cached(p).count_rows() == 120  # deleted rows gone


def test_resident_vectors_reload_after_append():
    from jude.vector import _resident_vectors

    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(50)), p, mode="create")
    ids, id_to_row, mat, norms = _resident_vectors(p, "vec")
    assert mat.shape[0] == 50
    # append rows, then re-read the resident matrix: must grow, not stay at 50
    _lance.write(_tbl(range(50, 130), seed=3), p, mode="append")
    ids2, id_to_row2, mat2, norms2 = _resident_vectors(p, "vec")
    assert mat2.shape[0] == 130
    assert 129 in id_to_row2 and 129 not in id_to_row  # new id now resident


def test_resident_vectors_reload_after_delete():
    from jude.vector import _resident_vectors

    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(80)), p, mode="create")
    _, id_to_row, mat, _ = _resident_vectors(p, "vec")
    assert mat.shape[0] == 80 and 79 in id_to_row
    _lance.delete(p, "id >= 40")
    _, id_to_row2, mat2, _ = _resident_vectors(p, "vec")
    assert mat2.shape[0] == 40
    assert 79 not in id_to_row2  # deleted id no longer resident


def test_index_build_invalidates_handle():
    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(300)), p, mode="create")
    ds = _lance.dataset_cached(p)  # cache a handle with no vector index
    assert not ds.list_indices()
    _lance.create_vector_index(p, "vec", index_type="IVF_PQ",
                               num_partitions=4, num_sub_vectors=2)
    # a fresh cached handle must SEE the new index (else ANN silently scans)
    assert _lance.dataset_cached(p).list_indices()


def test_actor_resident_shard_refreshes_on_version_bump():
    """A resident actor pool persists across calls; after a write to a shard path
    the actor must re-decode (version-stamped cache) instead of scoring the query
    against the pre-write matrix."""
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=2)
    from jude.runners._ray_shim import make_workers

    p = tempfile.mkdtemp() + "/ds"
    _lance.write(_tbl(range(60)), p, mode="create")
    w = make_workers(1)[0]
    q = [0.0] * 8
    r1 = ray.get(w.vector_exact_shard.remote(p, "vec", q, 100, "l2"))
    assert r1.num_rows == 60  # all rows scored
    # append rows: the persistent actor must pick them up on the next query
    _lance.write(_tbl(range(60, 200), seed=9), p, mode="append")
    r2 = ray.get(w.vector_exact_shard.remote(p, "vec", q, 300, "l2"))
    assert r2.num_rows == 200  # not the stale 60

