"""Multimodal + distributed: dist_image_dedup must match single-node
curate_mm.image_dedup (perceptual-hash LSH bands -> Hamming verify -> global UF)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from jude import curate_mm

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _img(seed, h=32, w=32):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype="uint8")


def _near(base):
    return np.clip(base.astype("int16") + 3, 0, 255).astype("uint8")


def _tbl(images, ids=None):
    cols = {"image": [im.tolist() for im in images]}
    if ids is not None:
        cols["id"] = ids
    return pa.table(cols)


def test_dist_image_dedup_matches_single_node():
    from jude.curate_dist import dist_image_dedup

    base = _img(0)
    imgs = [base, _near(base), _img(999), _img(1000), _near(_img(1000))]
    t = _tbl(imgs, ids=[0, 1, 2, 3, 4])
    single = curate_mm.image_dedup(t, max_distance=8)
    dist = dist_image_dedup(t, max_distance=8)
    assert sorted(dist.column("id").to_pylist()) == sorted(single.column("id").to_pylist())


def test_dist_image_dedup_distinct_kept():
    from jude.curate_dist import dist_image_dedup

    t = _tbl([_img(1), _img(2), _img(3)], ids=[0, 1, 2])
    out = dist_image_dedup(t, max_distance=2)
    assert out.num_rows == 3        # unrelated images -> all survive


def test_dist_image_dedup_keep_cluster():
    from jude.curate_dist import dist_image_dedup

    base = _img(5)
    t = _tbl([base, _near(base), _img(42)])
    out = dist_image_dedup(t, max_distance=8, keep_cluster=True)
    reps = out.column("img_cluster").to_pylist()
    assert reps[0] == reps[1] and reps[2] != reps[0]
