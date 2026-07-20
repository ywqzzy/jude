"""C5 (in-scope part): image_dedup with a configurable band count. Perceptual-
hash near-duplicate image dedup — no model inference (CLIP/NSFW/aesthetic scoring
is the user's UDF, out of jude's scope). Synthetic images, deterministic."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from jude import curate_mm


def _img(seed, h=32, w=32):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype="uint8")


def _near(base):
    # a lightly-edited copy: same picture + tiny brightness tweak
    out = base.astype("int16") + 3
    return np.clip(out, 0, 255).astype("uint8")


def _tbl(images, ids=None):
    # store decoded images as nested python lists (a 3D Arrow list column)
    cols = {"image": [im.tolist() for im in images]}
    if ids is not None:
        cols["id"] = ids
    return pa.table(cols)


def test_image_dedup_collapses_near_duplicates():
    base = _img(0)
    t = _tbl([base, _near(base), _img(999)], ids=[1, 2, 3])  # 1~2 near-dup, 3 distinct
    out = curate_mm.image_dedup(t, max_distance=8)
    kept = out.column("id").to_pylist()
    assert 3 in kept
    assert len(kept) == 2          # the near-dup pair collapsed to one


def test_bands_param_is_honored():
    base = _img(1)
    t = _tbl([base, _near(base)], ids=[1, 2])
    # both band settings should still detect the obvious near-dup
    for bands in (1, 4, 8):
        out = curate_mm.image_dedup(t, max_distance=8, bands=bands)
        assert out.num_rows == 1, f"bands={bands}"


def test_distinct_images_kept():
    t = _tbl([_img(1), _img(2), _img(3)], ids=[1, 2, 3])
    out = curate_mm.image_dedup(t, max_distance=2, bands=4)
    assert out.num_rows == 3       # unrelated images -> all kept


def test_keep_cluster_annotation():
    base = _img(5)
    t = _tbl([base, _near(base), _img(42)])
    out = curate_mm.image_dedup(t, max_distance=8, keep_cluster=True)
    reps = out.column("img_cluster").to_pylist()
    assert reps[0] == reps[1]      # near-dups share a cluster
    assert reps[2] != reps[0]
