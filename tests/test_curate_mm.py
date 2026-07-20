"""Multimodal (image) data-curation: perceptual-hash dedup + quality filter."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from jude import curate_mm as cm


def _pattern(h, w, phase=0):
    a = np.zeros((h, w, 3), dtype="uint8")
    for y in range(h):
        for x in range(w):
            a[y, x] = ((((x + phase) * 6 // w) + (y * 4 // h)) % 2) * 200 + 20
    return a


def _solid(h, w, val):
    return np.full((h, w, 3), val, dtype="uint8")


def _tbl(images: list, **cols) -> pa.Table:
    """Store decoded images as nested python lists in an Arrow column (a 3D
    tensor can't be a plain Arrow value; curate_mm coerces lists via numpy)."""
    data = {"image": [im.tolist() for im in images]}
    data.update(cols)
    return pa.table(data)


def test_add_image_hash_column():
    t = _tbl([_pattern(64, 64), _solid(32, 32, 128)])
    out = cm.add_image_hash(t, column="image", out_column="phash")
    hs = out.column("phash").to_pylist()
    assert len(hs) == 2
    assert all(len(h) == 16 for h in hs)  # 64-bit -> 16 hex


def test_image_dedup_removes_resized_duplicate():
    # same pattern at two resolutions = near-dup; a solid image = distinct
    t = _tbl([_pattern(64, 64), _pattern(48, 48), _solid(40, 40, 100)], id=[1, 2, 3])
    out = cm.image_dedup(t, column="image", max_distance=6)
    kept = out.column("id").to_pylist()
    assert 3 in kept  # the solid image survives
    assert len(kept) == 2  # the two patterns collapsed to one


def test_image_dedup_cluster_annotation():
    t = _tbl([_pattern(64, 64), _pattern(48, 48), _solid(40, 40, 200)])
    out = cm.image_dedup(t, max_distance=8, keep_cluster=True)
    clusters = out.column("img_cluster").to_pylist()
    assert clusters[0] == clusters[1]  # near-dup patterns same cluster
    assert clusters[2] != clusters[0]


def test_add_image_quality_columns():
    t = _tbl([_pattern(64, 48)])
    out = cm.add_image_quality(t)
    assert out.column("img_width").to_pylist()[0] == 48
    assert out.column("img_height").to_pylist()[0] == 64
    assert abs(out.column("img_aspect_ratio").to_pylist()[0] - 48 / 64) < 1e-6


def test_image_quality_filter_min_resolution():
    t = _tbl([_pattern(64, 64), _pattern(10, 10)], id=[1, 2])
    out = cm.image_quality_filter(t, min_width=32, min_height=32)
    assert out.column("id").to_pylist() == [1]


def test_image_quality_filter_drops_blurry():
    # solid image = zero sharpness (blurry); pattern = high sharpness
    t = _tbl([_pattern(64, 64), _solid(64, 64, 128)], id=[1, 2])
    out = cm.image_quality_filter(t, min_sharpness=10.0)
    assert out.column("id").to_pylist() == [1]


def test_image_hash_from_encoded_bytes():
    # PNG-encode a pattern, dedup should still work via PIL decode path
    from PIL import Image
    import io

    def enc(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    t = pa.table({"id": [1, 2], "image": [enc(_pattern(64, 64)), enc(_pattern(48, 48))]})
    out = cm.image_dedup(t, column="image", max_distance=8)
    assert out.num_rows == 1  # near-dups collapse
