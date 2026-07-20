"""jude.curate_mm — multimodal (image/frame) data-curation operators (C17).

The multimodal analogues of jude.curate's text operators — for cleaning image
(and video-frame) training data:

- ``add_image_hash`` / ``image_dedup``  — perceptual-hash (pHash/aHash/dHash)
  near-duplicate detection & removal. Catches the same image at different
  resolution / compression / minor edits (exact byte-hash can't).
- ``add_image_quality`` / ``image_quality_filter`` — resolution / aspect /
  brightness / blur(sharpness) / exposure signals + a filter for datacomp-style
  low-quality image removal.

These operate on a **decoded pixel tensor column** (H×W×C uint8, from
``jude.multimodal.decode_image_batch`` or a raw-bytes column decoded via PIL),
so they compose with the streaming video-frame source and the cosmos pipeline.
Compute cores are Rust (``jude.jude._curate``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from .jude import _curate

__all__ = [
    "add_image_hash",
    "image_dedup",
    "add_image_quality",
    "image_quality_filter",
]


def _decode_to_hwc(val: Any) -> tuple[list[int], int, int, int] | None:
    """Coerce one image cell to (flat_uint8, h, w, c). Accepts:
    - a numpy array (H,W,C) or (H,W)
    - raw encoded bytes (decoded via PIL)
    Returns None for null/undecodable."""
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(bytes(val))).convert("RGB")
            arr = np.asarray(img, dtype="uint8")
        except Exception:  # noqa: BLE001
            return None
    else:
        arr = np.asarray(val)
        if arr.dtype != np.uint8:
            arr = arr.astype("uint8")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        return None
    h, w, c = arr.shape
    return arr.reshape(-1).tolist(), int(h), int(w), int(c)


def _hash_column(table: pa.Table, column: str, kind: str) -> list[str | None]:
    out: list[str | None] = []
    for val in table.column(column).to_pylist():
        dec = _decode_to_hwc(val)
        if dec is None:
            out.append(None)
        else:
            flat, h, w, c = dec
            out.append(_curate.image_hash(flat, h, w, c, kind))
    return out


def add_image_hash(
    table: pa.Table, *, column: str = "image", out_column: str = "phash", kind: str = "phash"
) -> pa.Table:
    """Add a perceptual-hash column (``kind`` = phash|ahash|dhash). Near-duplicate
    images have small Hamming distance between hashes."""
    hashes = _hash_column(table, column, kind)
    return table.append_column(out_column, pa.array(hashes, type=pa.string()))


def image_dedup(
    table: pa.Table,
    *,
    column: str = "image",
    kind: str = "phash",
    max_distance: int = 6,
    bands: int = 4,
    keep_cluster: bool = False,
) -> pa.Table:
    """Remove near-duplicate images via perceptual hashing. Images whose hashes
    are within ``max_distance`` Hamming bits are the same picture (resized /
    recompressed / lightly edited) and collapse to one (lowest row index).

    Uses hash-prefix bucketing to avoid O(n^2): the hex hash is split into
    ``bands`` windows and images sharing ANY window are candidates, verified by
    exact Hamming distance, then union-found into clusters. More ``bands`` = more
    recall (tolerates differences in more windows) at more candidate pairs; tune
    it with ``max_distance``. ``keep_cluster`` annotates ``img_cluster`` instead
    of dropping.
    """
    hashes = _hash_column(table, column, kind)
    n = len(hashes)
    if n == 0:
        return table
    # LSH-style banding on the hex hash: split into `bands` windows; images
    # sharing ANY window are candidates. This tolerates differences outside a
    # window (a single-prefix bucket would miss near-dups that differ early).
    bands = max(1, int(bands))
    buckets: dict[tuple, list[int]] = {}
    for i, h in enumerate(hashes):
        if h is None:
            continue
        wlen = max(1, len(h) // bands)
        for b in range(bands):
            window = h[b * wlen : (b + 1) * wlen]
            buckets.setdefault((b, window), []).append(i)
    pairs: list[tuple[int, int]] = []
    checked: set = set()
    for ids in buckets.values():
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                a, b = ids[ai], ids[bi]
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                if _curate.image_hash_distance(hashes[a], hashes[b]) <= max_distance:
                    pairs.append((a, b))
    reps = _curate.connected_components(n, pairs)
    # null-hash rows are singletons (rep == self already)
    if keep_cluster:
        return table.append_column("img_cluster", pa.array(reps, type=pa.int64()))
    keep = [i for i in range(n) if reps[i] == i]
    return table.take(pa.array(keep, type=pa.int64()))


_IMG_Q_FIELDS = [
    ("width", pa.int64()),
    ("height", pa.int64()),
    ("aspect_ratio", pa.float64()),
    ("brightness", pa.float64()),
    ("sharpness", pa.float64()),
    ("extreme_ratio", pa.float64()),
]


def _quality_rows(table: pa.Table, column: str) -> list[dict | None]:
    out: list[dict | None] = []
    for val in table.column(column).to_pylist():
        dec = _decode_to_hwc(val)
        if dec is None:
            out.append(None)
        else:
            flat, h, w, c = dec
            out.append(_curate.image_quality(flat, h, w, c))
    return out


def add_image_quality(table: pa.Table, *, column: str = "image", prefix: str = "img_") -> pa.Table:
    """Add image quality-signal columns (prefixed): width, height, aspect_ratio,
    brightness, sharpness (Laplacian variance — low = blurry), extreme_ratio
    (over/under-exposed pixel fraction)."""
    rows = _quality_rows(table, column)
    out = table
    for name, typ in _IMG_Q_FIELDS:
        vals = [(r.get(name) if r is not None else None) for r in rows]
        out = out.append_column(prefix + name, pa.array(vals, type=typ))
    return out


def image_quality_filter(
    table: pa.Table,
    *,
    column: str = "image",
    min_width: int = 0,
    min_height: int = 0,
    min_sharpness: float = 0.0,
    max_extreme_ratio: float = 1.0,
    min_aspect: float = 0.0,
    max_aspect: float = float("inf"),
) -> pa.Table:
    """Keep images passing quality thresholds: minimum resolution, minimum
    sharpness (drop blurry), maximum over/under-exposed fraction, aspect-ratio
    bounds (drop banners/thin crops). datacomp-style image filtering."""
    rows = _quality_rows(table, column)
    keep: list[int] = []
    for i, q in enumerate(rows):
        if q is None:
            continue
        if q["width"] < min_width or q["height"] < min_height:
            continue
        if q["sharpness"] < min_sharpness:
            continue
        if q["extreme_ratio"] > max_extreme_ratio:
            continue
        if not (min_aspect <= q["aspect_ratio"] <= max_aspect):
            continue
        keep.append(i)
    return table.take(pa.array(keep, type=pa.int64()))
