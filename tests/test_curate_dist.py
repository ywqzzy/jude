"""Distributed curation operators: match single-node ground truth. Needs Ray."""

from __future__ import annotations

import pyarrow as pa
import pytest

import jude
from jude import curate, curate_dist as cd

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _runner():
    from jude.runners.ray import RayRunner

    return RayRunner(num_workers=4)


def _sorted(t, col):
    return sorted(t.column(col).to_pylist())


# --- map-style ---------------------------------------------------------------


def test_dist_quality_filter_matches_local():
    good = ("The history of natural language processing began in the nineteen fifties "
            "when researchers first explored automated translation and generation of text "
            "across many human languages using early digital computers and simple rules here.")
    docs = [good] * 6 + ["too short", "!@#$ " * 30]
    t = pa.table({"id": list(range(len(docs))), "text": docs})
    local = curate.quality_filter(t, min_words=20)
    dist = cd.dist_quality_filter(t, runner=_runner(), min_words=20)
    assert _sorted(dist, "id") == _sorted(local, "id")


def test_dist_detect_language_matches_local():
    t = pa.table({"id": [1, 2, 3], "text": [
        "the quick brown fox and the lazy dog is in the yard for a while now today",
        "这是一段中文文本内容需要识别语言",
        "le chat et le chien sont dans la maison des amis proches ici maintenant",
    ]})
    local = curate.detect_language(t)
    dist = cd.dist_detect_language(t, runner=_runner())
    lm = dict(zip(local.column("id").to_pylist(), local.column("lang").to_pylist()))
    dm = dict(zip(dist.column("id").to_pylist(), dist.column("lang").to_pylist()))
    assert lm == dm


def test_dist_chunk_text_matches_local_rowcount():
    t = pa.table({"id": [1, 2, 3], "text": ["a" * 30, "b" * 45, "c" * 10]})
    local = curate.chunk_text(t, chunk_chars=10, recursive=False)
    dist = cd.dist_chunk_text(t, runner=_runner(), chunk_chars=10, recursive=False)
    assert dist.num_rows == local.num_rows


# --- dedup shuffle -----------------------------------------------------------


def test_dist_exact_dedup_matches_local():
    docs = ["Hello World", "  hello   world  ", "different one", "DIFFERENT ONE", "unique text"]
    t = pa.table({"id": [1, 2, 3, 4, 5], "text": docs})
    local = curate.exact_dedup(t)
    dist = cd.dist_exact_dedup(t, runner=_runner())
    # same set of surviving normalized contents
    assert dist.num_rows == local.num_rows
    assert _sorted(dist, "id") == _sorted(local, "id")


def test_dist_exact_dedup_dedups_across_partitions():
    # 40 rows, only 2 distinct contents -> must dedup to 2 even when split across workers
    docs = (["alpha content here"] * 20) + (["beta content there"] * 20)
    t = pa.table({"id": list(range(40)), "text": docs})
    dist = cd.dist_exact_dedup(t, runner=_runner())
    assert dist.num_rows == 2


def test_dist_fuzzy_dedup_removes_near_dups():
    # 3 clearly-distinct topics, each with near-dup variants (one word changed)
    topics = [
        "the quick brown fox jumps over the lazy dog in the meadow",
        "rust systems programming delivers memory safety without garbage collection",
        "ocean tides rise and fall with the gravitational pull of the moon",
    ]
    docs = []
    for topic in topics:
        for k in range(5):
            docs.append(topic + (" indeed" if k % 2 else " truly"))  # tiny variation
    t = pa.table({"id": list(range(len(docs))), "text": docs})
    dist = cd.dist_fuzzy_dedup(t, runner=_runner(), threshold=0.6, bands=16)
    # 15 docs across 3 distinct near-dup groups -> far fewer than 15, and the
    # 3 topics are dissimilar so they never merge across groups (>= 3 survive)
    assert dist.num_rows < 15
    assert dist.num_rows >= 3


def test_dist_global_shuffle_preserves_rows():
    # global shuffle keeps every row exactly once, but reorders them
    t = pa.table({"id": list(range(500)), "v": list(range(500))})
    out = cd.dist_global_shuffle(t, seed=7, runner=_runner())
    assert out.num_rows == 500
    assert sorted(out.column("id").to_pylist()) == list(range(500))  # same rows
    # reordered (astronomically unlikely to be identity across a distributed shuffle)
    assert out.column("id").to_pylist() != list(range(500))


def test_dist_global_shuffle_deterministic():
    t = pa.table({"id": list(range(300))})
    a = cd.dist_global_shuffle(t, seed=3, runner=_runner()).column("id").to_pylist()
    b = cd.dist_global_shuffle(t, seed=3, runner=_runner()).column("id").to_pylist()
    assert a == b  # same seed -> same permutation


def test_dist_blend_datasets_weighted_and_shuffled():
    from collections import Counter

    a = pa.table({"x": list(range(100)), "src": ["a"] * 100})
    b = pa.table({"x": list(range(200, 300)), "src": ["b"] * 100})
    out = cd.dist_blend_datasets([a, b], [0.7, 0.3], total_rows=100, seed=1, runner=_runner())
    assert out.num_rows == 100
    mix = Counter(out.column("src").to_pylist())
    assert mix["a"] == 70 and mix["b"] == 30  # 70/30 weighted mix
