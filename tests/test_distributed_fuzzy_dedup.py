"""A3: distributed fuzzy dedup must match single-node curate.fuzzy_dedup recall.

The old distributed path routed each row by its FIRST LSH band only and ran
union-find per bucket — so near-dups sharing a later band, or clusters spanning
buckets (A~B here, B~C there), were silently missed. The fix routes by ALL
bands and runs ONE global connected-components, so the distributed result is the
SAME set of surviving rows as the single-node algorithm.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from jude import curate

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _kept_texts(tbl):
    return sorted(tbl.column("text").to_pylist())


def test_distributed_matches_single_node_recall():
    from jude.curate_dist import dist_fuzzy_dedup

    # exact dups + near dups + uniques; enough rows to spread across buckets
    base = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",           # exact dup of #0
        "the quick brown fox jumps over the lazy dog today",     # near dup of #0
        "machine learning models train on large text corpora",
        "machine learning models train on large text corpora!!", # near dup of #3
        "a completely unrelated sentence about oceans",
        "yet another distinct line concerning mountains and snow",
        "python is a programming language used for data science",
        "python is a programming language used for data science.",  # near dup of #7
    ] * 3  # repeat to get 27 rows over multiple partitions/buckets
    t = pa.table({"text": base, "rid": list(range(len(base)))})

    params = dict(column="text", threshold=0.6, num_hashes=64, ngram=2, bands=8, seed=1)
    single = curate.fuzzy_dedup(t, **params)
    dist = dist_fuzzy_dedup(t, **params)

    # same number of survivors AND the same surviving text multiset
    assert dist.num_rows == single.num_rows
    assert _kept_texts(dist) == _kept_texts(single)


def test_distributed_transitive_cluster_collapses():
    """A~B and B~C but A not directly ~C: all three must collapse to one row via
    the global union-find (the old per-bucket version could keep two)."""
    from jude.curate_dist import dist_fuzzy_dedup

    # a chain of gradually-shifting sentences (each near its neighbor)
    chain = [
        "alpha beta gamma delta epsilon zeta eta theta",
        "alpha beta gamma delta epsilon zeta eta iota",   # ~ prev
        "alpha beta gamma delta epsilon zeta kappa iota", # ~ prev, less ~ first
        "an entirely separate unrelated control sentence here",
    ]
    t = pa.table({"text": chain})
    params = dict(column="text", threshold=0.5, num_hashes=64, ngram=2, bands=8, seed=3)
    single = curate.fuzzy_dedup(t, **params)
    dist = dist_fuzzy_dedup(t, **params)
    assert dist.num_rows == single.num_rows
    assert _kept_texts(dist) == _kept_texts(single)


def test_distributed_keep_cluster_labels_match():
    from jude.curate_dist import dist_fuzzy_dedup

    texts = ["red red red", "red red red", "blue sky above", "green field below", "blue sky above"]
    t = pa.table({"text": texts})
    params = dict(column="text", threshold=0.6, num_hashes=64, ngram=2, bands=8, seed=1)
    single = curate.fuzzy_dedup(t, keep_cluster=True, **params)
    dist = dist_fuzzy_dedup(t, keep_cluster=True, **params)
    # same cluster assignment (representative row id per row)
    assert dist.column("dup_cluster").to_pylist() == single.column("dup_cluster").to_pylist()
