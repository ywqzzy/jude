"""L2.1: incremental union-by-min union-find backing distributed global dedup."""

from __future__ import annotations

import pytest

from jude.curate_dist import _UnionFind


def test_union_by_min_root_is_smallest():
    uf = _UnionFind(10)
    uf.union(5, 3)
    uf.union(3, 8)
    uf.union(9, 5)
    # component {3,5,8,9} -> root is the min, 3
    for x in (3, 5, 8, 9):
        assert uf.find(x) == 3
    assert uf.find(0) == 0                      # untouched singleton


def test_incremental_equivalent_to_batch_order():
    # edges fed in any order give the same components (order-independent)
    import random
    edges = [(1, 2), (2, 3), (5, 6), (6, 1), (8, 9)]
    a = _UnionFind(10)
    for x, y in edges:
        a.union(x, y)
    b = _UnionFind(10)
    r = list(edges)
    random.Random(0).shuffle(r)
    for x, y in r:
        b.union(x, y)
    assert [a.find(i) for i in range(10)] == [b.find(i) for i in range(10)]
    # {1,2,3,5,6} all root at 1; {8,9} root at 8
    assert a.find(6) == 1 and a.find(3) == 1 and a.find(9) == 8


def test_transitive_chain():
    uf = _UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)                              # chain 0-1-2-3
    assert len({uf.find(i) for i in range(4)}) == 1   # one component
    assert uf.find(3) == 0                      # min-index root


def test_streaming_dedup_larger_corpus():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    import pyarrow as pa
    from jude import curate
    from jude.curate_dist import dist_fuzzy_dedup
    from jude.runners.ray import RayRunner

    # many exact + near duplicates across a bigger corpus -> streamed edges
    base = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(40)]
    docs = (base + base + [b + " today" for b in base]) * 2   # heavy dup structure
    t = pa.table({"text": docs})
    params = dict(column="text", threshold=0.6, num_hashes=64, ngram=2, seed=1)
    single = curate.fuzzy_dedup(t, **params)
    dist = dist_fuzzy_dedup(t, runner=RayRunner(num_workers=4), **params)
    assert dist.num_rows == single.num_rows      # streaming UF == single-node count
