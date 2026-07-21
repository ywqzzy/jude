"""L2.1 follow-up: distributed label-propagation connected-components (label
array sharded across workers, not held whole on the driver)."""

from __future__ import annotations

import pyarrow as pa
import pytest

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def _ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield


def _edges(pairs):
    return ray.put(pa.table({"a": pa.array([x for x, _ in pairs], type=pa.int64()),
                             "b": pa.array([y for _, y in pairs], type=pa.int64())}))


def test_cc_transitive_and_components():
    from jude.dist_cc import connected_components
    # {0,1,2,3} chain across separate edge shards + {5,6}; 8 has no edge (singleton)
    refs = [_edges([(3, 1), (1, 2)]), _edges([(2, 0)]), _edges([(6, 5)])]
    label = connected_components(refs, num_workers=3)
    # component-min semantics: merged rids map to the smallest id; reps absent
    assert label[1] == 0 and label[2] == 0 and label[3] == 0
    assert label[6] == 5
    assert 0 not in label and 5 not in label       # representatives aren't listed
    assert 8 not in label                          # untouched singleton


def test_cc_matches_union_find_on_random_graph():
    import random
    from jude.curate_dist import _UnionFind
    from jude.dist_cc import connected_components

    rng = random.Random(0)
    n = 400
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(300)]
    pairs = [(a, b) for a, b in pairs if a != b]
    # reference: single-machine union-by-min
    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)
    ref_reps = [uf.find(i) for i in range(n)]
    # distributed CC, edges split across several refs
    refs = [_edges(pairs[i::5]) for i in range(5)]
    label = connected_components(refs, num_workers=4)
    dist_reps = [label.get(i, i) for i in range(n)]
    assert dist_reps == ref_reps                   # identical component-min labeling


def test_dist_fuzzy_dedup_cc_workers_matches_single_node():
    import pyarrow as pa
    from jude import curate
    from jude.curate_dist import dist_fuzzy_dedup
    from jude.runners.ray import RayRunner

    base = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(30)]
    docs = base + base + [b + " today" for b in base]      # dup + near-dup structure
    t = pa.table({"text": docs})
    params = dict(column="text", threshold=0.6, num_hashes=64, ngram=2, seed=1)
    single = curate.fuzzy_dedup(t, **params)
    dist = dist_fuzzy_dedup(t, cc_workers=3, runner=RayRunner(num_workers=4), **params)
    assert dist.num_rows == single.num_rows        # distributed label-prop == single-node
