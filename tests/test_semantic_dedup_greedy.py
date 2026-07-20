"""A5: semantic dedup is GREEDY and non-transitive.

The old semantic_dedup used connected-components over a cosine>=threshold graph,
so an A~B~C chain (A close to B, B close to C, but A NOT close to C) collapsed
all three into one cluster — over-deduplicating. Real SemDeDup drops a row only
when it is within threshold of an already-KEPT survivor, so the chain keeps A
and C.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from jude import curate


def test_non_transitive_chain_keeps_endpoints():
    # unit vectors at 0, 40, 80 degrees: neighbors (40 deg apart) are similar,
    # but the endpoints (80 deg apart) are not -> should NOT collapse to one.
    def unit(deg):
        r = np.radians(deg)
        return [float(np.cos(r)), float(np.sin(r))]

    embs = [unit(0), unit(40), unit(80)]
    # cos(40 deg)=0.766 (>=0.7 -> near), cos(80 deg)=0.174 (<0.7 -> far)
    t = pa.table({"id": [1, 2, 3], "embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.7)
    kept = out.column("id").to_pylist()
    # greedy: keep 1; 2 within 0.7 of survivor 1 -> drop; 3 NOT within 0.7 of
    # survivor 1 (only of dropped 2) -> keep. Transitive closure would keep only 1.
    assert kept == [1, 3]


def test_exact_and_near_dups_still_collapse():
    embs = [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]]
    t = pa.table({"id": [1, 2, 3], "embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9)
    assert out.column("id").to_pylist() == [1, 3]


def test_null_embedding_treated_distinct():
    embs = [[1.0, 0.0], [1.0, 0.0], None]
    t = pa.table({"id": [1, 2, 3], "embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9)
    kept = out.column("id").to_pylist()
    assert 1 in kept and 3 in kept and 2 not in kept  # null row kept as distinct


def test_cluster_labels_map_to_survivor():
    embs = [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]
    t = pa.table({"embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9, keep_cluster=True)
    reps = out.column("sem_cluster").to_pylist()
    assert reps[0] == reps[1] == 0   # row 1 maps to survivor 0
    assert reps[2] == 2              # distinct
