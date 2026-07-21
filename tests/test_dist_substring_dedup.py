"""L2.2: distributed exact-substring dedup — must match single-node
curate.substring_dedup (shuffle window hashes, global min-(doc,pos) keeper)."""

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


def test_dist_matches_single_node_shared_passage():
    from jude.curate_dist import dist_substring_dedup

    passage = " ".join(f"w{i}" for i in range(60))
    docs = [f"alpha start {passage} alpha end",
            f"beta lead {passage} beta tail",
            "totally unrelated document with no shared span at all here"]
    t = pa.table({"text": docs})
    single = curate.substring_dedup(t, k=50)
    dist = dist_substring_dedup(t, k=50)
    assert dist.column("text").to_pylist() == single.column("text").to_pylist()


def test_dist_matches_single_node_random_corpus():
    from jude.curate_dist import dist_substring_dedup
    import numpy as np

    rng = np.random.default_rng(0)
    shared = " ".join(f"s{i}" for i in range(55))
    docs = []
    for i in range(40):
        body = " ".join(f"u{i}_{j}" for j in range(rng.integers(20, 60)))
        # ~half the docs embed the shared passage in varying positions
        docs.append(f"{body} {shared} {body}" if i % 2 == 0 else body)
    t = pa.table({"text": docs, "id": list(range(len(docs)))})
    single = curate.substring_dedup(t, k=50)
    dist = dist_substring_dedup(t, k=50)
    assert dist.column("text").to_pylist() == single.column("text").to_pylist()
    assert dist.column("id").to_pylist() == list(range(len(docs)))   # rows preserved/aligned


def test_dist_no_shared_span_noop():
    from jude.curate_dist import dist_substring_dedup

    docs = [" ".join(f"a{i}" for i in range(60)), " ".join(f"b{i}" for i in range(60))]
    t = pa.table({"text": docs})
    out = dist_substring_dedup(t, k=50)
    assert out.column("text").to_pylist() == docs   # nothing repeated -> unchanged
