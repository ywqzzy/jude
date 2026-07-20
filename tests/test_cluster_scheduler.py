"""Cross-query cluster placement (jude.dist.ClusterScheduler) — worst-fit
bin-packing of actor/task bundles onto Ray node capacities, decided in Rust."""
from collections import Counter

import pytest

import jude


def test_balances_gpu_bundles():
    s = jude.dist.ClusterScheduler([("A", 8.0, 2.0, 0), ("B", 8.0, 2.0, 0)])
    place = s.place([(0.0, 1.0, 0)] * 4)
    assert Counter(place) == {"A": 2, "B": 2}
    assert s.overflow_count == 0


def test_overflow_when_no_fit():
    s = jude.dist.ClusterScheduler([("A", 0.0, 1.0, 0)])
    place = s.place([(0.0, 1.0, 0), (0.0, 1.0, 0)])
    assert place == ["A", "A"]
    assert s.overflow_count == 1


def test_memory_packs_apart():
    s = jude.dist.ClusterScheduler([("A", 0.0, 0.0, 100), ("B", 0.0, 0.0, 100)])
    place = s.place([(0.0, 0.0, 60), (0.0, 0.0, 60)])
    assert place[0] != place[1]


def test_reset():
    s = jude.dist.ClusterScheduler([("A", 0.0, 1.0, 0)])
    s.place([(0.0, 1.0, 0)])
    s.reset()
    assert s.overflow_count == 0
    assert s.place([(0.0, 1.0, 0)]) == ["A"]


def test_empty_nodes_raises():
    s = jude.dist.ClusterScheduler([])
    with pytest.raises(Exception):
        s.place([(0.0, 1.0, 0)])
