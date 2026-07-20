"""L5.4: corpus profiling — HLL approximate cardinality + length/lang stats."""

from __future__ import annotations

import pyarrow as pa

from jude.profile import HyperLogLog, profile


def test_hll_accuracy():
    h = HyperLogLog()
    for i in range(10000):
        h.add(f"item-{i}")
    est = h.estimate()
    assert abs(est - 10000) / 10000 < 0.03      # ~1% std error, allow 3%


def test_hll_small_cardinality():
    h = HyperLogLog()
    for x in ["a", "b", "c", "a", "b"]:
        h.add(x)
    assert abs(h.estimate() - 3) <= 1           # linear counting handles small n


def test_profile_dup_rate():
    docs = ["hello world"] * 30 + [f"unique doc {i}" for i in range(70)]
    p = profile(pa.table({"text": docs}))
    assert p["rows"] == 100
    assert abs(p["approx_distinct"] - 71) <= 2  # 1 (dup group) + 70 unique
    assert 0.2 < p["dup_rate"] < 0.4


def test_profile_length_percentiles():
    docs = ["a"] * 60 + ["a much longer document with many words here now"] * 40
    p = profile(pa.table({"text": docs}))
    assert p["words"]["p50"] == 1               # median (60% short) is the short doc
    assert p["words"]["max"] >= 9
    assert p["chars"]["mean"] > 1


def test_profile_language_histogram():
    docs = ["the cat and the dog"] * 3 + ["这是中文文本内容"] * 2
    p = profile(pa.table({"text": docs}), langs=True)
    assert "langs" in p
    assert p["langs"].get("en", 0) == 3 and p["langs"].get("zh", 0) == 2


def test_profile_empty():
    p = profile(pa.table({"text": pa.array([], type=pa.string())}))
    assert p["rows"] == 0 and p["dup_rate"] == 0.0
