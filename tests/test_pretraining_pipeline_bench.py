"""X.4: the end-to-end pretraining pipeline harness produces a sane funnel +
training-ready token shards (real PB/cluster numbers await infra; this verifies
the whole chain composes and the report is coherent)."""

from __future__ import annotations

from benchmarking.bench_pretraining_pipeline import run


def test_pipeline_end_to_end_funnel():
    r = run(docs=800, seq_len=128)
    assert 0 < r["kept_docs"] <= 800
    assert r["keep_rate"] < 1.0                       # junk/dupes/non-EN were dropped
    assert r["packed_sequences"] > 0 and r["total_tokens"] > 0
    assert r["shards"] >= 1
    # funnel is monotonic non-increasing through the filter stages
    filt = [f for f in r["funnel"] if "rows_out" in f]
    for f in filt:
        assert f["rows_out"] <= f["rows_in"]
    # every filter stage recorded a time + row counts
    names = [f["stage"] for f in r["funnel"]]
    assert names[:4] == ["language_filter", "quality_filter", "exact_dedup", "fuzzy_dedup"]
    assert names[-1] == "tokenize+pack+shard"


def test_pipeline_deterministic():
    a = run(docs=500, seq_len=128)
    b = run(docs=500, seq_len=128)
    assert a["kept_docs"] == b["kept_docs"]           # synthetic corpus + ops are seeded
    assert a["total_tokens"] == b["total_tokens"]
