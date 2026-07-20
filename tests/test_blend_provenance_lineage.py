"""L5.1-L5.3: token-ratio blend + provenance + reproducibility lineage."""

from __future__ import annotations

import tempfile

import pyarrow as pa

from jude import curate, lineage


def test_add_provenance():
    t = pa.table({"text": ["a", "b"]})
    out = curate.add_provenance(t, "web")
    assert out.column("_source").to_pylist() == ["web", "web"]


def test_blend_by_tokens_ratio_and_provenance():
    # source A: short docs; source B: long docs. Blend 50/50 BY TOKENS.
    a = pa.table({"text": ["x"] * 200})                          # ~1 token each (bytes)
    b = pa.table({"text": ["a much longer document here indeed"] * 200})
    out = curate.blend_by_tokens([a, b], [0.5, 0.5], tokenizer="bytes",
                                 source_names=["A", "B"], seed=0)
    srcs = out.column("_source").to_pylist()
    # token budget is split 50/50, so B (long docs) contributes FAR fewer rows
    # than A to reach the same token count -> row counts differ, tokens balance.
    from collections import Counter
    c = Counter(srcs)
    assert c["A"] > c["B"]                                       # more short docs for equal tokens
    assert set(srcs) == {"A", "B"}


def test_blend_by_tokens_upsamples_small_source():
    big = pa.table({"text": ["word " * 20] * 100})
    small = pa.table({"text": ["tiny"]})                         # 1 doc, must upsample
    out = curate.blend_by_tokens([big, small], [0.5, 0.5], tokenizer="bytes",
                                 source_names=["big", "small"], seed=1)
    from collections import Counter
    c = Counter(out.column("_source").to_pylist())
    assert c["small"] > 1                                        # upsampled to hit its quota


def test_pipeline_signature_stable_and_sensitive():
    cfg1 = {"dedup": "fuzzy", "threshold": 0.7, "stages": ["quality", "dedup"]}
    cfg2 = {"stages": ["quality", "dedup"], "threshold": 0.7, "dedup": "fuzzy"}  # reordered
    assert lineage.pipeline_signature(cfg1) == lineage.pipeline_signature(cfg2)   # order-insensitive
    cfg3 = {**cfg1, "threshold": 0.8}
    assert lineage.pipeline_signature(cfg1) != lineage.pipeline_signature(cfg3)   # value-sensitive


def test_lineage_sidecar_roundtrip():
    cfg = {"stages": ["quality_filter", "fuzzy_dedup"], "threshold": 0.7}
    lin = lineage.dataset_lineage(cfg, inputs={"web": 3, "code": 7}, output_version=12)
    assert lin["pipeline_signature"] == lineage.pipeline_signature(cfg)
    path = tempfile.mkdtemp() + "/train.lance"
    lineage.write_lineage(path, lin)
    back = lineage.read_lineage(path)
    assert back["inputs"] == {"web": 3, "code": 7}
    assert back["output_version"] == 12
    assert lineage.read_lineage(tempfile.mkdtemp() + "/nope") is None
