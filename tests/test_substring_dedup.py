"""C1b: exact-substring dedup (Lee et al. rolling-hash form). Repeated passages of
>= k tokens are removed after their first occurrence, even when the surrounding
documents differ (which whole-document dedup misses)."""

from __future__ import annotations

import pyarrow as pa

from jude import curate


def test_shared_passage_removed_after_first():
    passage = " ".join(f"w{i}" for i in range(60))          # a 60-token shared block
    docs = [
        f"unique alpha start {passage} unique alpha end",
        f"different beta lead {passage} different beta tail",
    ]
    out = curate.substring_dedup(pa.table({"text": docs}), k=50)
    d0, d1 = out.column("text").to_pylist()
    # first doc keeps the passage; second has it stripped, its unique text intact
    assert "w30" in d0
    assert "w30" not in d1
    assert "different beta lead" in d1 and "different beta tail" in d1


def test_short_docs_untouched():
    docs = ["short one here", "short two there"]
    out = curate.substring_dedup(pa.table({"text": docs}), k=50)
    assert out.column("text").to_pylist() == docs   # < k tokens -> unchanged


def test_no_shared_passage_keeps_all():
    docs = [" ".join(f"a{i}" for i in range(60)), " ".join(f"b{i}" for i in range(60))]
    out = curate.substring_dedup(pa.table({"text": docs}), k=50)
    kept = out.column("text").to_pylist()
    assert "a30" in kept[0] and "b30" in kept[1]     # nothing repeated -> nothing removed


def test_intra_document_repeat_removed():
    block = " ".join(f"t{i}" for i in range(55))
    doc = f"{block} bridge text between {block}"        # same block twice in one doc
    out = curate.substring_dedup(pa.table({"text": [doc]}), k=50)
    # the second copy is stripped; the first + the bridge remain
    text = out.column("text")[0].as_py()
    assert "bridge text between" in text
    assert text.count("t54") == 1


def test_out_column():
    passage = " ".join(f"z{i}" for i in range(60))
    out = curate.substring_dedup(pa.table({"text": [passage, passage]}),
                                 k=50, out_column="deduped")
    assert "deduped" in out.column_names and "text" in out.column_names
    assert out.column("deduped")[1].as_py() == ""   # second doc fully duplicated -> emptied
