"""L2.3: incremental dedup — dedup a new dump against a persistent hash index."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

from jude.incremental_dedup import HashIndex, incremental_dedup


def test_in_memory_add_returns_novel_only():
    idx = HashIndex()
    d1 = pa.table({"text": ["a", "b", "c"]})
    assert idx.add(d1).num_rows == 3            # all novel first time
    d2 = pa.table({"text": ["b", "c", "d", "e"]})  # b,c seen; d,e new
    novel = idx.add(d2)
    assert sorted(novel.column("text").to_pylist()) == ["d", "e"]
    assert len(idx) == 5                         # a,b,c,d,e


def test_within_batch_dedup():
    idx = HashIndex()
    out = idx.add(pa.table({"text": ["x", "x", "y"]}))
    assert sorted(out.column("text").to_pylist()) == ["x", "y"]  # first occurrence wins


def test_normalize_matches_case_space():
    idx = HashIndex(normalize=True)
    idx.add(pa.table({"text": ["Hello World"]}))
    out = idx.add(pa.table({"text": ["  hello   world  ", "new one"]}))
    assert out.column("text").to_pylist() == ["new one"]   # normalized dup dropped


def test_persist_and_reload_across_runs():
    lance = pytest.importorskip("lance")
    path = tempfile.mkdtemp() + "/dedup_index.lance"
    # run 1: dump A
    novel1 = incremental_dedup(pa.table({"text": ["doc1", "doc2"]}), path)
    assert novel1.num_rows == 2
    # run 2 (fresh index object, loads persisted hashes): dump B overlaps A
    novel2 = incremental_dedup(pa.table({"text": ["doc2", "doc3"]}), path)
    assert novel2.column("text").to_pylist() == ["doc3"]   # doc2 already in prior run
    # index now holds 3 distinct docs
    assert len(HashIndex(path)) == 3


def test_contains_no_mutation():
    idx = HashIndex()
    idx.add(pa.table({"text": ["seen"]}))
    flags = idx.contains(pa.table({"text": ["seen", "unseen"]}))
    assert flags == [True, False]
    assert len(idx) == 1                         # contains() didn't add
