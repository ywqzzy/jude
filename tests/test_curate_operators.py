"""Curation operators: chunking, exact dedup, quality signals/filter, blend,
global shuffle, content hash — behavioral coverage of the map-style curators."""

from __future__ import annotations

import pyarrow as pa

from jude import curate


# --- chunk_text --------------------------------------------------------------

def test_chunk_text_splits_and_indexes():
    t = pa.table({"id": [1], "text": ["word " * 300]})  # ~1500 chars
    out = curate.chunk_text(t, chunk_chars=200)
    assert out.num_rows > 1
    assert "chunk" in out.column_names and "chunk_index" in out.column_names
    # chunk indices are 0..k-1 within the source row
    idxs = out.column("chunk_index").to_pylist()
    assert idxs == list(range(len(idxs)))
    # every chunk is within the char budget (allowing a small word-boundary slack)
    assert all(len(c) <= 260 for c in out.column("chunk").to_pylist())


def test_chunk_text_short_input_single_chunk():
    t = pa.table({"text": ["a short doc"]})
    out = curate.chunk_text(t, chunk_chars=1024)
    assert out.num_rows == 1


# --- exact_dedup + content hash ----------------------------------------------

def test_exact_dedup_normalized():
    t = pa.table({"text": ["Hello World", "hello world", "HELLO WORLD", "different"]})
    out = curate.exact_dedup(t)          # normalize=True -> case/space-insensitive
    assert out.num_rows == 2             # one "hello world" + "different"


def test_exact_dedup_case_sensitive():
    t = pa.table({"text": ["Hello", "hello", "Hello"]})
    out = curate.exact_dedup(t, normalize=False)
    assert out.num_rows == 2             # "Hello" and "hello" distinct


def test_content_hash_stable_and_distinct():
    t = pa.table({"text": ["same", "same", "other"]})
    out = curate.add_content_hash(t)
    h = out.column("content_hash").to_pylist()
    assert h[0] == h[1] and h[0] != h[2]


# --- quality signals / filter ------------------------------------------------

def test_quality_signals_columns_present():
    t = pa.table({"text": ["The quick brown fox jumps over the lazy dog."]})
    out = curate.quality_signals(t)
    for c in ("q_char_count", "q_word_count", "q_alpha_ratio", "q_dup_line_ratio"):
        assert c in out.column_names
    assert out.column("q_word_count")[0].as_py() == 9


def test_quality_filter_drops_low_quality():
    good = ("Machine learning systems increasingly rely on carefully curated "
            "training corpora, where deduplication, quality filtering, and "
            "decontamination each remove a distinct class of harmful examples "
            "before any model ever sees the data during its optimization phase, "
            "which materially improves downstream evaluation accuracy and reduces "
            "memorization of sensitive personal information across many domains.")
    t = pa.table({"text": [good, "aaa aaa aaa", "!!!! ????"]})
    out = curate.quality_filter(t)
    assert out.num_rows == 1
    assert out.column("text")[0].as_py().startswith("Machine learning")


def test_quality_filter_reason_annotates_all():
    t = pa.table({"text": ["short", "also short"]})
    out = curate.quality_filter(t, reason_column="why")
    assert out.num_rows == 2                         # annotate keeps all
    assert all(r for r in out.column("why").to_pylist())  # each has a reason


# --- blend + shuffle ---------------------------------------------------------

def test_blend_datasets_total_rows():
    a = pa.table({"x": [1, 2, 3]})
    b = pa.table({"x": [10, 11, 12, 13, 14]})
    out = curate.blend_datasets([a, b], weights=[0.5, 0.5], total_rows=8, seed=0)
    assert out.num_rows == 8


def test_global_shuffle_preserves_multiset():
    t = pa.table({"x": list(range(50))})
    out = curate.global_shuffle(t, seed=1)
    assert sorted(out.column("x").to_pylist()) == list(range(50))  # no rows lost
    assert out.column("x").to_pylist() != list(range(50))          # order changed


def test_global_shuffle_deterministic_with_seed():
    t = pa.table({"x": list(range(50))})
    a = curate.global_shuffle(t, seed=7).column("x").to_pylist()
    b = curate.global_shuffle(t, seed=7).column("x").to_pylist()
    assert a == b   # same seed -> same permutation
