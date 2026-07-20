"""L4.5: token-aware length gate — filter by token count, not word count."""

from __future__ import annotations

import pyarrow as pa

from jude import curate


def test_filter_by_token_count_bytes():
    # byte tokenizer: token count == byte length
    t = pa.table({"text": ["hi", "a longer document here", "x"]})
    out = curate.token_length_filter(t, tokenizer="bytes", min_tokens=5)
    kept = out.column("text").to_pylist()
    assert kept == ["a longer document here"]        # only the >=5-token doc


def test_max_tokens():
    t = pa.table({"text": ["short", "this one is quite a bit longer than the others"]})
    out = curate.token_length_filter(t, tokenizer="bytes", max_tokens=10)
    assert out.column("text").to_pylist() == ["short"]


def test_count_column_annotates():
    t = pa.table({"text": ["ab", "abc"]})
    out = curate.token_length_filter(t, tokenizer="bytes", count_column="ntok")
    assert out.num_rows == 2 and out.column("ntok").to_pylist() == [2, 3]


def test_callable_tokenizer():
    t = pa.table({"text": ["one two", "one two three four five"]})
    out = curate.token_length_filter(t, tokenizer=lambda s: s.split(), min_tokens=3)
    assert out.column("text").to_pylist() == ["one two three four five"]
