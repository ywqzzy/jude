"""X.3 sample_dropped (inspect what a filter removed) + L4.4 shuffled token shards."""

from __future__ import annotations

import glob
import tempfile

import numpy as np
import pyarrow as pa

from jude import curate
from jude import tokenize as tk


# --- X.3 ---------------------------------------------------------------------

def test_sample_dropped_shows_removed_rows():
    good = ("This is a genuinely long and clean paragraph of prose that comfortably "
            "exceeds the minimum word count required by the quality filter, with "
            "plenty of ordinary stopwords and varied vocabulary so it is not flagged "
            "as spam or boilerplate by any of the Gopher-style heuristic checks here, "
            "and it continues for several more words to be sure it clears the fifty "
            "word floor that the default quality gate imposes on every input document.")
    before = pa.table({"text": [good, "aa", "!!!"]})
    after = curate.quality_filter(before)              # drops the junk rows
    dropped = curate.sample_dropped(before, after)
    dvals = set(dropped.column("text").to_pylist())
    assert "aa" in dvals and "!!!" in dvals            # the dropped junk is surfaced
    assert good not in dvals                            # the survivor is not shown


def test_sample_dropped_multiplicity():
    before = pa.table({"text": ["dup", "dup", "dup", "uniq"]})
    after = curate.exact_dedup(before)                 # 3 "dup" -> 1
    dropped = curate.sample_dropped(before, after)
    assert dropped.num_rows == 2                        # 2 of the 3 duplicates dropped


def test_sample_dropped_caps_n():
    before = pa.table({"text": [f"junk{i}" for i in range(100)]})
    after = before.slice(0, 0)                          # everything dropped
    dropped = curate.sample_dropped(before, after, n=10)
    assert dropped.num_rows == 10


# --- L4.4 --------------------------------------------------------------------

def test_shuffled_shards_cover_all_tokens():
    t = pa.table({"text": [f"document number {i} with some words" for i in range(40)]})
    packed = tk.pack_sequences(tk.tokenize(t, tokenizer="bytes"), seq_len=8, drop_remainder=True)
    path = tempfile.mkdtemp() + "/shard"
    metas = tk.write_shuffled_shards(packed, path, n_shards=4, fmt="bin", seed=0)
    assert len(metas) >= 1
    # every sequence lands in exactly one shard; union == original multiset
    all_rows = []
    for f in sorted(glob.glob(path + ".*.bin")):
        seq_len = 8
        all_rows.extend(np.memmap(f, dtype="int32", mode="r").reshape(-1, seq_len).tolist())
    orig = packed.column("input_ids").to_pylist()
    assert sorted(all_rows) == sorted(orig)            # no loss, no dup — just reordered


def test_shuffled_shards_deterministic():
    t = pa.table({"text": [f"doc {i}" for i in range(20)]})
    packed = tk.pack_sequences(tk.tokenize(t, tokenizer="bytes"), seq_len=4, drop_remainder=False)
    p1 = tempfile.mkdtemp() + "/a"
    p2 = tempfile.mkdtemp() + "/b"
    tk.write_shuffled_shards(packed, p1, n_shards=3, fmt="bin", seed=7)
    tk.write_shuffled_shards(packed, p2, n_shards=3, fmt="bin", seed=7)
    a = np.memmap(p1 + ".00000.bin", dtype="int32", mode="r").tolist()
    b = np.memmap(p2 + ".00000.bin", dtype="int32", mode="r").tolist()
    assert a == b                                       # same seed -> same shuffle
