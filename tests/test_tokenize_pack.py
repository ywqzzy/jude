"""P1/L4: tokenize -> pack -> write token shards (training-ready output).

Uses the built-in zero-dep byte tokenizer so it runs without HF/tiktoken. Verifies
token conservation, EOS placement, seq_len packing, doc-boundary tracking, and
both Lance and flat .bin/.idx.json outputs (memory-mappable by a training loader).
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pyarrow as pa
import pytest

from jude import tokenize as tk


def test_tokenize_bytes_with_eos():
    t = pa.table({"text": ["ab", "hello"]})
    out = tk.tokenize(t, tokenizer="bytes")
    ids = out.column("input_ids").to_pylist()
    assert ids[0] == [97, 98, tk.BYTE_EOS]            # 'a','b',EOS
    assert ids[1] == list(b"hello") + [tk.BYTE_EOS]
    assert out.column("n_tokens").to_pylist() == [3, 6]


def test_tokenize_callable():
    t = pa.table({"text": ["one two three"]})
    out = tk.tokenize(t, tokenizer=lambda s: [len(w) for w in s.split()], add_eos=False)
    assert out.column("input_ids").to_pylist() == [[3, 3, 5]]


def test_pack_sequences_conserves_and_windows():
    # 3 docs -> tokenize -> pack into seq_len=4
    t = pa.table({"text": ["aaaa", "bb", "cccccc"]})  # 4+1,2+1,6+1 EOS = 15 tokens
    tok = tk.tokenize(t, tokenizer="bytes")
    total = sum(tok.column("n_tokens").to_pylist())
    packed = tk.pack_sequences(tok, seq_len=4, drop_remainder=True)
    # 15 tokens -> 3 full windows of 4 (last 3 dropped)
    assert packed.num_rows == total // 4
    for seq in packed.column("input_ids").to_pylist():
        assert len(seq) == 4
    # doc_ids parallel + same shape
    assert all(len(d) == 4 for d in packed.column("doc_ids").to_pylist())


def test_pack_pad_remainder():
    t = pa.table({"text": ["abc"]})                    # 3 + EOS = 4 tokens
    tok = tk.tokenize(t, tokenizer="bytes")
    packed = tk.pack_sequences(tok, seq_len=8, drop_remainder=False, pad_id=0)
    assert packed.num_rows == 1
    seq = packed.column("input_ids")[0].as_py()
    assert len(seq) == 8 and seq[4:] == [0, 0, 0, 0]   # padded
    docs = packed.column("doc_ids")[0].as_py()
    assert docs[4:] == [-1, -1, -1, -1]                # pad marked with doc -1


def test_write_bin_roundtrip_memmap():
    t = pa.table({"text": ["hello world", "foo bar baz"]})
    tok = tk.tokenize(t, tokenizer="bytes")
    packed = tk.pack_sequences(tok, seq_len=4, drop_remainder=True)
    path = tempfile.mkdtemp() + "/shard"
    meta = tk.write_token_shards(packed, path, fmt="bin", dtype="int32")
    assert meta["seq_len"] == 4 and meta["num_sequences"] == packed.num_rows
    # loader path: memmap the flat .bin and reshape to [n, seq_len]
    arr = np.memmap(path + ".bin", dtype="int32", mode="r").reshape(-1, 4)
    assert arr.shape[0] == packed.num_rows
    assert arr.tolist() == packed.column("input_ids").to_pylist()  # bit-identical
    with open(path + ".idx.json") as f:
        assert json.load(f)["total_tokens"] == arr.size


def test_write_lance_roundtrip():
    lance = pytest.importorskip("lance")
    import jude

    t = pa.table({"text": ["alpha beta", "gamma"]})
    packed = tk.pack_sequences(tk.tokenize(t, tokenizer="bytes"), seq_len=4, drop_remainder=False)
    path = tempfile.mkdtemp() + "/toks"
    info = tk.write_token_shards(packed, path, fmt="lance")
    assert info["sequences"] == packed.num_rows
    back = jude._lance.read_table(path)
    assert back.num_rows == packed.num_rows
    assert "input_ids" in back.column_names
