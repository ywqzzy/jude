"""jude.tokenize — the last mile: cleaned text → training-ready token shards.

An LLM data pipeline's real output isn't clean text, it's **tokenized, packed
sequences** a training loader can memory-map. This module closes that gap:

    tokenize(table)            # text column -> input_ids (list<int32>) + n_tokens
      -> pack_sequences(...)   # concat docs, split into fixed seq_len windows
      -> write_token_shards()  # Lance (queryable/versioned) or flat .bin + .idx.json

The tokenizer is **pluggable** and needs no heavy dependency by default:
- ``"bytes"`` — built-in UTF-8 byte tokenizer (zero-dep; a real byte-level
  vocab of 0..255 + EOS=256, and what the tests use);
- any ``Callable[[str], list[int]]`` — bring your own (HF ``tokenizers``,
  ``tiktoken``, sentencepiece, …);
- a string name — lazily resolved via ``tiktoken`` then HF ``tokenizers`` if
  installed, else a clear error (no hard dependency).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pyarrow as pa

BYTE_EOS = 256  # one past the 0..255 byte range → EOS for the byte tokenizer


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8", "ignore"))


def _resolve_tokenizer(tokenizer: Any) -> tuple[Callable[[str], list[int]], int | None]:
    """Return (encode_fn, default_eos_id). ``default_eos`` is the tokenizer's
    natural EOS if known, else None."""
    if callable(tokenizer):
        return tokenizer, None
    if tokenizer in ("bytes", "byte"):
        return _byte_encode, BYTE_EOS
    # named tokenizer — resolve lazily, no hard dependency.
    try:  # tiktoken (e.g. "gpt2", "cl100k_base")
        import tiktoken

        enc = tiktoken.get_encoding(tokenizer) if tokenizer in tiktoken.list_encoding_names() \
            else tiktoken.encoding_for_model(tokenizer)
        return (lambda t: enc.encode(t or "")), getattr(enc, "eot_token", None)
    except Exception:  # noqa: BLE001
        pass
    try:  # HF tokenizers
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(tokenizer)
        return (lambda t: tok.encode(t or "").ids), None
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"cannot resolve tokenizer {tokenizer!r}: install tiktoken or "
            f"tokenizers, pass a callable, or use 'bytes' ({e})"
        ) from e


def tokenize(
    table: pa.Table,
    *,
    column: str = "text",
    out_column: str = "input_ids",
    tokenizer: Any = "bytes",
    add_eos: bool = True,
    eos_id: int | None = None,
    n_tokens_column: str | None = "n_tokens",
) -> pa.Table:
    """Tokenize a text column into an ``input_ids`` (``list<int32>``) column,
    optionally appending an EOS token per document and an ``n_tokens`` count."""
    enc, default_eos = _resolve_tokenizer(tokenizer)
    eos = eos_id if eos_id is not None else default_eos
    ids_col: list = []
    ntok: list = []
    for t in table.column(column).to_pylist():
        ids = list(enc(t or ""))
        if add_eos and eos is not None:
            ids = ids + [int(eos)]
        ids_col.append(ids)
        ntok.append(len(ids))
    out = table.append_column(out_column, pa.array(ids_col, type=pa.list_(pa.int32())))
    if n_tokens_column:
        out = out.append_column(n_tokens_column, pa.array(ntok, type=pa.int64()))
    return out


def pack_sequences(
    table: pa.Table,
    *,
    id_column: str = "input_ids",
    seq_len: int = 1024,
    drop_remainder: bool = True,
    pad_id: int = 0,
    doc_column: str | None = "doc_ids",
) -> pa.Table:
    """Concatenate every document's tokens and split into fixed ``seq_len``
    windows — the standard pretraining packing (no padding waste; documents flow
    across window boundaries). Emits one row per packed sequence: ``input_ids``
    (``fixed_size_list<int32, seq_len>``) and, if ``doc_column`` is set, a
    parallel ``doc_ids`` marking which source document each token came from (for
    building an intra-document attention mask). A trailing partial window is
    dropped (``drop_remainder``) or padded with ``pad_id`` (doc id -1)."""
    all_tok: list[int] = []
    all_doc: list[int] = []
    for di, ids in enumerate(table.column(id_column).to_pylist()):
        if not ids:
            continue
        all_tok.extend(ids)
        all_doc.extend([di] * len(ids))
    n = len(all_tok)
    seqs: list = []
    docs: list = []
    for i in range(0, n, seq_len):
        chunk = all_tok[i : i + seq_len]
        dchunk = all_doc[i : i + seq_len]
        if len(chunk) < seq_len:
            if drop_remainder:
                break
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
            dchunk = dchunk + [-1] * (seq_len - len(dchunk))
        seqs.append(chunk)
        docs.append(dchunk)
    cols = {id_column: pa.array(seqs, type=pa.list_(pa.int32(), seq_len))}
    if doc_column:
        cols[doc_column] = pa.array(docs, type=pa.list_(pa.int32(), seq_len))
    return pa.table(cols)


def write_token_shards(
    table: pa.Table,
    path: str,
    *,
    column: str = "input_ids",
    fmt: str = "lance",
    dtype: str = "int32",
    mode: str = "create",
) -> dict:
    """Write packed token sequences as training shards.

    ``fmt="lance"``: a Lance dataset (queryable, versioned) with the ``input_ids``
    list column — jude's native, inspectable output. ``fmt="bin"``: a flat
    little-endian ``path.bin`` token stream + ``path.idx.json`` sidecar
    (dtype / seq_len / num_sequences / total_tokens), memory-mappable by a
    training loader (``np.memmap(path.bin, dtype).reshape(-1, seq_len)``)."""
    if fmt == "lance":
        from jude import _lance

        info = _lance.write(table, path, mode=mode)
        return {"format": "lance", **info, "sequences": table.num_rows}
    if fmt == "bin":
        import numpy as np

        col = table.column(column).combine_chunks()
        seq_len = col.type.list_size if pa.types.is_fixed_size_list(col.type) else None
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        with open(path + ".bin", "wb") as f:
            flat.tofile(f)
        meta = {"format": "bin", "dtype": dtype, "seq_len": seq_len,
                "num_sequences": table.num_rows, "total_tokens": int(flat.size)}
        with open(path + ".idx.json", "w") as f:
            json.dump(meta, f)
        return meta
    raise ValueError(f"unknown fmt {fmt!r}; use 'lance' or 'bin'")


def write_shuffled_shards(
    table: pa.Table,
    path: str,
    *,
    column: str = "input_ids",
    n_shards: int = 8,
    fmt: str = "bin",
    dtype: str = "int32",
    seed: int = 0,
) -> list[dict]:
    """Globally shuffle packed sequences and write them across ``n_shards`` shards
    (L4.4) — the training-loader layout: a random order so consecutive training
    steps see decorrelated sequences, split into shard files for parallel loading.
    Deterministic given ``seed``. Writes ``path.00000``, ``path.00001``, … (each a
    Lance dataset or a .bin/.idx.json pair)."""
    import numpy as np

    n = table.num_rows
    perm = np.random.default_rng(seed).permutation(n)
    shuffled = table.take(pa.array(perm.tolist(), type=pa.int64()))
    ns = max(1, int(n_shards))
    per = (n + ns - 1) // ns
    out: list[dict] = []
    for s in range(ns):
        lo = s * per
        if lo >= n:
            break
        part = shuffled.slice(lo, min(per, n - lo))
        shard_path = f"{path}.{s:05d}"
        out.append(write_token_shards(part, shard_path, column=column, fmt=fmt, dtype=dtype))
    return out
