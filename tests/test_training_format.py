"""Training-format writers (C8): WebDataset .tar, Mosaic MDS, sharded Parquet."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from jude import training_format as tf


def _tmp():
    return tempfile.mkdtemp(prefix="jude_tf_")


def test_sharded_parquet_sizes():
    t = pa.table({"x": list(range(25))})
    d = _tmp()
    info = tf.write_sharded_parquet(t, d, rows_per_shard=10)
    assert info["num_shards"] == 3  # 10,10,5
    assert [s["rows"] for s in info["shards"]] == [10, 10, 5]
    # round-trip row count
    total = sum(pq.read_table(s["path"]).num_rows for s in info["shards"])
    assert total == 25


def test_webdataset_records():
    t = pa.table({
        "id": [1, 2, 3],
        "caption": ["a cat", "a dog", "a bird"],
        "image": [b"\x01\x02", b"\x03\x04", b"\x05\x06"],
        "score": [0.1, 0.2, 0.3],
    })
    d = _tmp()
    info = tf.write_webdataset(
        t, d, rows_per_shard=2, key_column="id",
        text_columns=["caption"], binary_columns={"image": "jpg"},
    )
    assert info["num_shards"] == 2  # 2 + 1
    # inspect first shard
    members = []
    with tarfile.open(info["shards"][0]["path"]) as tar:
        members = tar.getnames()
    # record key 1: caption txt + jpg + json(meta: score)
    assert "1.caption.txt" in members
    assert "1.jpg" in members
    assert "1.json" in members


def test_webdataset_reads_back_bytes():
    t = pa.table({"id": [7], "image": [b"HELLO"], "caption": ["hi"]})
    d = _tmp()
    info = tf.write_webdataset(t, d, key_column="id", text_columns=["caption"], binary_columns={"image": "bin"})
    with tarfile.open(info["shards"][0]["path"]) as tar:
        data = tar.extractfile("7.bin").read()
        cap = tar.extractfile("7.caption.txt").read().decode()
    assert data == b"HELLO"
    assert cap == "hi"


def test_mds_index():
    t = pa.table({"x": list(range(30)), "y": ["a"] * 30})
    d = _tmp()
    info = tf.write_mds(t, d, rows_per_shard=10)
    assert info["num_shards"] == 3
    with open(os.path.join(d, "index.json")) as fh:
        idx = json.load(fh)
    assert idx["rows"] == 30
    assert set(idx["columns"].keys()) == {"x", "y"}
    assert len(idx["shards"]) == 3
