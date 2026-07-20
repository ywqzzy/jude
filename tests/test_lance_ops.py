"""Lance high-value ops: FTS/hybrid, take/sample, upsert/add_columns, compact."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

import jude
from jude import lance as jl

lance = pytest.importorskip("lance")


def _dataset(rows=20):
    d = tempfile.mkdtemp(prefix="jude_lance_")
    path = d + "/ds"
    t = pa.table({
        "id": list(range(rows)),
        "text": [f"document number {i} about " + ("cats" if i % 2 else "dogs") for i in range(rows)],
        "score": [float(i) for i in range(rows)],
    })
    jude._lance.write(t, path, mode="create")
    return path


def test_take_point_access():
    path = _dataset(20)
    out = jl.take(path, [0, 5, 10], columns=["id", "score"])
    assert out.column("id").to_pylist() == [0, 5, 10]
    assert set(out.column_names) == {"id", "score"}


def test_sample():
    path = _dataset(50)
    out = jl.sample(path, 10)
    assert out.num_rows == 10


def test_add_columns_no_rewrite():
    path = _dataset(10)
    jl.add_columns(path, {"score2": "score * 2"})
    out = jude._lance.read_table(path, columns=["id", "score2"])
    assert "score2" in out.column_names
    d = dict(zip(out.column("id").to_pylist(), out.column("score2").to_pylist()))
    assert d[3] == 6.0


def test_merge_insert_upsert():
    path = _dataset(5)
    # update id 2, insert id 99
    new = pa.table({"id": [2, 99], "text": ["updated", "new one"], "score": [999.0, 100.0]})
    jl.merge_insert(path, new, on="id")
    out = jude._lance.read_table(path)
    d = dict(zip(out.column("id").to_pylist(), out.column("score").to_pylist()))
    assert d[2] == 999.0  # updated
    assert d[99] == 100.0  # inserted
    assert out.num_rows == 6  # 5 + 1 new


def test_delete_rows():
    path = _dataset(10)
    jl.delete(path, "id < 3")
    out = jude._lance.read_table(path, columns=["id"])
    assert min(out.column("id").to_pylist()) == 3
    assert out.num_rows == 7


def test_compact_runs():
    path = _dataset(10)
    info = jl.compact(path)  # should not error even on a small dataset
    assert info["path"] == path


def test_fts_search():
    path = _dataset(20)
    try:
        jl.create_fts_index(path, "text")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"FTS index unavailable: {e}")
    out = jl.full_text_search(path, "text", "cats", k=5)
    # every returned doc mentions cats (odd ids)
    assert out.num_rows > 0
    assert all("cats" in t for t in out.column("text").to_pylist())
