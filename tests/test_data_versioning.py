"""Lance git-like data versioning: time-travel, tags, branches, restore, and
incremental maintenance (add_columns/merge_insert/delete). Audit blind spot:
versioning had no behavioral tests.
"""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

from jude import _lance

lance = pytest.importorskip("lance")


def _ds():
    p = tempfile.mkdtemp() + "/ds"
    _lance.write(pa.table({"id": [1, 2, 3], "text": ["a", "b", "c"]}), p, mode="create")
    return p


def test_time_travel_reads_old_version():
    p = _ds()
    v0 = lance.dataset(p).version
    _lance.write(pa.table({"id": [4, 5], "text": ["d", "e"]}), p, mode="append")
    # latest sees 5 rows; version v0 still sees the original 3
    assert _lance.read_table(p).num_rows == 5
    assert _lance.read_table(p, version=v0).num_rows == 3


def test_list_versions_grows():
    p = _ds()
    n0 = _lance.list_versions(p).num_rows
    _lance.write(pa.table({"id": [9], "text": ["z"]}), p, mode="append")
    assert _lance.list_versions(p).num_rows > n0


def test_tag_and_read_by_tag():
    p = _ds()
    v0 = lance.dataset(p).version
    _lance.create_tag(p, "golden", v0)
    _lance.write(pa.table({"id": [7], "text": ["g"]}), p, mode="append")
    tags = _lance.list_tags(p).column("tag").to_pylist()
    assert "golden" in tags
    assert _lance.read_table(p, version="golden").num_rows == 3  # tag pins the old snapshot


def test_restore_rolls_back():
    p = _ds()
    v0 = lance.dataset(p).version
    _lance.write(pa.table({"id": [8], "text": ["h"]}), p, mode="append")
    assert _lance.read_table(p).num_rows == 4
    _lance.restore(p, v0)                        # roll back to the 3-row snapshot
    assert _lance.read_table(p).num_rows == 3


def test_delete_and_merge_insert():
    p = _ds()
    _lance.delete(p, "id = 2")
    ids = set(_lance.read_table(p).column("id").to_pylist())
    assert ids == {1, 3}
    # upsert: update id=1, insert id=42
    _lance.merge_insert(p, pa.table({"id": [1, 42], "text": ["A", "new"]}), on="id")
    rows = {r["id"]: r["text"] for r in _lance.read_table(p).to_pylist()}
    assert rows[1] == "A" and rows[42] == "new" and rows[3] == "c"


def test_add_columns_no_rewrite():
    p = _ds()
    _lance.add_columns(p, {"n": "length(text)"})
    t = _lance.read_table(p)
    assert "n" in t.column_names
    assert t.num_rows == 3


def test_take_and_sample():
    p = _ds()
    taken = _lance.take(p, [0, 2], columns=["id"])
    assert taken.column("id").to_pylist() == [1, 3]
    s = _lance.sample(p, 2)
    assert s.num_rows == 2
