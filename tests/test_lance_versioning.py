"""Git-like data versioning on Lance: log / time-travel checkout / tags /
restore(rollback). Lance commits are versions; jude exposes a unified API."""
import os
import tempfile

import pyarrow as pa
import pytest

pytest.importorskip("lance")
import jude


@pytest.fixture
def versioned():
    p = os.path.join(tempfile.mkdtemp(), "h.lance")
    c = jude.connect()
    c.from_arrow(pa.table({"x": [1, 2, 3]})).write_lance(p, mode="create")    # v1
    c.from_arrow(pa.table({"x": [4, 5]})).write_lance(p, mode="append")       # v2
    return c, p


def test_version_log(versioned):
    c, p = versioned
    assert [r[0] for r in c.lance_versions(p).fetchall()] == [1, 2]


def test_time_travel_checkout(versioned):
    c, p = versioned
    assert len(jude.read_lance(p).fetchall()) == 5           # latest
    assert len(jude.read_lance(p, version=1).fetchall()) == 3  # past version


def test_tags(versioned):
    c, p = versioned
    c.lance_tag(p, "stable", 1)
    assert c.lance_tags(p).fetchall() == [("stable", 1)]
    assert len(jude.read_lance(p, version="stable").fetchall()) == 3


def test_restore_rollback_preserves_history(versioned):
    c, p = versioned
    meta = c.lance_restore(p, 1)
    assert meta["version"] == 3            # rollback is a new commit
    assert len(jude.read_lance(p).fetchall()) == 3   # state rolled back
    # history preserved: v2 still readable
    assert len(jude.read_lance(p, version=2).fetchall()) == 5
